"""
Generates a mock MDT dataset for one urban scenario (city center around Bologna, WGS84
coordinates) with a 3GPP TR 38.901-style radio model:

* Pathloss: UMa NLOS, 3GPP TR 38.901 Table 7.4.1-1.
* Shadow fading: log-normal with standard deviation SHADOW_FADING_STD_DB, spatially
  correlated with an exponential autocorrelation and the TR 38.901 Table 7.5-6 UMa NLOS
  decorrelation distance (50 m).
* MDT reporting: RSRP clipped to the reporting range [-140, -44] dBm (TS 36.133) and
  quantized to 1 dB; a UE reports only the cells it detects (RSRP >= DETECTION_THRESHOLD_DBM),
  at most MAX_REPORTED_CELLS strongest ones. The other cells are "not served" and stored
  as UNSERVED_RSRP.

The dataset is split 70/10/20 into train/validation/test.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
N_UE = 10000

# Local East-North frame (metres) around the reference point, converted to WGS84
REF_LAT, REF_LON = 44.4938, 11.3426   # Bologna, Piazza Maggiore
EARTH_RADIUS_M = 6371008.8

CARRIER_FREQ_GHZ = 1.8      # LTE band 3
RS_EIRP_DBM = 30.0          # reference-signal power per resource element incl. antenna gain
H_BS, H_UT = 25.0, 1.5      # base station / UE heights (m), UMa

# Shadow fading standard deviation (dB), as used in the manuscript.
# NOTE: 3GPP TR 38.901 Table 7.4.1-1 gives 6 dB for UMa NLOS (7.82 dB is UMi-Street Canyon NLOS).
SHADOW_FADING_STD_DB = 7.8
DECORRELATION_DIST_M = 50.0   # 3GPP TR 38.901 Table 7.5-6, UMa NLOS

RSRP_MIN_DBM, RSRP_MAX_DBM = -140.0, -44.0   # RSRP reporting range (TS 36.133)
DETECTION_THRESHOLD_DBM = -124.0             # weakest detectable cell
MAX_REPORTED_CELLS = 8                        # strongest cells in an MDT measurement report
UNSERVED_RSRP = -160                          # value stored for a cell that is not served

# Scenario layout in the local frame (metres)
N_CELLS = 35
CELL_AREA_HALF_SIDE = 1000.0   # cells placed uniformly in a 2 km x 2 km area
N_HOTSPOTS = 6                 # squares / streets where users concentrate
HOTSPOT_AREA_HALF_SIDE = 600.0
HOTSPOT_STD = 80.0

# ----------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------
def local_to_wgs84(x, y):
    """Local East (x) / North (y) offsets in metres -> (latitude, longitude) in degrees."""
    lat = REF_LAT + np.degrees(y / EARTH_RADIUS_M)
    lon = REF_LON + np.degrees(x / (EARTH_RADIUS_M * np.cos(np.radians(REF_LAT))))
    return lat, lon

def sample_ue_positions(n, rng):
    """UE positions (metres, local frame) around hotspots."""
    hotspots = rng.uniform(-HOTSPOT_AREA_HALF_SIDE, HOTSPOT_AREA_HALF_SIDE, (N_HOTSPOTS, 2))
    idx = rng.integers(0, N_HOTSPOTS, n)
    return hotspots[idx] + rng.normal(0, HOTSPOT_STD, (n, 2))

# ----------------------------------------------------------------------------------
# 3GPP TR 38.901 UMa NLOS pathloss (Table 7.4.1-1), fc in GHz, distances in metres
# ----------------------------------------------------------------------------------
def pathloss_uma_nlos(d2d, fc=CARRIER_FREQ_GHZ, h_bs=H_BS, h_ut=H_UT):
    d3d = np.sqrt(d2d**2 + (h_bs - h_ut)**2)
    d_bp = 4 * (h_bs - 1.0) * (h_ut - 1.0) * fc * 1e9 / 3e8
    pl_los = np.where(
        d2d <= d_bp,
        28.0 + 22 * np.log10(d3d) + 20 * np.log10(fc),
        28.0 + 40 * np.log10(d3d) + 20 * np.log10(fc) - 9 * np.log10(d_bp**2 + (h_bs - h_ut)**2))
    pl_nlos = 13.54 + 39.08 * np.log10(d3d) + 20 * np.log10(fc) - 0.6 * (h_ut - 1.5)
    return np.maximum(pl_los, pl_nlos)

# ----------------------------------------------------------------------------------
# Spatially correlated shadow fading
# ----------------------------------------------------------------------------------
def correlated_field(rng, d_corr, n=1024, dx=10.0):
    """
    Zero-mean, unit-variance Gaussian random field on an n x n grid (spacing dx metres)
    with exponential autocorrelation exp(-d / d_corr), by spectral synthesis.
    """
    f = np.fft.fftfreq(n, d=dx)
    fx, fy = np.meshgrid(f, f, indexing='ij')
    spectrum = d_corr**2 / (1.0 + (2 * np.pi * d_corr)**2 * (fx**2 + fy**2))**1.5
    field = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, n))) * np.sqrt(spectrum)))
    return (field - field.mean()) / field.std()

def sample_field(field, origin, points, dx=10.0):
    """Bilinear interpolation of a field whose grid starts at origin (metres); 0 outside the grid."""
    n = field.shape[0]
    u = (points[:, 0] - origin[0]) / dx
    v = (points[:, 1] - origin[1]) / dx
    inside = (u >= 0) & (u < n - 1) & (v >= 0) & (v < n - 1)
    out = np.zeros(len(points))
    i, j = np.floor(u[inside]).astype(int), np.floor(v[inside]).astype(int)
    a, b = u[inside] - i, v[inside] - j
    out[inside] = ((1 - a) * (1 - b) * field[i, j] + a * (1 - b) * field[i + 1, j]
                   + (1 - a) * b * field[i, j + 1] + a * b * field[i + 1, j + 1])
    return out

# ----------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------
def generate_mdt_data(n_ue=N_UE, seed=SEED):
    rng = np.random.default_rng(seed)

    # 1. Users and cells
    ue_xy = sample_ue_positions(n_ue, rng)
    cell_xy = rng.uniform(-CELL_AREA_HALF_SIDE, CELL_AREA_HALF_SIDE, (N_CELLS, 2))

    # 2. RSRP = RS EIRP - pathloss + correlated shadow fading (one field per cell)
    n_grid, dx = 1024, 10.0
    rsrp = np.empty((n_ue, N_CELLS))
    for c, cell in enumerate(cell_xy):
        d2d = np.maximum(np.linalg.norm(ue_xy - cell, axis=1), 10.0)
        field = correlated_field(rng, DECORRELATION_DIST_M, n_grid, dx)
        shadowing = SHADOW_FADING_STD_DB * sample_field(field, cell - n_grid * dx / 2, ue_xy, dx)
        rsrp[:, c] = RS_EIRP_DBM - pathloss_uma_nlos(d2d) + shadowing

    # 3. MDT reporting: reporting range, 1 dB resolution, detected and strongest cells only
    rsrp = np.round(np.clip(rsrp, RSRP_MIN_DBM, RSRP_MAX_DBM))
    rank = np.argsort(np.argsort(-rsrp, axis=1), axis=1)
    served = (rsrp >= DETECTION_THRESHOLD_DBM) & (rank < MAX_REPORTED_CELLS)
    rsrp = np.where(served, rsrp, UNSERVED_RSRP)

    lat, lon = local_to_wgs84(ue_xy[:, 0], ue_xy[:, 1])
    df = pd.DataFrame({'LATITUDE': lat, 'LONGITUDE': lon})
    rsrp_df = pd.DataFrame(rsrp.astype(int), columns=[f'RSRP_{c+1}' for c in range(N_CELLS)])
    return pd.concat([df, rsrp_df], axis=1)

def main():
    os.makedirs('data', exist_ok=True)

    print("Generating Mock MDT Data...")
    full_df = generate_mdt_data()

    # 70/10/20 split: 20% test first, then 1/8 of the remaining 80% (= 10%) as validation
    train_val_df, test_df = train_test_split(full_df, test_size=0.2, random_state=SEED)
    train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=SEED)

    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/validation.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    n = len(full_df)
    for name, part in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        print(f"Saved data/{name}.csv ({len(part)} samples, {len(part) / n:.0%})")
    served = (full_df.filter(like='RSRP').values != UNSERVED_RSRP).sum(axis=1)
    print(f"Cells: {N_CELLS} | served cells per report: mean {served.mean():.1f}, min {served.min()}")

if __name__ == "__main__":
    main()
