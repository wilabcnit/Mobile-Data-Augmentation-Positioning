"""
Generates a mock MDT dataset in WGS84 coordinates for four regions (City center, Stadium,
Airport, Highway) around Bologna, with a 3GPP TR 38.901-style radio model:

* Pathloss: UMa NLOS (City center, Stadium, Airport) and RMa NLOS (Highway),
  3GPP TR 38.901 Table 7.4.1-1.
* Shadow fading: log-normal, region-specific standard deviation (see SHADOW_FADING_STD_DB),
  spatially correlated with an exponential autocorrelation and the TR 38.901 Table 7.5-6
  decorrelation distances (UMa NLOS 50 m, RMa NLOS 120 m).
* MDT reporting: RSRP clipped to the reporting range [-140, -44] dBm (TS 36.133) and
  quantized to 1 dB; a UE reports only the cells it detects (RSRP >= DETECTION_THRESHOLD_DBM),
  at most MAX_REPORTED_CELLS strongest ones. The other cells are "not served" and stored
  as UNSERVED_RSRP.

The dataset is split 70/10/20 into train/validation/test (stratified by region).
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

# Local East-North frame (metres) around the reference point, converted to WGS84
REF_LAT, REF_LON = 44.4938, 11.3426   # Bologna, Piazza Maggiore
EARTH_RADIUS_M = 6371008.8

CARRIER_FREQ_GHZ = 1.8      # LTE band 3
RS_EIRP_DBM = 30.0          # reference-signal power per resource element incl. antenna gain
H_UT = 1.5                  # UE height (m)

# Shadow fading standard deviation (dB) per region, as used in the manuscript.
# NOTE: 3GPP TR 38.901 Table 7.4.1-1 gives 6 dB for UMa NLOS (7.82 dB is UMi-Street Canyon
# NLOS) and 8 dB for RMa NLOS.
SHADOW_FADING_STD_DB = {'city_center': 7.8, 'stadium': 7.8, 'airport': 7.8, 'highway': 8.0}

# Shadow fading decorrelation distance (m), 3GPP TR 38.901 Table 7.5-6 (NLOS)
DECORRELATION_DIST_M = {'UMa': 50.0, 'RMa': 120.0}

RSRP_MIN_DBM, RSRP_MAX_DBM = -140.0, -44.0   # RSRP reporting range (TS 36.133)
DETECTION_THRESHOLD_DBM = -124.0             # weakest detectable cell
MAX_REPORTED_CELLS = 8                        # strongest cells in an MDT measurement report
UNSERVED_RSRP = -160                          # value stored for a cell that is not served

# Region layout in the local frame (metres). Cell counts follow the real PCI counts
# (140/104/209/52) scaled by 1/4.
REGIONS = {
    'city_center': dict(scenario='UMa', n_ue=3500, n_cells=35, center=(0.0, 0.0), cell_spread=1000.0),
    'stadium':     dict(scenario='UMa', n_ue=2000, n_cells=26, center=(-2630.0, -190.0), cell_spread=700.0),
    'airport':     dict(scenario='UMa', n_ue=3000, n_cells=52, center=(-4280.0, 4630.0), cell_spread=1200.0),
    'highway':     dict(scenario='RMa', n_ue=1500, n_cells=13, start=(-1000.0, 5500.0), end=(9000.0, 7500.0)),
}

# ----------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------
def local_to_wgs84(x, y):
    """Local East (x) / North (y) offsets in metres -> (latitude, longitude) in degrees."""
    lat = REF_LAT + np.degrees(y / EARTH_RADIUS_M)
    lon = REF_LON + np.degrees(x / (EARTH_RADIUS_M * np.cos(np.radians(REF_LAT))))
    return lat, lon

def sample_ue_positions(name, cfg, n, rng):
    """UE positions (metres, local frame) for one region."""
    if name == 'highway':
        start, end = np.array(cfg['start']), np.array(cfg['end'])
        direction = (end - start) / np.linalg.norm(end - start)
        normal = np.array([-direction[1], direction[0]])
        s = rng.uniform(0, 1, n)[:, None]
        lateral = rng.normal(0, 8.0, n)[:, None]           # carriageway width
        return start + s * (end - start) + lateral * normal

    cx, cy = cfg['center']
    if name == 'stadium':
        # Crowd around the stands + two parking areas
        n_ring = int(0.7 * n)
        angle = rng.uniform(0, 2 * np.pi, n_ring)
        radius = rng.normal(120.0, 25.0, n_ring)
        ring = np.c_[cx + radius * np.cos(angle), cy + radius * np.sin(angle)]
        parking_centers = np.array([[cx + 300.0, cy + 150.0], [cx - 250.0, cy - 280.0]])
        idx = rng.integers(0, 2, n - n_ring)
        parking = parking_centers[idx] + rng.normal(0, 60.0, (n - n_ring, 2))
        return np.vstack([ring, parking])

    # City center / Airport: hotspots (squares, streets, terminals, gates)
    n_hotspots = 6 if name == 'city_center' else 4
    hotspots = np.c_[cx, cy] + rng.uniform(-600.0, 600.0, (n_hotspots, 2))
    idx = rng.integers(0, n_hotspots, n)
    return hotspots[idx] + rng.normal(0, 80.0, (n, 2))

def place_cells(name, cfg, rng):
    """Cell (site) positions in metres, local frame."""
    if name == 'highway':
        start, end = np.array(cfg['start']), np.array(cfg['end'])
        direction = (end - start) / np.linalg.norm(end - start)
        normal = np.array([-direction[1], direction[0]])
        s = (np.arange(cfg['n_cells']) + 0.5) / cfg['n_cells']
        offset = rng.uniform(50.0, 250.0, cfg['n_cells']) * rng.choice([-1, 1], cfg['n_cells'])
        return start + s[:, None] * (end - start) + offset[:, None] * normal
    return np.array(cfg['center']) + rng.uniform(-cfg['cell_spread'], cfg['cell_spread'], (cfg['n_cells'], 2))

# ----------------------------------------------------------------------------------
# 3GPP TR 38.901 pathloss (Table 7.4.1-1), fc in GHz, distances in metres
# ----------------------------------------------------------------------------------
def pathloss_uma_nlos(d2d, fc=CARRIER_FREQ_GHZ, h_bs=25.0, h_ut=H_UT):
    d3d = np.sqrt(d2d**2 + (h_bs - h_ut)**2)
    d_bp = 4 * (h_bs - 1.0) * (h_ut - 1.0) * fc * 1e9 / 3e8
    pl_los = np.where(
        d2d <= d_bp,
        28.0 + 22 * np.log10(d3d) + 20 * np.log10(fc),
        28.0 + 40 * np.log10(d3d) + 20 * np.log10(fc) - 9 * np.log10(d_bp**2 + (h_bs - h_ut)**2))
    pl_nlos = 13.54 + 39.08 * np.log10(d3d) + 20 * np.log10(fc) - 0.6 * (h_ut - 1.5)
    return np.maximum(pl_los, pl_nlos)

def pathloss_rma_nlos(d2d, fc=CARRIER_FREQ_GHZ, h_bs=35.0, h_ut=H_UT, h=5.0, w=20.0):
    d3d = np.sqrt(d2d**2 + (h_bs - h_ut)**2)
    d_bp = 2 * np.pi * h_bs * h_ut * fc * 1e9 / 3e8
    pl1 = lambda d: (20 * np.log10(40 * np.pi * d * fc / 3) + min(0.03 * h**1.72, 10) * np.log10(d)
                     - min(0.044 * h**1.72, 14.77) + 0.002 * np.log10(h) * d)
    pl_los = np.where(d2d <= d_bp, pl1(d3d), pl1(d_bp) + 40 * np.log10(d3d / d_bp))
    pl_nlos = (161.04 - 7.1 * np.log10(w) + 7.5 * np.log10(h)
               - (24.37 - 3.7 * (h / h_bs)**2) * np.log10(h_bs)
               + (43.42 - 3.1 * np.log10(h_bs)) * (np.log10(d3d) - 3)
               + 20 * np.log10(fc) - (3.2 * np.log10(11.75 * h_ut)**2 - 4.97))
    return np.maximum(pl_los, pl_nlos)

PATHLOSS = {'UMa': pathloss_uma_nlos, 'RMa': pathloss_rma_nlos}

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
def generate_mdt_data(seed=SEED):
    rng = np.random.default_rng(seed)

    # 1. Users and cells
    ue_xy, ue_region, cell_xy, cell_scenario = [], [], [], []
    for name, cfg in REGIONS.items():
        ue_xy.append(sample_ue_positions(name, cfg, cfg['n_ue'], rng))
        ue_region += [name] * cfg['n_ue']
        cells = place_cells(name, cfg, rng)
        cell_xy.append(cells)
        cell_scenario += [cfg['scenario']] * len(cells)
    ue_xy, cell_xy = np.vstack(ue_xy), np.vstack(cell_xy)
    ue_region = np.array(ue_region)
    ue_scenario = np.array([REGIONS[r]['scenario'] for r in ue_region])
    ue_sigma = np.array([SHADOW_FADING_STD_DB[r] for r in ue_region])

    # 2. RSRP = RS EIRP - pathloss(UE scenario) + correlated shadow fading
    n_grid, dx = 1024, 10.0
    rsrp = np.empty((len(ue_xy), len(cell_xy)))
    for c, (cell, scenario) in enumerate(zip(cell_xy, cell_scenario)):
        d2d = np.maximum(np.linalg.norm(ue_xy - cell, axis=1), 10.0)
        pathloss = np.where(ue_scenario == 'UMa', PATHLOSS['UMa'](d2d), PATHLOSS['RMa'](d2d))
        field = correlated_field(rng, DECORRELATION_DIST_M[scenario], n_grid, dx)
        shadowing = ue_sigma * sample_field(field, cell - n_grid * dx / 2, ue_xy, dx)
        rsrp[:, c] = RS_EIRP_DBM - pathloss + shadowing

    # 3. MDT reporting: reporting range, 1 dB resolution, detected and strongest cells only
    rsrp = np.round(np.clip(rsrp, RSRP_MIN_DBM, RSRP_MAX_DBM))
    rank = np.argsort(np.argsort(-rsrp, axis=1), axis=1)
    served = (rsrp >= DETECTION_THRESHOLD_DBM) & (rank < MAX_REPORTED_CELLS)
    rsrp = np.where(served, rsrp, UNSERVED_RSRP)

    lat, lon = local_to_wgs84(ue_xy[:, 0], ue_xy[:, 1])
    df = pd.DataFrame({'REGION': ue_region, 'LATITUDE': lat, 'LONGITUDE': lon})
    rsrp_df = pd.DataFrame(rsrp.astype(int), columns=[f'RSRP_{c+1}' for c in range(len(cell_xy))])
    return pd.concat([df, rsrp_df], axis=1)

def main():
    os.makedirs('data', exist_ok=True)

    print("Generating Mock MDT Data...")
    full_df = generate_mdt_data()

    # 70/10/20 split: 20% test first, then 1/8 of the remaining 80% (= 10%) as validation
    train_val_df, test_df = train_test_split(full_df, test_size=0.2, random_state=SEED, stratify=full_df['REGION'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=SEED, stratify=train_val_df['REGION'])

    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/validation.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    n = len(full_df)
    for name, part in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        print(f"Saved data/{name}.csv ({len(part)} samples, {len(part) / n:.0%})")
    served = (full_df.filter(like='RSRP').values != UNSERVED_RSRP).sum(axis=1)
    print(f"Cells: {full_df.filter(like='RSRP').shape[1]} | served cells per report: mean {served.mean():.1f}, min {served.min()}")

if __name__ == "__main__":
    main()
