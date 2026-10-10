import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Scenario: city center around Bologna (WGS84), 3GPP TR 38.901 UMa NLOS radio model
REF_LAT, REF_LON = 44.4938, 11.3426   # reference point (Piazza Maggiore)
EARTH_RADIUS = 6371008.8              # m

FC = 1.8                  # carrier frequency (GHz)
P_RS = 30.0               # reference-signal power per resource element incl. antenna gain (dBm)
H_BS, H_UT = 25.0, 1.5    # base station / UE heights (m)
SF_STD = 7.8              # shadow fading std (dB), as in the manuscript (TR 38.901: UMa NLOS 6 dB, UMi NLOS 7.82 dB)
SF_DCORR = 50.0           # shadow fading decorrelation distance (m), TR 38.901 Table 7.5-6, UMa NLOS

MAX_REPORTED_CELLS = 10   # strongest cells in an MDT measurement report
DETECTION_THRESHOLD = -124.0
UNSERVED_RSRP = -160      # RSRP stored for a cell that is not served

def pathloss_uma_nlos(d2d):
    """3GPP TR 38.901 UMa NLOS pathloss (Table 7.4.1-1), distance in metres."""
    d3d = np.sqrt(d2d**2 + (H_BS - H_UT)**2)
    d_bp = 4 * (H_BS - 1.0) * (H_UT - 1.0) * FC * 1e9 / 3e8
    pl_los = np.where(d2d <= d_bp,
                      28.0 + 22 * np.log10(d3d) + 20 * np.log10(FC),
                      28.0 + 40 * np.log10(d3d) + 20 * np.log10(FC) - 9 * np.log10(d_bp**2 + (H_BS - H_UT)**2))
    pl_nlos = 13.54 + 39.08 * np.log10(d3d) + 20 * np.log10(FC) - 0.6 * (H_UT - 1.5)
    return np.maximum(pl_los, pl_nlos)

def shadowing_map(rng, n=1024, dx=10.0):
    """Unit-variance Gaussian field (n x n grid, dx m) with exponential autocorrelation exp(-d / SF_DCORR)."""
    f = np.fft.fftfreq(n, d=dx)
    fx, fy = np.meshgrid(f, f, indexing='ij')
    spectrum = SF_DCORR**2 / (1.0 + (2 * np.pi * SF_DCORR)**2 * (fx**2 + fy**2))**1.5
    field = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, n))) * np.sqrt(spectrum)))
    return (field - field.mean()) / field.std()

def sample_map(field, origin, points, dx=10.0):
    """Bilinear interpolation of the field (grid starting at origin, m) at the given points; 0 outside."""
    n = field.shape[0]
    u, v = (points[:, 0] - origin[0]) / dx, (points[:, 1] - origin[1]) / dx
    inside = (u >= 0) & (u < n - 1) & (v >= 0) & (v < n - 1)
    out = np.zeros(len(points))
    i, j = np.floor(u[inside]).astype(int), np.floor(v[inside]).astype(int)
    a, b = u[inside] - i, v[inside] - j
    out[inside] = ((1 - a) * (1 - b) * field[i, j] + a * (1 - b) * field[i + 1, j]
                   + (1 - a) * b * field[i, j + 1] + a * b * field[i + 1, j + 1])
    return out

def generate_mdt_data(n_samples=10000, n_cells=35, seed=42):
    """
    Generates synthetic MDT data with spatial hotspots and a 3GPP radio model.
    """
    rng = np.random.default_rng(seed)

    # 1. Spatial Generation (Hotspots to mimic user clusters), local East/North metres
    n_clusters = 6
    centroids = rng.uniform(-600.0, 600.0, (n_clusters, 2))
    ue = centroids[rng.integers(0, n_clusters, n_samples)] + rng.normal(0, 80.0, (n_samples, 2))

    # 2. Radio Generation (RSRP = P_RS - pathloss + spatially correlated shadow fading)
    cell_locs = rng.uniform(-1000.0, 1000.0, (n_cells, 2))
    rsrp = np.empty((n_samples, n_cells))
    for i, cell in enumerate(cell_locs):
        dist = np.maximum(np.linalg.norm(ue - cell, axis=1), 10.0)   # avoid log(0)
        shadowing = SF_STD * sample_map(shadowing_map(rng), cell - 1024 * 10.0 / 2, ue)
        rsrp[:, i] = P_RS - pathloss_uma_nlos(dist) + shadowing

    # 3. MDT reporting: range [-140, -44] dBm, 1 dB steps, only detected and strongest cells
    rsrp = np.round(np.clip(rsrp, -140, -44))
    rank = np.argsort(np.argsort(-rsrp, axis=1), axis=1)
    served = (rsrp >= DETECTION_THRESHOLD) & (rank < MAX_REPORTED_CELLS)
    rsrp = np.where(served, rsrp, UNSERVED_RSRP).astype(int)

    # WGS84 coordinates
    lat = REF_LAT + np.degrees(ue[:, 1] / EARTH_RADIUS)
    lon = REF_LON + np.degrees(ue[:, 0] / (EARTH_RADIUS * np.cos(np.radians(REF_LAT))))
    df = pd.DataFrame({'LATITUDE': lat, 'LONGITUDE': lon})
    return pd.concat([df, pd.DataFrame(rsrp, columns=[f'RSRP_{i+1}' for i in range(n_cells)])], axis=1)

def main():
    # Create directories
    os.makedirs('data', exist_ok=True)

    print("Generating Mock MDT Data...")
    full_df = generate_mdt_data(n_samples=10000, n_cells=35)

    # Split into Train (70%), Validation (10%) and Test (20%)
    train_val_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42)

    # Save
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/validation.csv', index=False)   # hyperparameter selection only
    test_df.to_csv('data/test.csv', index=False)

    print(f"Saved data/train.csv ({len(train_df)} samples)")
    print(f"Saved data/validation.csv ({len(val_df)} samples)")
    print(f"Saved data/test.csv ({len(test_df)} samples)")

if __name__ == "__main__":
    main()
