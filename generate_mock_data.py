import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def sample_positions(n_samples, centroids, grid_size):
    """
    Samples user positions around the hotspot centroids (clusters mimic user hotspots).
    """
    n_clusters = len(centroids)
    latitudes = []
    longitudes = []

    for _ in range(n_samples):
        # Pick a random cluster center
        cluster_idx = np.random.randint(0, n_clusters)
        center = centroids[cluster_idx]

        # Add Gaussian noise to create the hotspot
        loc = center + np.random.normal(0, grid_size * 0.05, 2)
        latitudes.append(loc[0])
        longitudes.append(loc[1])

    return pd.DataFrame({
        'LATITUDE': latitudes,
        'LONGITUDE': longitudes
    })

def add_radio(df, cell_locs):
    """
    Adds RSRP columns (log-distance pathloss + shadowing) for the given cell tower locations.
    """
    n_samples = len(df)

    # Pathloss constants
    P_tx = 46  # dBm (Macro cell)
    alpha = 3.5 # Path loss exponent

    for i, (cell_x, cell_y) in enumerate(cell_locs):
        # Calculate distance from user to this cell
        dist = np.sqrt((df['LATITUDE'] - cell_x)**2 + (df['LONGITUDE'] - cell_y)**2)
        dist = np.maximum(dist, 10) # Avoid log(0)

        # Log-distance path loss model + Shadowing (Normal noise)
        # PL(d) = PL(d0) + 10*alpha*log10(d) + X_sigma
        shadowing = np.random.normal(0, 6, n_samples) # 6dB std dev
        path_loss = 120 + 10 * alpha * np.log10(dist / 1000) # Simple model

        rsrp = P_tx - path_loss + shadowing

        # Clip RSRP to realistic values (-140 to -40 dBm)
        rsrp = np.clip(rsrp, -140, -40)

        df[f'RSRP_{i+1}'] = rsrp

    return df

def generate_mdt_data(n_samples=5000, n_val_samples=1000, n_cells=30, grid_size=1000, n_clusters=5, seed=42):
    """
    Generates synthetic MDT data with spatial clusters and radio pathloss.

    The main set and the validation set share the same scenario (hotspot centroids and
    cell tower locations), but the validation set uses new, independently drawn users,
    so it never overlaps with the train/test data.

    Returns:
        (pd.DataFrame, pd.DataFrame): main set (to be split into train/test), validation set.
    """
    np.random.seed(seed)

    # 1. Scenario + spatial generation
    centroids = np.random.uniform(0, grid_size, (n_clusters, 2))
    df = sample_positions(n_samples, centroids, grid_size)

    # 2. Radio generation (RSRP based on distance to cell towers)
    cell_locs = np.random.uniform(0, grid_size, (n_cells, 2))
    df = add_radio(df, cell_locs)

    # 3. Validation set: new users drawn from the SAME scenario
    val_df = sample_positions(n_val_samples, centroids, grid_size)
    val_df = add_radio(val_df, cell_locs)

    return df, val_df

def main():
    # Create directories
    os.makedirs('data', exist_ok=True)

    print("Generating Mock MDT Data...")
    full_df, val_df = generate_mdt_data(n_samples=10000, n_val_samples=1000, n_cells=30)

    # Split into Train (80%) and Test (20%)
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42)

    # Validation set used to select the hyperparameters (never the test set)

    # Save
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/validation.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    print(f"Saved data/train.csv ({len(train_df)} samples)")
    print(f"Saved data/validation.csv ({len(val_df)} samples)")
    print(f"Saved data/test.csv ({len(test_df)} samples)")
    print(f"Columns: {list(train_df.columns[:5])} ... {train_df.columns[-1]}")

if __name__ == "__main__":
    main()
