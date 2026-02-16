import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def generate_mdt_data(n_samples=5000, n_cells=30, grid_size=1000):
    """
    Generates synthetic MDT data with spatial clusters and radio pathloss.
    """
    np.random.seed(42)
    
    # 1. Spatial Generation (Clusters to mimic user hotspots)
    n_clusters = 5
    centroids = np.random.uniform(0, grid_size, (n_clusters, 2))
    
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
        
    df = pd.DataFrame({
        'LATITUDE': latitudes,
        'LONGITUDE': longitudes
    })

    # 2. Radio Generation (RSRP based on distance to cell towers)
    # Generate random cell tower locations
    cell_locs = np.random.uniform(0, grid_size, (n_cells, 2))
    
    # Pathloss constants
    P_tx = 46  # dBm (Macro cell)
    alpha = 3.5 # Path loss exponent
    
    for i in range(n_cells):
        cell_x, cell_y = cell_locs[i]
        
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

def main():
    # Create directories
    os.makedirs('data', exist_ok=True)
    
    print("Generating Mock MDT Data...")
    full_df = generate_mdt_data(n_samples=10000, n_cells=30)
    
    # Split into Train (80%) and Test (20%)
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42)
    
    # Save
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print(f"Saved data/train.csv ({len(train_df)} samples)")
    print(f"Saved data/test.csv ({len(test_df)} samples)")
    print(f"Columns: {list(train_df.columns[:5])} ... {list(train_df.columns[-1])}")

if __name__ == "__main__":
    main()