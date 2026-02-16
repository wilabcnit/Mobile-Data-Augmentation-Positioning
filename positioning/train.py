import sys
import os
import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def train_positioning(train_data_path, model_name='wknn_baseline'):
    """
    Trains a wKNN positioning model using the specified dataset.
    
    Args:
        train_data_path (str): Path to the CSV file (Real or Synthetic MDT).
        model_name (str): Name to save the model as.
    """
    print(f"Training Positioning Model: {model_name}...")
    print(f"Using Data: {train_data_path}")
    
    # 1. Load Data
    if not os.path.exists(train_data_path):
        print(f"Error: Data file {train_data_path} not found.")
        return

    df = pd.read_csv(train_data_path)
    
    # Features: RSRP columns | Targets: Lat, Lon
    rsrp_cols = [c for c in df.columns if 'RSRP' in c]
    X = df[rsrp_cols].values
    y = df[['LATITUDE', 'LONGITUDE']].values
    
    # 2. Train wKNN (Weighted by inverse distance in signal space)
    wknn = KNeighborsRegressor(n_neighbors=3, weights='distance', metric='euclidean')
    wknn.fit(X, y)
    
    # 3. Save Model
    os.makedirs('models/positioning', exist_ok=True)
    save_path = f'models/positioning/{model_name}.pkl'
    joblib.dump(wknn, save_path)
    print(f"Positioning model saved to {save_path}")

if __name__ == "__main__":
    #train_positioning('data/train.csv', 'wknn_baseline')
    train_positioning('augm_data/synthetic_kde_knn.csv', 'wknn_kde_knn')