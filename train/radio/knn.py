import sys
import os
import joblib
from sklearn.neighbors import KNeighborsRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.geo import GeoRegressor
from generate.full_pipeline import KNN_K_VALUES, knn_model_path

def train_knn_radio(k):
    print(f"Training Radio KNN (K={k})...")
    
    # 1. Load Data (Lat/Lon -> RSRP)
    # returns tensors, convert to numpy for sklearn
    coords, rsrps = load_mdt_data('data/train.csv', mode='radio')
    coords = coords.numpy()
    rsrps = rsrps.numpy()
    
    # 2. Train
    # Neighbours are searched in metres (GeoRegressor), not in degrees
    knn = GeoRegressor(KNeighborsRegressor(n_neighbors=k, weights='distance'))
    knn.fit(coords, rsrps)
    
    # 3. Save
    os.makedirs('models/radio', exist_ok=True)
    save_path = knn_model_path(k)
    joblib.dump(knn, save_path)
    print(f"Model saved to {save_path}")
    return save_path

if __name__ == "__main__":
    for k in KNN_K_VALUES:
        train_knn_radio(k)
