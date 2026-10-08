import sys
import os
import joblib
from sklearn.neighbors import KNeighborsRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.geo import GeoRegressor

def train_knn_radio(k, save_path=None):
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
    save_path = save_path or f'models/radio/knn_k{k}.pkl'
    joblib.dump(knn, save_path)
    print(f"Model saved to {save_path}")
    return save_path

if __name__ == "__main__":
    # K=1 as in the paper; train/radio/knn_k_ablation.py reports K = 1..5 on the validation set
    train_knn_radio(k=1, save_path='models/radio/knn.pkl')
