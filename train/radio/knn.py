import sys
import os
import joblib
from sklearn.neighbors import KNeighborsRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data

def train_knn_radio(k=1):
    print(f"Training Radio KNN (K={k})...")
    
    # 1. Load Data (Lat/Lon -> RSRP)
    # returns tensors, convert to numpy for sklearn
    coords, rsrps = load_mdt_data('data/train.csv', mode='radio')
    coords = coords.numpy()
    rsrps = rsrps.numpy()
    
    # 2. Train
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn.fit(coords, rsrps)
    
    # 3. Save
    os.makedirs('models/radio', exist_ok=True)
    save_path = f'models/radio/knn_k{k}.pkl' if k != 1 else 'models/radio/knn.pkl'
    joblib.dump(knn, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_knn_radio(k=1)
