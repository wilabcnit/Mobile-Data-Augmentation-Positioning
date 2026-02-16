import sys
import os
import joblib
from sklearn.neighbors import KNeighborsRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data

def train_knn_radio():
    print("Training Radio KNN...")
    
    # 1. Load Data (Lat/Lon -> RSRP)
    # returns tensors, convert to numpy for sklearn
    coords, rsrps = load_mdt_data('data/train.csv', mode='radio')
    coords = coords.numpy()
    rsrps = rsrps.numpy()
    
    # 2. Train
    knn = KNeighborsRegressor(n_neighbors=2, weights='distance')
    knn.fit(coords, rsrps)
    
    # 3. Save
    os.makedirs('models/radio', exist_ok=True)
    joblib.dump(knn, 'models/radio/knn.pkl')
    print("Model saved to models/radio/knn.pkl")

if __name__ == "__main__":
    train_knn_radio()