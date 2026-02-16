import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import calculate_radio_mae

def test_knn_radio():
    print("Testing Radio KNN...")
    
    if not os.path.exists('models/radio/knn.pkl'):
        print("Error: Model not found.")
        return
        
    knn = joblib.load('models/radio/knn.pkl')
    
    # Load Test Data
    coords, real_rsrps = load_mdt_data('data/test.csv', mode='radio')
    coords = coords.numpy()
    real_rsrps = real_rsrps.numpy()
    
    # Predict
    pred_rsrps = knn.predict(coords)
    
    mae = calculate_radio_mae(real_rsrps, pred_rsrps)
    print(f"KNN Radio MAE: {mae:.4f} dBm")

if __name__ == "__main__":
    test_knn_radio()