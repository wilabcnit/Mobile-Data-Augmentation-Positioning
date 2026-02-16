import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import calculate_radio_mae

def test_rf_radio():
    print("Testing Radio Random Forest...")
    
    if not os.path.exists('models/radio/rf.pkl'):
        print("Error: Model not found.")
        return

    rf = joblib.load('models/radio/rf.pkl')
    
    coords, real_rsrps = load_mdt_data('data/test.csv', mode='radio')
    
    pred_rsrps = rf.predict(coords.numpy())
    
    mae = calculate_radio_mae(real_rsrps.numpy(), pred_rsrps)
    print(f"RF Radio MAE: {mae:.4f} dBm")

if __name__ == "__main__":
    test_rf_radio()