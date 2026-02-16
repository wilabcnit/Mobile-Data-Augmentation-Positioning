import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import calculate_radio_mae

def test_gpr_rq():
    print("Testing Radio GPR (RQ)...")
    
    if not os.path.exists('models/radio/gpr_rq.pkl'):
        print("Error: Model not found.")
        return

    gpr = joblib.load('models/radio/gpr_rq.pkl')
    
    coords, real_rsrps = load_mdt_data('data/test.csv', mode='radio')
    
    pred_rsrps = gpr.predict(coords.numpy())
    
    mae = calculate_radio_mae(real_rsrps.numpy(), pred_rsrps)
    print(f"GPR-RQ Radio MAE: {mae:.4f} dBm")

if __name__ == "__main__":
    test_gpr_rq()