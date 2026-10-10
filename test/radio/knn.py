import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import calculate_radio_mae
from generate.full_pipeline import KNN_K_VALUES, knn_model_path

def test_knn_radio():
    print("Testing Radio KNN...")
    
    # Load Validation and Test Data
    data = {name: load_mdt_data(f'data/{name}.csv', mode='radio') for name in ('validation', 'test')}
    
    print(f"{'K':<5}{'Validation MAE (dB)':<22}{'Test MAE (dB)':<15}")
    for k in KNN_K_VALUES:
        if not os.path.exists(knn_model_path(k)):
            print(f"Error: Model {knn_model_path(k)} not found.")
            continue
        knn = joblib.load(knn_model_path(k))
        
        # Predict
        maes = [calculate_radio_mae(rsrps.numpy(), knn.predict(coords.numpy())) for coords, rsrps in data.values()]
        print(f"{k:<5}{maes[0]:<22.2f}{maes[1]:<15.2f}")

if __name__ == "__main__":
    test_knn_radio()
