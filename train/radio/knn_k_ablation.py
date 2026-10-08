import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from train.radio.knn import train_knn_radio
from utils.data_loader import load_mdt_data
from utils.metrics import calculate_radio_mae
import joblib

def run_k_ablation(k_values=(1, 2, 3, 4, 5)):
    # K is a hyperparameter: it is evaluated on the VALIDATION set, never on the test set
    coords_val, real_rsrp = load_mdt_data('data/validation.csv', mode='radio')
    coords_val, real_rsrp = coords_val.numpy(), real_rsrp.numpy()

    results = {}
    for k in k_values:
        save_path = train_knn_radio(k=k)
        knn = joblib.load(save_path)
        pred = knn.predict(coords_val)
        results[k] = calculate_radio_mae(real_rsrp, pred)

    print(f"{'K':<5}{'Validation MAE (dBm)':<12}")
    for k, mae in results.items():
        print(f"{k:<5}{mae:<12.4f}")
    return results

if __name__ == "__main__":
    run_k_ablation()
