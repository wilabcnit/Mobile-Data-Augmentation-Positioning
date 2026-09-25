import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from train.radio.knn import train_knn_radio
from utils.data_loader import load_mdt_data
from utils.metrics import calculate_radio_mae
import joblib

def run_k_ablation(k_values=(1, 2, 3, 4, 5)):
    coords_test, real_rsrp = load_mdt_data('data/test.csv', mode='radio')
    coords_test, real_rsrp = coords_test.numpy(), real_rsrp.numpy()

    print(f"{'K':<5}{'MAE (dBm)':<12}")
    for k in k_values:
        train_knn_radio(k=k)
        knn = joblib.load(f'models/radio/knn_k{k}.pkl')
        pred = knn.predict(coords_test)
        mae = calculate_radio_mae(real_rsrp, pred)
        print(f"{k:<5}{mae:<12.4f}")

if __name__ == "__main__":
    run_k_ablation()
