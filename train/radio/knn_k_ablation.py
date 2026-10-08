import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from train.radio.knn import train_knn_radio
from utils.data_loader import load_mdt_data
from utils.metrics import calculate_radio_mae
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

def validation_positioning_error(knn, kde, train_df, val_df, n_samples=2000, seed=42):
    """
    Downstream utility of a radio KNN: median error on the VALIDATION set of a wKNN trained
    on real + synthetic data (KDE positions, radio from this KNN), as in the full pipeline.
    """
    coords = kde.sample(n_samples, random_state=seed)
    rsrp_cols = [c for c in train_df.columns if 'RSRP' in c]
    X = np.vstack([train_df[rsrp_cols].values, knn.predict(coords)])
    y = np.vstack([train_df[['LATITUDE', 'LONGITUDE']].values, coords])
    wknn = KNeighborsRegressor(n_neighbors=3, weights='distance', metric='euclidean').fit(X, y)
    errors = np.linalg.norm(wknn.predict(val_df[rsrp_cols].values) - val_df[['LATITUDE', 'LONGITUDE']].values, axis=1)
    return np.median(errors)

def run_k_ablation(k_values=(1, 2, 3, 5, 10, 20)):
    # K is a hyperparameter: it is evaluated on the VALIDATION set, never on the test set
    coords_val, real_rsrp = load_mdt_data('data/validation.csv', mode='radio')
    coords_val, real_rsrp = coords_val.numpy(), real_rsrp.numpy()
    train_df = load_mdt_data('data/train.csv', mode='all')
    val_df = load_mdt_data('data/validation.csv', mode='all')
    kde = joblib.load('models/spatial/kde.pkl')

    results = {}
    for k in k_values:
        save_path = train_knn_radio(k=k)
        knn = joblib.load(save_path)
        pred = knn.predict(coords_val)
        results[k] = (calculate_radio_mae(real_rsrp, pred), validation_positioning_error(knn, kde, train_df, val_df))

    print(f"{'K':<5}{'Val. radio MAE (dBm)':<24}{'Val. median positioning error (m)':<12}")
    for k, (mae, pos_err) in results.items():
        print(f"{k:<5}{mae:<24.4f}{pos_err:<12.3f}")
    best_k = min(results, key=lambda k: results[k][1])
    print(f"Selected K (lowest validation positioning error): {best_k}")
    return results, best_k

if __name__ == "__main__":
    run_k_ablation()
