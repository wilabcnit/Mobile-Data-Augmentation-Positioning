"""
Radio KNN K ablation on positioning: KDE positions + radio KNN (K = 1..5), wKNN trained
on real + synthetic data. K is selected on the VALIDATION set; the TEST set column reports
the same comparison, with the paired bootstrap test against the original data and K = 1.
Run train/radio/knn.py first (it trains K = 1..5).
"""
import sys
import os
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.geo import haversine_distance
from utils.stats_compare import compare_models_paired
from generate.full_pipeline import knn_radio, n_synthetic, KNN_K_VALUES as K_VALUES, knn_model_path
from positioning.train import build_wknn

def positioning_errors(train_df, eval_df, synthetic=None):
    """Errors on eval_df of a wKNN trained on train_df (+ synthetic (coords, rsrp))."""
    rsrp_cols = [c for c in train_df.columns if 'RSRP' in c]
    X, y = train_df[rsrp_cols].values, train_df[['LATITUDE', 'LONGITUDE']].values
    if synthetic is not None:
        X, y = np.vstack([X, synthetic[1]]), np.vstack([y, synthetic[0]])
    wknn = build_wknn().fit(X, y)
    return haversine_distance(wknn.predict(eval_df[rsrp_cols].values), eval_df[['LATITUDE', 'LONGITUDE']].values)

def run_k_ablation(seed=42):
    train_df = pd.read_csv('data/train.csv')
    sets = {'validation': pd.read_csv('data/validation.csv'), 'test': pd.read_csv('data/test.csv')}
    kde = joblib.load('models/spatial/kde.pkl')
    coords = kde.sample(n_synthetic(), random_state=seed)   # same synthetic positions for every K

    baseline = {name: positioning_errors(train_df, df) for name, df in sets.items()}
    errors = {}
    for k in K_VALUES:
        rsrp = knn_radio(joblib.load(knn_model_path(k)), coords)
        errors[k] = {name: positioning_errors(train_df, df, (coords, rsrp)) for name, df in sets.items()}

    print(f"Original data: validation {np.median(baseline['validation']):.3f} m | test {np.median(baseline['test']):.3f} m")
    print(f"{'K':<4}{'Validation (m)':<16}{'Test (m)':<10}{'Test vs original, 95% CI':<36}{'Test, K=1 minus K, 95% CI'}")
    for k in K_VALUES:
        vs_base = compare_models_paired(errors[k]['test'], baseline['test'])
        line = (f"{k:<4}{np.median(errors[k]['validation']):<16.3f}{np.median(errors[k]['test']):<10.3f}"
                f"{vs_base['Median Difference']:+.3f} {str(vs_base['95% CI']):<29}")
        if k != 1:
            vs_1 = compare_models_paired(errors[1]['test'], errors[k]['test'])
            line += f"{vs_1['Median Difference']:+.3f} {vs_1['95% CI']}"
        print(line)
    best_k = min(K_VALUES, key=lambda k: np.median(errors[k]['validation']))
    print(f"Lowest validation positioning error: K={best_k}")

if __name__ == "__main__":
    run_k_ablation()
