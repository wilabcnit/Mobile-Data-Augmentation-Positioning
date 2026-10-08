"""
Test-set positioning of KDE + radio KNN for K = 1..5 (real + synthetic training).

K is selected on the validation set by train/radio/knn_k_ablation.py; this script only
reports, for each K, the median positioning error on the test set and the paired
bootstrap comparison with the original data and with K = 1.
Run train/radio/knn_k_ablation.py first (it saves models/radio/knn_k{K}.pkl).
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import joblib
import numpy as np
import pandas as pd
from utils.geo import haversine_distance
from utils.stats_compare import compare_models_paired
from generate.full_pipeline import knn_radio, n_synthetic
from positioning.train import build_wknn

def run(k_values=(1, 2, 3, 4, 5), n_samples=None, seed=42):
    train_df, test_df = pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')
    coords_cols = ['LATITUDE', 'LONGITUDE']
    rsrp_cols = [c for c in train_df.columns if 'RSRP' in c]
    kde = joblib.load('models/spatial/kde.pkl')
    coords = kde.sample(n_samples or n_synthetic(), random_state=seed)   # same synthetic positions for every K

    def errors(X, y):
        wknn = build_wknn().fit(X, y)
        return haversine_distance(wknn.predict(test_df[rsrp_cols].values), test_df[coords_cols].values)

    baseline = errors(train_df[rsrp_cols].values, train_df[coords_cols].values)
    errs = {}
    for k in k_values:
        knn = joblib.load(f'models/radio/knn_k{k}.pkl')
        errs[k] = errors(np.vstack([train_df[rsrp_cols].values, knn_radio(knn, coords)]),
                         np.vstack([train_df[coords_cols].values, coords]))

    print(f"Original data: median {np.median(baseline):.3f} m")
    for k, e in errs.items():
        vs_base = compare_models_paired(e, baseline)
        line = (f"KDE + KNN K={k}: median {np.median(e):.3f} m | vs original "
                f"{vs_base['Median Difference']:+.3f} m, 95% CI {vs_base['95% CI']}")
        if k != 1:
            vs_1 = compare_models_paired(errs[1], e)
            line += f" | K=1 minus K={k}: {vs_1['Median Difference']:+.3f} m, 95% CI {vs_1['95% CI']}"
        print(line)

if __name__ == "__main__":
    run()
