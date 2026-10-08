"""
Radio KNN K ablation on positioning, for every spatial model: synthetic positions from
KDE / NF / GAN / GMM + radio KNN (K = 1..5), wKNN trained on real + synthetic data.
Median positioning error on the VALIDATION set (where K is selected) and on the TEST set.
Run train/radio/knn.py first (it trains K = 1..5).
"""
import sys
import os
import joblib
import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.geo import haversine_distance
from utils.stats_compare import compare_models_paired
from generate.full_pipeline import (synthesize, n_synthetic,
                                    SPATIAL_MODELS, KNN_K_VALUES as K_VALUES, knn_model_path)
from positioning.train import build_wknn

def positioning_errors(train_df, eval_df, synthetic=None):
    """Errors on eval_df of a wKNN trained on train_df (+ synthetic (coords, rsrp))."""
    rsrp_cols = [c for c in train_df.columns if 'RSRP' in c]
    X, y = train_df[rsrp_cols].values, train_df[['LATITUDE', 'LONGITUDE']].values
    if synthetic is not None:
        X, y = np.vstack([X, synthetic[1]]), np.vstack([y, synthetic[0]])
    wknn = build_wknn().fit(X, y)
    return haversine_distance(wknn.predict(eval_df[rsrp_cols].values), eval_df[['LATITUDE', 'LONGITUDE']].values)

def run_k_ablation():
    train_df = pd.read_csv('data/train.csv')
    sets = {'validation': pd.read_csv('data/validation.csv'), 'test': pd.read_csv('data/test.csv')}
    knns = {k: joblib.load(knn_model_path(k)) for k in K_VALUES}

    baseline = {name: positioning_errors(train_df, df) for name, df in sets.items()}
    errors = {}
    for s in SPATIAL_MODELS:
        for k in K_VALUES:
            # Same synthetic positions for every K, identical to generate/full_pipeline.py: the
            # spatial model is loaded inside synthesize() after seeding (loading the GAN / NF
            # initializes torch layers, which consumes random numbers)
            synthetic = synthesize(s, 'knn', n_synthetic(), radio=(knns[k], None))
            errors[s, k] = {name: positioning_errors(train_df, df, synthetic) for name, df in sets.items()}

    header = ["Spatial \\ Radio KNN"] + [f"K={k}" for k in K_VALUES]
    for name in sets:
        rows = [["Original data"] + [f"{np.median(baseline[name]):.3f}m"] * len(K_VALUES)]
        for s in SPATIAL_MODELS:
            row = [s.upper()]
            for k in K_VALUES:
                cell = f"{np.median(errors[s, k][name]):.3f}m"
                # '*': median significantly different from the original data (paired bootstrap 95% CI)
                if name == 'test' and compare_models_paired(errors[s, k][name], baseline[name])["Significant"]:
                    cell += "*"
                row.append(cell)
            rows.append(row)
        print("\n" + "=" * 80)
        print(f"{f'MEDIAN POSITIONING ERROR ({name.upper()} SET) - RADIO KNN, K = 1..5':^80}")
        print("=" * 80)
        print(tabulate(rows, headers=header, tablefmt="grid"))
    print("* test median significantly different from the original data (paired bootstrap 95% CI excludes 0)")
    for s in SPATIAL_MODELS:
        best_k = min(K_VALUES, key=lambda k: np.median(errors[s, k]['validation']))
        print(f"{s.upper()}: lowest validation positioning error at K={best_k}")

if __name__ == "__main__":
    run_k_ablation()
