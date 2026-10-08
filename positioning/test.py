import sys
import os
import joblib
import pandas as pd
import numpy as np
from tabulate import tabulate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from generate.full_pipeline import SPATIAL_MODELS, RADIO_MODELS
from utils.stats_compare import compare_models_paired
from utils.geo import haversine_distance

def positioning_errors(model_name):
    """Returns the per-point positioning errors on the ORIGINAL test set, or None if the model does not exist."""
    model_path = f'models/positioning/{model_name}.pkl'
    if not os.path.exists(model_path):
        return None

    # 1. Load Model
    wknn = joblib.load(model_path)

    # 2. Load ORIGINAL Test Data
    df_test = pd.read_csv('data/test.csv')
    rsrp_cols = [c for c in df_test.columns if 'RSRP' in c]
    X_test = df_test[rsrp_cols].values
    y_test = df_test[['LATITUDE', 'LONGITUDE']].values

    # 3. Predict Locations
    y_pred = wknn.predict(X_test)

    # 4. Calculate Positioning Error (great-circle distance in metres between WGS84 positions)
    return haversine_distance(y_pred, y_test)

def test_positioning(model_name):
    errors = positioning_errors(model_name)
    if errors is None:
        return None
    return {
        "median": np.median(errors),
        "mean": np.mean(errors),
        "p95": np.percentile(errors, 95),
        "errors": errors
    }

def print_table(title, suffix, baseline, stat):
    header = ["Spatial \\ Radio"] + [r.upper() for r in RADIO_MODELS]
    rows = [["Original data"] + [f"{baseline[stat]:.3f}m" if baseline else "X"] * len(RADIO_MODELS)]
    for s in SPATIAL_MODELS:
        row = [s.upper()]
        for r in RADIO_MODELS:
            res = test_positioning(f"wknn_{s}_{r}{suffix}")
            if res is None:
                row.append("X")
                continue
            cell = f"{res[stat]:.3f}m"
            # '*': median significantly different from the baseline (paired bootstrap 95% CI)
            if stat == 'median' and baseline and compare_models_paired(res['errors'], baseline['errors'])["Significant"]:
                cell += "*"
            row.append(cell)
        rows.append(row)
    print("\n" + "="*100)
    print(f"{title:^100}")
    print("="*100)
    print(tabulate(rows, headers=header, tablefmt="grid"))

if __name__ == "__main__":
    # Results Print
    baseline = test_positioning('wknn_baseline')
    print_table("MEDIAN POSITIONING ERROR - wKNN TRAINED ON REAL + SYNTHETIC MDT DATA", "", baseline, "median")
    print_table("MEDIAN POSITIONING ERROR - wKNN TRAINED ON SYNTHETIC MDT DATA ONLY", "_synth", baseline, "median")
    print_table("MEAN POSITIONING ERROR - wKNN TRAINED ON REAL + SYNTHETIC MDT DATA", "", baseline, "mean")
    print_table("MEAN POSITIONING ERROR - wKNN TRAINED ON SYNTHETIC MDT DATA ONLY", "_synth", baseline, "mean")
    print("* median error significantly different from the original data (paired bootstrap 95% CI on the median difference excludes 0)")
