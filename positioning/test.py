import sys
import os
import joblib
import pandas as pd
import numpy as np
from tabulate import tabulate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.geo import haversine_distance
from utils.stats_compare import compare_models_paired

def test_positioning(model_name):
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

    # 4. Calculate Positioning Error (metres between WGS84 positions)
    errors = haversine_distance(y_pred, y_test)

    return {
        "median": np.median(errors),
        "errors": errors
    }

if __name__ == "__main__":
    # Results Print
    spatial_models = ['baseline', 'kde', 'nf', 'gan', 'gmm']
    radio_models = ['knn', 'mlp', 'cgan', 'gpr_rq', 'gpr_se', 'nf', 'rec_flow', 'rf']
    header = ["Spatial \\ Radio"] + [r.upper() for r in radio_models]
    rows = []
    print("\n" + "="*100)
    print(f"{'POSITIONING PERFORMANCE (MEDIAN ERROR) - wKNN TRAINED ON REAL + SYNTHETIC MDT DATA':^100}")
    print("="*100)
    baseline = test_positioning('wknn_baseline')
    for s in spatial_models:
        if s == 'baseline':
            val = f"{baseline['median']:.2f}m" if baseline else "X"
            rows.append(["Original data"] + [val] * (len(radio_models)))
            continue

        row = [s.upper()]
        for r in radio_models:
            res = test_positioning(f"wknn_{s}_{r}")

            if res:
                cell = f"{res['median']:.2f}m"
                # '*': median significantly different from the original data (paired bootstrap 95% CI)
                if baseline and compare_models_paired(res['errors'], baseline['errors'])["Significant"]:
                    cell += "*"
                row.append(cell)
            else:
                row.append("X")
        rows.append(row)
    print(tabulate(rows, headers=header, tablefmt="grid"))
    print("* median error significantly different from the original data (paired bootstrap 95% CI excludes 0)")
