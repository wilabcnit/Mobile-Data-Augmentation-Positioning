import sys
import os
import joblib
import pandas as pd
import numpy as np
from tabulate import tabulate

def haversine_distance(pred, true):
    # Standard Euclidean distance for positioning error
    return np.sqrt(np.sum((pred - true)**2, axis=1))

def test_positioning(model_name):
    model_path = f'models/positioning/{model_name}.pkl'
    if not os.path.exists(model_path):
        return None
        
    try:
        # 1. Load Model
        wknn = joblib.load(model_path)

        # 2. Load ORIGINAL Test Data
        df_test = pd.read_csv('data/test.csv')
        rsrp_cols = [c for c in df_test.columns if 'RSRP' in c]
        X_test = df_test[rsrp_cols].values
        y_test = df_test[['LATITUDE', 'LONGITUDE']].values
        
        # 3. Predict Locations
        y_pred = wknn.predict(X_test)

        # 4. Calculate Positioning Error
        errors = haversine_distance(y_pred, y_test)
        
        return {
            "mean": np.mean(errors),
            "p95": np.percentile(errors, 95)
        }
    except Exception:
        return None

if __name__ == "__main__":
    # Results Print
    spatial_models = ['baseline', 'kde', 'nf', 'gan', 'gmm']
    radio_models = ['knn', 'mlp', 'cgan', 'gpr_rq', 'gpr_se', 'nf', 'rec_flow', 'rf']
    header = ["Spatial \\ Radio"] + [r.upper() for r in radio_models]
    rows = []
    print("\n" + "="*100)
    print(f"{'POSITIONING PERFORMANCE(MedAE) OF MDT DATA GENERATED THROUGH ALL MODELS':^100}")
    print("="*100)
    for s in spatial_models:
        if s == 'baseline':
            row = ["Original data"]
            res = test_positioning('wknn_baseline')
            val = f"{res['mean']:.3f}m" if res else "X"
            rows.append([row[0]] + [val] * (len(radio_models)))
            continue

        row = [s.upper()]
        for r in radio_models:
            model_name = f"wknn_{s}_{r}"
            res = test_positioning(model_name)
            
            if res:
                row.append(f"{res['mean']:.3f}m")
            else:
                row.append("X")
        rows.append(row)
    print(tabulate(rows, headers=header, tablefmt="grid"))