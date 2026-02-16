import sys
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def haversine_distance(pred, true):
    diff = pred - true
    dist = np.sqrt(np.sum(diff**2, axis=1))
    return dist

def test_positioning(model_name='wknn_baseline'):
    print(f"Testing Positioning Model: {model_name}...")
    
    model_path = f'models/positioning/{model_name}.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model {model_name} not found.")
        return
        
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
    
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    p95_error = np.percentile(errors, 95)
    
    print("-" * 30)
    print(f"Positioning Results ({model_name}):")
    print(f"Mean Error  : {mean_error:.4f} m")
    print(f"Median Error: {median_error:.4f} m")
    print(f"95th % Error: {p95_error:.4f} m")
    print("-" * 30)
    
    return errors

if __name__ == "__main__":
    test_positioning('wknn_baseline')
    test_positioning('wknn_kde_knn')