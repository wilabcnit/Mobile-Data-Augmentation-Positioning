import sys
import os
import joblib
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import calculate_spatial_ks

def test_kde():
    print("Testing Spatial KDE...")
    
    # 1. Load Model
    if not os.path.exists('models/spatial/kde.pkl'):
        print("Error: Model not found. Run train/spatial/kde.py first.")
        return
        
    kde = joblib.load('models/spatial/kde.pkl')
    
    # 2. Load Test Data
    real_data = load_mdt_data('data/test.csv', mode='spatial')
    n_samples = len(real_data)
    
    # 3. Generate Synthetic Data
    generated_data = kde.sample(n_samples)
    
    # 4. Evaluate
    results = calculate_spatial_ks(real_data, generated_data)
    
    print("-" * 30)
    print(f"KDE Evaluation Results:")
    print(f"KS Latitude : {results['ks_lat']:.4f} (p={results['p_lat']:.4f})")
    print(f"KS Longitude: {results['ks_lon']:.4f} (p={results['p_lon']:.4f})")
    print(f"Average KS  : {results['avg_ks']:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    test_kde()