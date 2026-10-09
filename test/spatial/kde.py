import sys
import os
import joblib
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import median_spatial_ks

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
    
    # 3. Generate Synthetic Data (one independent draw per seed)
    def sample(seed):
        return kde.sample(n_samples, random_state=seed)
    
    # 4. Evaluate: median over 10 draws
    results = median_spatial_ks(real_data, sample)
    
    print("-" * 30)
    print(f"KDE Evaluation Results:")
    print(f"2-D KS (FF) : {results['ks_2d']:.4f} (p={results['p_2d']:.4f}), median of {results['n_draws']} draws")
    print("-" * 30)

if __name__ == "__main__":
    test_kde()