import sys
import os
import joblib
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import mean_spatial_ks

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
    
    # 4. Evaluate: mean over 10 draws
    results = mean_spatial_ks(real_data, sample)
    
    print("-" * 30)
    print(f"KDE Evaluation Results:")
    p_txt = f"{results['p_2d']:.2f}" if results['p_2d'] >= 0.01 else "<0.01"
    print(f"2-D KS (FF) : {results['ks_2d']:.2f} (p={p_txt}), mean of {results['n_draws']} draws")
    print("-" * 30)

if __name__ == "__main__":
    test_kde()