import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import median_spatial_ks

def test_gmm():
    print("Testing Spatial GMM...")
    
    if not os.path.exists('models/spatial/gmm.pkl'):
        print("Error: Model not found. Run train/spatial/gmm.py first.")
        return
        
    gmm = joblib.load('models/spatial/gmm.pkl')
    
    real_data = load_mdt_data('data/test.csv', mode='spatial')
    n_samples = len(real_data)
    
    # Generate samples (one reproducible draw per seed)
    def sample(seed):
        gmm.random_state = seed
        return gmm.sample(n_samples)[0]
    
    # Median over 10 draws
    results = median_spatial_ks(real_data, sample)
    
    print("-" * 30)
    print(f"GMM Evaluation Results:")
    print(f"2-D KS (FF) : {results['ks_2d']:.4f} (p={results['p_2d']:.4f}), median of {results['n_draws']} draws")
    print("-" * 30)

if __name__ == "__main__":
    test_gmm()