import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import mean_spatial_ks

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
    
    # Mean over 10 draws
    results = mean_spatial_ks(real_data, sample)
    
    print("-" * 30)
    print(f"GMM Evaluation Results:")
    p_txt = f"{results['p_2d']:.2f}" if results['p_2d'] >= 0.01 else "<0.01"
    print(f"2-D KS (FF) : {results['ks_2d']:.2f} (p={p_txt}), mean of {results['n_draws']} draws")
    print("-" * 30)

if __name__ == "__main__":
    test_gmm()