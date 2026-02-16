import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import calculate_spatial_ks

def test_gmm():
    print("Testing Spatial GMM...")
    
    if not os.path.exists('models/spatial/gmm.pkl'):
        print("Error: Model not found. Run train/spatial/gmm.py first.")
        return
        
    gmm = joblib.load('models/spatial/gmm.pkl')
    
    real_data = load_mdt_data('data/test.csv', mode='spatial')
    n_samples = len(real_data)
    
    # Generate samples
    generated_data, _ = gmm.sample(n_samples)
    
    results = calculate_spatial_ks(real_data, generated_data)
    
    print("-" * 30)
    print(f"GMM Evaluation Results:")
    print(f"KS Latitude : {results['ks_lat']:.4f} (p={results['p_lat']:.4f})")
    print(f"KS Longitude: {results['ks_lon']:.4f} (p={results['p_lon']:.4f})")
    print(f"Average KS  : {results['avg_ks']:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    test_gmm()