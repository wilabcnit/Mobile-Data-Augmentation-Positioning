import sys
import os
import joblib
from sklearn.mixture import GaussianMixture

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data

def train_gmm():
    print("Training Spatial GMM...")
    
    # 1. Load Data
    data = load_mdt_data('data/train.csv', mode='spatial')
    
    # 2. Train GMM
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, max_iter=200)
    gmm.fit(data)
    
    print(f"GMM Converged: {gmm.converged_}")
    print(f"Iterations: {gmm.n_iter_}")
    
    # 3. Save Model
    os.makedirs('models/spatial', exist_ok=True)
    joblib.dump(gmm, 'models/spatial/gmm.pkl')
    print("Model saved to models/spatial/gmm.pkl")

if __name__ == "__main__":
    train_gmm()