import sys
import os
import joblib
from sklearn.mixture import GaussianMixture

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data

def train_gmm(max_components=15):
    print("Training Spatial GMM...")
    
    # 1. Load Data
    data = load_mdt_data('data/train.csv', mode='spatial')
    val_data = load_mdt_data('data/validation.csv', mode='spatial')
    
    # 2. Select the number of components with the Bayesian Information Criterion
    best_gmm, best_bic = None, float('inf')
    for k in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42, max_iter=200, n_init=3)
        gmm.fit(data)
        bic = gmm.bic(data)
        if bic < best_bic:
            best_gmm, best_bic = gmm, bic
    gmm = best_gmm
    
    print(f"Best n_components: {gmm.n_components} (BIC: {best_bic:.1f}, validation log-likelihood: {gmm.score(val_data):.4f})")
    print(f"GMM Converged: {gmm.converged_}")
    print(f"Iterations: {gmm.n_iter_}")
    
    # 3. Save Model
    os.makedirs('models/spatial', exist_ok=True)
    joblib.dump(gmm, 'models/spatial/gmm.pkl')
    print("Model saved to models/spatial/gmm.pkl")

if __name__ == "__main__":
    train_gmm()
