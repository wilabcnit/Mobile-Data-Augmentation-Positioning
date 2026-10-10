import sys
import os
import joblib
import numpy as np
from sklearn.neighbors import KernelDensity

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data

def train_kde():
    print("Training Spatial KDE...")
    
    # 1. Load Data
    data = load_mdt_data('data/train.csv', mode='spatial')
    val_data = load_mdt_data('data/validation.csv', mode='spatial')
    
    # 2. Select the bandwidth by validation log-likelihood
    spread = data.std(axis=0).mean()
    bandwidths = spread * np.logspace(-3, 0, 31)
    scores = [KernelDensity(kernel='gaussian', bandwidth=bw).fit(data).score(val_data) for bw in bandwidths]
    best_idx = int(np.argmax(scores))
    if best_idx in (0, len(bandwidths) - 1):
        print("Warning: best bandwidth is at the edge of the search grid.")
    
    best_kde = KernelDensity(kernel='gaussian', bandwidth=bandwidths[best_idx]).fit(data)
    print(f"Best Bandwidth: {best_kde.bandwidth:.3g} (validation log-likelihood: {scores[best_idx]:.2f})")
    
    # 3. Save Model
    os.makedirs('models/spatial', exist_ok=True)
    joblib.dump(best_kde, 'models/spatial/kde.pkl')
    print("Model saved to models/spatial/kde.pkl")

if __name__ == "__main__":
    train_kde()
