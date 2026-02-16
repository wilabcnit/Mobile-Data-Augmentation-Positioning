import sys
import os
import joblib
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data

def train_kde():
    print("Training Spatial KDE...")
    
    # 1. Load Data
    data = load_mdt_data('data/train.csv', mode='spatial')
    
    # 2. Setup KDE with Grid Search for Bandwidth
    params = {'bandwidth': np.logspace(-2, 0, 5)}
    grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=3)
    grid.fit(data)
    
    best_kde = grid.best_estimator_
    print(f"Best Bandwidth: {best_kde.bandwidth}")
    
    # 3. Save Model
    os.makedirs('models/spatial', exist_ok=True)
    joblib.dump(best_kde, 'models/spatial/kde.pkl')
    print("Model saved to models/spatial/kde.pkl")

if __name__ == "__main__":
    import numpy as np
    train_kde()