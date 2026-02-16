import sys
import os
import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data

def train_gpr_se():
    print("Training Radio GPR (Squared Exponential)...")
    
    coords, rsrps = load_mdt_data('data/train.csv', mode='radio')
    X = coords.numpy()
    y = rsrps.numpy()
    
    # Kernel: RBF (Squared Exponential) + Noise
    kernel = 1.0 * RBF(length_scale=100.0) + WhiteKernel(noise_level=1.0)
    
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, copy_X_train=False)
    gpr.fit(X, y)
    
    os.makedirs('models/radio', exist_ok=True)
    joblib.dump(gpr, 'models/radio/gpr_se.pkl')
    print("Model saved to models/radio/gpr_se.pkl")

if __name__ == "__main__":
    train_gpr_se()