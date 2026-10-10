import sys
import os
import joblib
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.gpr import fit_gpr

def train_gpr_se():
    print("Training Radio GPR (Squared Exponential)...")
    
    coords, rsrps = load_mdt_data('data/train.csv', mode='radio')
    X = coords.numpy()
    y = rsrps.numpy()
    
    # Kernel: Amplitude * RBF (Squared Exponential) + White noise (shadowing)
    kernel = ConstantKernel(1.0) * RBF(length_scale=100.0) + WhiteKernel(noise_level=0.1)
    gpr = fit_gpr(X, y, kernel)
    os.makedirs('models/radio', exist_ok=True)
    joblib.dump(gpr, 'models/radio/gpr_se.pkl')
    print("Model saved to models/radio/gpr_se.pkl")

if __name__ == "__main__":
    train_gpr_se()
