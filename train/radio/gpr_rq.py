import sys
import os
import joblib
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic, WhiteKernel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.gpr import fit_gpr

def train_gpr_rq():
    print("Training Radio GPR (Rational Quadratic)...")
    
    coords, rsrps = load_mdt_data('data/train.csv', mode='radio')
    X = coords.numpy()
    y = rsrps.numpy()
    
    # Kernel: Amplitude * Rational Quadratic + White noise (shadowing)
    # Initial values only; they are learned by maximizing the marginal likelihood
    kernel = ConstantKernel(1.0) * RationalQuadratic(length_scale=100.0, alpha=1.0) + WhiteKernel(noise_level=0.1)
    gpr = fit_gpr(X, y, kernel)
    os.makedirs('models/radio', exist_ok=True)
    joblib.dump(gpr, 'models/radio/gpr_rq.pkl')
    print("Model saved to models/radio/gpr_rq.pkl")

if __name__ == "__main__":
    train_gpr_rq()