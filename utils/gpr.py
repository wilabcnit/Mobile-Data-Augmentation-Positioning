import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from utils.geo import GeoRegressor, to_local_m

def fit_gpr(X, y, kernel, n_hyper=2000, seed=42):
    """
    Fits a GPR radio map in two steps:
    1. Learns the kernel hyperparameters (amplitude, length scale, noise) by maximizing the
       marginal likelihood on a random subset of n_hyper points (exact GPR is O(N^3)).
    2. Refits on ALL training points with those hyperparameters kept fixed.
    
    normalize_y=True centers/scales each RSRP column, so far from the data the prediction
    reverts to the mean RSRP instead of 0 dBm.
    
    X is (LATITUDE, LONGITUDE) in degrees; the GPR works on planar metres (GeoRegressor),
    so the length scale is in metres.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_hyper, len(X)), replace=False)
    
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed)
    gpr.fit(to_local_m(X[idx]), y[idx])
    print(f"Learned kernel: {gpr.kernel_}")
    
    gpr_full = GeoRegressor(GaussianProcessRegressor(kernel=gpr.kernel_, normalize_y=True, optimizer=None, copy_X_train=False))
    gpr_full.fit(X, y)
    # The N x N Cholesky factor is only needed for the predictive standard deviation, which
    # the pipeline does not use; dropping it keeps the saved model small (mean prediction unchanged)
    gpr_full.model.L_ = None
    return gpr_full
