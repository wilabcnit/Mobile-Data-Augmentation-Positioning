import numpy as np
from sklearn.metrics import mean_absolute_error

from utils.ks2d_ff import ks2d2s

def calculate_radio_mae(real_rsrp, predicted_rsrp):
    """
    Computes Mean Absolute Error for Radio Map reconstruction.

    Args:
        real_rsrp (np.array): Real RSRP values [N, n_cells].
        predicted_rsrp (np.array): Predicted RSRP values [N, n_cells].

    Returns:
        float: The MAE averaged over all samples and all cells.
    """
    return mean_absolute_error(real_rsrp, predicted_rsrp)

def calculate_spatial_ks(real_coords, generated_coords, n_perm=1000, seed=0):
    """
    Compares real and generated (Lat, Lon) distributions with the 2-D
    Fasano-Franceschini Kolmogorov-Smirnov test.

    Args:
        real_coords (np.array): Real positions [N, 2].
        generated_coords (np.array): Generated positions [M, 2].
        n_perm (int): Number of permutations for the p-value.
        seed (int): Seed of the permutation test.

    Returns:
        dict: 2-D KS statistic and permutation p-value.
    """
    real_coords = np.asarray(real_coords, dtype=float)
    generated_coords = np.asarray(generated_coords, dtype=float)
    ks_2d = ks2d2s(real_coords, generated_coords, method="permutation", n_perm=n_perm, seed=seed)
    return {"ks_2d": ks_2d["D"], "p_2d": ks_2d["p"]}

def mean_spatial_ks(real_coords, sample_fn, n_draws=10):
    """
    2-D KS test of n_draws independent generated sets: sample_fn(seed) returns the
    generated positions of draw seed = 0..n_draws-1. Returns the mean statistic
    and the mean p-value over the draws.
    """
    draws = [calculate_spatial_ks(real_coords, sample_fn(seed)) for seed in range(n_draws)]
    return {"ks_2d": np.mean([d["ks_2d"] for d in draws]),
            "p_2d": np.mean([d["p_2d"] for d in draws]), "n_draws": n_draws}
