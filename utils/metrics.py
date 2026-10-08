import numpy as np
from scipy.stats import ks_2samp
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
    Compares real and generated (Lat, Lon) distributions with Kolmogorov-Smirnov tests.

    Args:
        real_coords (np.array): Real positions [N, 2].
        generated_coords (np.array): Generated positions [M, 2].
        n_perm (int): Number of permutations for the 2-D test p-value.
        seed (int): Seed of the permutation test.

    Returns:
        dict: 1-D KS statistic and p-value per coordinate, their average, and the
              2-D Fasano-Franceschini statistic and permutation p-value.
    """
    real_coords = np.asarray(real_coords, dtype=float)
    generated_coords = np.asarray(generated_coords, dtype=float)

    ks_lat = ks_2samp(real_coords[:, 0], generated_coords[:, 0])
    ks_lon = ks_2samp(real_coords[:, 1], generated_coords[:, 1])
    ks_2d = ks2d2s(real_coords, generated_coords, method="permutation", n_perm=n_perm, seed=seed)

    return {
        "ks_lat": ks_lat.statistic, "p_lat": ks_lat.pvalue,
        "ks_lon": ks_lon.statistic, "p_lon": ks_lon.pvalue,
        "avg_ks": 0.5 * (ks_lat.statistic + ks_lon.statistic),
        "ks_2d": ks_2d["D"], "p_2d": ks_2d["p"]
    }
