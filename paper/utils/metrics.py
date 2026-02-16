import numpy as np
from scipy.stats import ks_2samp

def calculate_spatial_ks(real_data, generated_data):
    """
    Computes the 2-sample KS test for spatial distributions.
    Option A: Independent 1D KS tests for Latitude and Longitude.
    
    Args:
        real_data (np.array): Real (LAT, LON) data [N, 2].
        generated_data (np.array): Generated (LAT, LON) data [M, 2].
        
    Returns:
        dict: KS statistics and p-values.
    """
    # 1. KS Test for Latitude (Column 0)
    ks_lat_stat, ks_lat_p = ks_2samp(real_data[:, 0], generated_data[:, 0])
    
    # 2. KS Test for Longitude (Column 1)
    ks_lon_stat, ks_lon_p = ks_2samp(real_data[:, 1], generated_data[:, 1])
    
    # 3. Average KS Statistic (Lower is better, closer to 0 means same distribution)
    avg_ks = (ks_lat_stat + ks_lon_stat) / 2.0
    
    return {
        "ks_lat": ks_lat_stat,
        "p_lat": ks_lat_p,
        "ks_lon": ks_lon_stat,
        "p_lon": ks_lon_p,
        "avg_ks": avg_ks
    }
    

from sklearn.metrics import mean_absolute_error

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