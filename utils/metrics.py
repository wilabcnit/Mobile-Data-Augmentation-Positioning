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
