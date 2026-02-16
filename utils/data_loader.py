import pandas as pd
import torch
import numpy as np

def load_mdt_data(path, mode='spatial'):
    """
    Loads MDT data from CSV.
    
    Args:
        path (str): Path to the CSV file.
        mode (str): 'spatial' returns (LAT, LON), 'radio' returns (LAT, LON, RSRPs), 'all' returns dataframe.
    
    Returns:
        np.ndarray or torch.Tensor: The requested data.
    """
    df = pd.read_csv(path)
    
    if mode == 'spatial':
        # Return Latitude and Longitude as Numpy array (for Scikit-Learn models like KDE/GMM)
        return df[['LATITUDE', 'LONGITUDE']].values
        
    elif mode == 'radio':
        # Return Inputs (Lat, Lon) and Targets (RSRP_1...N)
        coords = df[['LATITUDE', 'LONGITUDE']].values
        rsrp_cols = [c for c in df.columns if 'RSRP' in c]
        rsrp = df[rsrp_cols].values
        return torch.tensor(coords, dtype=torch.float32), torch.tensor(rsrp, dtype=torch.float32)
    
    elif mode == 'all':
        return df
    
    else:
        raise ValueError("Mode must be 'spatial', 'radio', or 'all'")