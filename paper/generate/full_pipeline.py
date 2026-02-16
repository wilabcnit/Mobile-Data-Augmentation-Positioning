import sys
import os
import torch
import joblib
import pandas as pd
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.spatial.definitions import SpatialGenerator, SpatialNF
from models.radio.definitions import RadioGenerator, RadioNF, VelocityField

def load_spatial_model(model_type):
    path = f'models/spatial/{model_type}.pkl'
    path_torch = f'models/spatial/{model_type}.pth'
    
    if model_type in ['kde', 'gmm']:
        return joblib.load(path), None
    elif model_type == 'gan':
        ckpt = torch.load(path_torch)
        model = SpatialGenerator(latent_dim=ckpt['latent_dim'])
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt
    elif model_type == 'nf':
        ckpt = torch.load(path_torch)
        model = SpatialNF(input_dim=2, n_flows=6) 
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt
    return None, None

def load_radio_model(model_type):
    path = f'models/radio/{model_type}.pkl'
    path_torch = f'models/radio/{model_type}.pth'
    
    sklearn_models = ['knn', 'rf', 'gpr_rq', 'gpr_se']
    
    if model_type in sklearn_models:
        return joblib.load(path), None
        
    elif model_type == 'mlp':
        ckpt = torch.load(path_torch)
        from models.radio.definitions import RadioMLP
        model = RadioMLP(input_dim=2, output_dim=30)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt
        
    elif model_type == 'cgan':
        ckpt = torch.load(path_torch)
        from models.radio.definitions import RadioGenerator
        model = RadioGenerator(condition_dim=2, latent_dim=ckpt['latent_dim'], output_dim=30)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt
        
    elif model_type == 'nf':
        ckpt = torch.load(path_torch)
        from models.radio.definitions import RadioNF
        model = RadioNF(input_dim=30, condition_dim=2, n_flows=8)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt
        
    return None, None

def generate_synthetic_data(spatial_algo, radio_algo, n_samples=5000):
    print(f"Running Augmentation Pipeline: Spatial[{spatial_algo}] -> Radio[{radio_algo}]")
    
    # --- 1. Generate Spatial Coords (Lat, Lon) ---
    spatial_model, s_ckpt = load_spatial_model(spatial_algo)
    
    if spatial_algo == 'kde':
        coords = spatial_model.sample(n_samples)
    elif spatial_algo == 'gmm':
        coords, _ = spatial_model.sample(n_samples)
    elif spatial_algo == 'gan':
        z = torch.randn(n_samples, s_ckpt['latent_dim'])
        with torch.no_grad():
            coords_norm = spatial_model(z).numpy()
        coords = coords_norm * s_ckpt['std'] + s_ckpt['mean']
    elif spatial_algo == 'nf':
        z = torch.randn(n_samples, 2)
        with torch.no_grad():
            coords_norm = spatial_model.inverse(z).numpy()
        coords = coords_norm * s_ckpt['std'] + s_ckpt['mean']
    
    # --- 2. Predict Radio Map (RSRP) ---
    radio_model, r_ckpt = load_radio_model(radio_algo)
    
    # Prepare coords for radio input
    coords_tensor = torch.FloatTensor(coords)
    
    sklearn_models = ['knn', 'rf', 'gpr_rq', 'gpr_se']
    
    if radio_algo in sklearn_models:
        rsrp = radio_model.predict(coords)
    else:
        # Deep Learning Models require normalization
        mean_in = torch.tensor(r_ckpt['mean_in']).float()
        std_in = torch.tensor(r_ckpt['std_in']).float()
        coords_norm = (coords_tensor - mean_in) / std_in
        
        with torch.no_grad():
            if radio_algo == 'mlp':
                rsrp_norm = radio_model(coords_norm)
            elif radio_algo == 'cgan':
                z = torch.randn(n_samples, r_ckpt['latent_dim'])
                rsrp_norm = radio_model(z, coords_norm)
            elif radio_algo == 'nf':
                z = torch.randn(n_samples, 30) # 30 cells
                rsrp_norm = radio_model.inverse(z, coords_norm)
        
        rsrp = rsrp_norm.numpy() * r_ckpt['std_out'] + r_ckpt['mean_out']

    # --- 3. Save ---
    df_aug = pd.DataFrame(coords, columns=['LATITUDE', 'LONGITUDE'])
    for i in range(rsrp.shape[1]):
        df_aug[f'RSRP_{i+1}'] = rsrp[:, i]
        
    os.makedirs('augm_data', exist_ok=True)
    out_path = f'augm_data/synthetic_{spatial_algo}_{radio_algo}.csv'
    df_aug.to_csv(out_path, index=False)
    print(f"Saved synthetic dataset to {out_path}")
    return out_path

if __name__ == "__main__":
    generate_synthetic_data('kde', 'knn', n_samples=2000)