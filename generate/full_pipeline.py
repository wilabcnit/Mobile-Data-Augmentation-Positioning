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
        ckpt = torch.load(path_torch, weights_only=False)
        #ckpt = torch.load(path_torch)
        model = SpatialGenerator(latent_dim=ckpt['latent_dim'])
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt
    elif model_type == 'nf':
        #ckpt = torch.load(path_torch)
        ckpt = torch.load(path_torch, weights_only=False)
        model = SpatialNF(input_dim=2, n_flows=6) 
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt
    return None, None

def load_radio_model(model_type):
    if model_type.startswith('knn_k'):          # radio KNN with K = 2..5 (ablation)
        return joblib.load(knn_model_path(int(model_type[len('knn_k'):]))), None
    path = f'models/radio/{model_type}.pkl'
    path_torch = f'models/radio/{model_type}.pth'
    
    sklearn_models = ['knn', 'rf', 'gpr_rq', 'gpr_se']
    
    if model_type in sklearn_models:
        return joblib.load(path), None
        
    elif model_type == 'mlp':
        ckpt = torch.load(path_torch)
        from models.radio.definitions import RadioMLP
        model = RadioMLP(input_dim=2, output_dim=len(ckpt['mean_out']))
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt
        
    elif model_type == 'cgan':
        ckpt = torch.load(path_torch)
        from models.radio.definitions import RadioGenerator
        model = RadioGenerator(condition_dim=2, latent_dim=ckpt['latent_dim'], output_dim=len(ckpt['mean_out']))
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt
        
    elif model_type == 'nf':
        ckpt = torch.load(path_torch)
        from models.radio.definitions import RadioNF
        model = RadioNF(input_dim=len(ckpt['mean_x']), condition_dim=2, n_flows=8)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt
    
    elif model_type == 'rec_flow':
        if not os.path.exists(path_torch):
            print(f"Error: {path_torch} not found!")
            return None, None
        ckpt = torch.load(path_torch)
        model = VelocityField(input_dim=len(ckpt['mean_x']), condition_dim=2)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt
        
    return None, None

UNSERVED_RSRP = -160

KNN_K_VALUES = (1, 2, 3, 4, 5)   # radio KNN: K = 1 is used by the pipeline; K = 2..5 for the ablation

def knn_model_path(k):
    return 'models/radio/knn.pkl' if k == 1 else f'models/radio/knn_k{k}.pkl'

def knn_radio(radio_model, coords):
    """
    Radio KNN generation: RSRP predicted by the KNN, then the unserved-cell state of the
    nearest real measurement is kept: a cell that is not served (RSRP = -160) at the nearest
    original position is also not served at the synthetic position.
    
    One neighbour search serves both steps. With K = 1 the prediction is the nearest real
    fingerprint itself (it already carries its not-served cells), so it is a row lookup.
    """
    k = radio_model.model.n_neighbors
    dist, idx = radio_model.kneighbors(coords, k)
    neighbours = radio_model.y_train_[idx]                       # [N, K, n_cells]
    if k == 1:
        rsrp = neighbours[:, 0].astype(float)
    else:
        # Inverse-distance weights as in KNeighborsRegressor(weights='distance'):
        # a neighbour at distance 0 takes all the weight
        with np.errstate(divide='ignore'):
            weights = 1.0 / dist
        exact = np.isinf(weights)
        weights[exact.any(axis=1)] = exact[exact.any(axis=1)]
        rsrp = np.einsum('nk,nkc->nc', weights, neighbours) / weights.sum(axis=1, keepdims=True)
        rsrp[neighbours[:, 0] == UNSERVED_RSRP] = UNSERVED_RSRP
    # In real MDT : Apply shadow fading variations to served cells as required by the environment (rsrp+noise)
    # Round to the 1 dB MDT reporting resolution (astype(int) alone truncates towards 0)
    return np.rint(rsrp).astype(int)

# Size of each synthetic dataset: SYNTHETIC_RATIO x the number of real training samples
SYNTHETIC_RATIO = 5

def n_synthetic(train_path='data/train.csv'):
    with open(train_path) as f:
        n_real = sum(1 for _ in f) - 1   # minus the header line
    return SYNTHETIC_RATIO * n_real

SPATIAL_MODELS = ['kde', 'nf', 'gan', 'gmm']
KNN_ABLATION_MODELS = [f'knn_k{k}' for k in KNN_K_VALUES if k != 1]   # augm_data/synthetic_<spatial>_knn_k<K>.csv
RADIO_MODELS = ['knn', 'mlp', 'cgan', 'gpr_rq', 'gpr_se', 'nf', 'rec_flow', 'rf']

PREDICT_BATCH = 20000   # sklearn models predict in batches (exact GPR needs a batch x N_train kernel matrix)

def batched_predict(fn, coords, batch=PREDICT_BATCH):
    return np.concatenate([fn(coords[i:i + batch]) for i in range(0, len(coords), batch)])

def synthesize(spatial_algo, radio_algo, n_samples, seed=42, spatial=None, radio=None):
    """
    Generates n_samples synthetic MDT samples: positions from the spatial model, then
    RSRP from the radio model. spatial / radio are (model, checkpoint) pairs; they are
    loaded from models/ when not given. Returns (coords [N, 2], rsrp [N, n_cells]).
    """
    # Fixed seeds so the synthetic datasets (and the results table) are reproducible
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # --- 1. Generate Spatial Coords (Lat, Lon) ---
    spatial_model, s_ckpt = spatial or load_spatial_model(spatial_algo)
    
    if spatial_algo == 'kde':
        coords = spatial_model.sample(n_samples, random_state=seed)
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
    radio_model, r_ckpt = radio or load_radio_model(radio_algo)
    
    # Prepare coords for radio input
    coords_tensor = torch.FloatTensor(coords)
    
    sklearn_models = ['rf', 'gpr_rq', 'gpr_se']
    if radio_algo.startswith('knn'):
        rsrp = batched_predict(lambda c: knn_radio(radio_model, c), coords)
    elif radio_algo in sklearn_models:
        rsrp = batched_predict(radio_model.predict, coords)
        
    else:
        # Deep Learning Models require normalization
        mean_key = 'mean_c' if 'mean_c' in r_ckpt else 'mean_in'
        std_key = 'std_c' if 'std_c' in r_ckpt else 'std_in'
        mean_in = torch.as_tensor(r_ckpt[mean_key]).float()
        std_in = torch.as_tensor(r_ckpt[std_key]).float()
        coords_norm = (coords_tensor - mean_in) / std_in
        
        with torch.no_grad():
            if radio_algo == 'mlp':
                rsrp_norm = radio_model(coords_norm)
            elif radio_algo == 'cgan':
                z = torch.randn(n_samples, r_ckpt['latent_dim'])
                rsrp_norm = radio_model(z, coords_norm)
            elif radio_algo == 'nf':
                z = torch.randn(n_samples, len(r_ckpt['mean_x']))
                rsrp_norm = radio_model.inverse(z, coords_norm)
            elif radio_algo == 'rec_flow':
                xt = torch.randn(n_samples, len(r_ckpt['mean_x']))
                n_steps = 25 
                dt = 1.0 / n_steps
                for i in range(n_steps):
                    t_val = torch.ones(n_samples, 1) * (i / n_steps)
                    xt = xt + radio_model(xt, t_val, coords_norm) * dt
                rsrp_norm = xt
        out_m_key = 'mean_x' if 'mean_x' in r_ckpt else 'mean_out'
        out_s_key = 'std_x' if 'std_x' in r_ckpt else 'std_out'
        m_out = r_ckpt[out_m_key].cpu().numpy()
        s_out = r_ckpt[out_s_key].cpu().numpy()
        rsrp = rsrp_norm.numpy() * s_out + m_out
    
    return coords, rsrp

def generate_synthetic_data(spatial_algo, radio_algo, n_samples=5000, seed=42):
    print(f"Running Augmentation Pipeline: Spatial[{spatial_algo}] -> Radio[{radio_algo}]")
    coords, rsrp = synthesize(spatial_algo, radio_algo, n_samples, seed)

    # --- 3. Save ---
    df_aug = pd.DataFrame(coords, columns=['LATITUDE', 'LONGITUDE'])
    rsrp_df = pd.DataFrame(rsrp, columns=[f'RSRP_{i+1}' for i in range(rsrp.shape[1])])
    df_aug = pd.concat([df_aug, rsrp_df], axis=1)
        
    os.makedirs('augm_data', exist_ok=True)
    out_path = f'augm_data/synthetic_{spatial_algo}_{radio_algo}.csv'
    df_aug.to_csv(out_path, index=False)
    print(f"Saved synthetic dataset to {out_path}")
    return out_path

if __name__ == "__main__":
    # Usage: python generate/full_pipeline.py [spatial radio | all]   (default: kde knn)
    args = sys.argv[1:]
    if args == ['all']:
        # All 32 spatial x radio combinations + radio KNN with K = 2..5 (ablation)
        combos = [(s, r) for s in SPATIAL_MODELS for r in RADIO_MODELS + KNN_ABLATION_MODELS]
    elif len(args) == 2:
        combos = [tuple(args)]
    else:
        combos = [('kde', 'knn')]
    for spatial_algo, radio_algo in combos:
        generate_synthetic_data(spatial_algo, radio_algo, n_samples=n_synthetic())
