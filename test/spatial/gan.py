import sys
import os
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import mean_spatial_ks
from models.spatial.definitions import SpatialGenerator

def test_gan():
    print("Testing Spatial GAN...")
    
    if not os.path.exists('models/spatial/gan.pth'):
        print("Error: Model not found.")
        return

    # 1. Load Checkpoint
    checkpoint = torch.load('models/spatial/gan.pth', weights_only=False)
    latent_dim = checkpoint['latent_dim']
    mean = checkpoint['mean']
    std = checkpoint['std']
    
    G = SpatialGenerator(latent_dim=latent_dim)
    G.load_state_dict(checkpoint['model_state_dict'])
    G.eval()
    
    # 2. Load Real Test Data
    real_data = load_mdt_data('data/test.csv', mode='spatial')
    n_samples = len(real_data)
    
    # 3. Generate (one reproducible draw per seed)
    def sample(seed):
        torch.manual_seed(seed)
        with torch.no_grad():
            z = torch.randn(n_samples, latent_dim)
            generated_norm = G(z).numpy()
        # Denormalize
        return generated_norm * std + mean
    
    # 4. Evaluate: mean over 10 draws
    results = mean_spatial_ks(real_data, sample)
    
    print("-" * 30)
    print(f"GAN Evaluation Results:")
    p_txt = f"{results['p_2d']:.2f}" if results['p_2d'] >= 0.01 else "<0.01"
    print(f"2-D KS (FF) : {results['ks_2d']:.2f} (p={p_txt}), mean of {results['n_draws']} draws")
    print("-" * 30)

if __name__ == "__main__":
    test_gan()