import sys
import os
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import median_spatial_ks
from models.spatial.definitions import SpatialNF

def test_nf():
    print("Testing Spatial Normalizing Flow...")
    
    if not os.path.exists('models/spatial/nf.pth'):
        print("Error: Model not found.")
        return

    checkpoint = torch.load('models/spatial/nf.pth', weights_only=False)
    mean = checkpoint['mean']
    std = checkpoint['std']
    
    flow = SpatialNF(input_dim=2, n_flows=6)
    flow.load_state_dict(checkpoint['model_state_dict'])
    flow.eval()
    
    real_data = load_mdt_data('data/test.csv', mode='spatial')
    n_samples = len(real_data)
    
    def sample(seed):
        torch.manual_seed(seed)  # one reproducible draw per seed
        with torch.no_grad():
            # Sample from prior (Normal distribution)
            z = torch.randn(n_samples, 2)
            # Inverse flow: z -> x
            generated_norm = flow.inverse(z).numpy()
        return generated_norm * std + mean
    
    # Median over 10 draws
    results = median_spatial_ks(real_data, sample)
    
    print("-" * 30)
    print(f"NF Evaluation Results:")
    print(f"2-D KS (FF) : {results['ks_2d']:.4f} (p={results['p_2d']:.4f}), median of {results['n_draws']} draws")
    print("-" * 30)

if __name__ == "__main__":
    test_nf()