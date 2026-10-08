import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import calculate_radio_mae
from models.radio.definitions import RadioNF

def test_nf_radio(n_draws=100):
    print("Testing Radio NF...")
    
    if not os.path.exists('models/radio/nf.pth'):
        print("Error: Model not found.")
        return

    checkpoint = torch.load('models/radio/nf.pth')
    
    n_cells = len(checkpoint['mean_x'])
    flow = RadioNF(input_dim=n_cells, condition_dim=2, n_flows=8)
    flow.load_state_dict(checkpoint['model_state_dict'])
    flow.eval()
    
    coords, real_rsrps = load_mdt_data('data/test.csv', mode='radio')
    coords_norm = (coords - checkpoint['mean_c']) / checkpoint['std_c']
    
    # NF is stochastic: report the MAE of one draw (what the augmentation uses) and the
    # MAE of the mean over n_draws (its radio-map estimate, comparable to KNN/RF/GPR/MLP)
    with torch.no_grad():
        # Sample z from prior, then inverse: z, c -> x
        draws = torch.stack([flow.inverse(torch.randn(coords.size(0), n_cells), coords_norm) for _ in range(n_draws)])
        
    draws_rsrp = draws * checkpoint['std_x'] + checkpoint['mean_x']
    
    mae_single = calculate_radio_mae(real_rsrps.numpy(), draws_rsrp[0].numpy())
    mae_mean = calculate_radio_mae(real_rsrps.numpy(), draws_rsrp.mean(dim=0).numpy())
    print(f"NF Radio MAE (single draw): {mae_single:.4f} dBm")
    print(f"NF Radio MAE (mean of {n_draws} draws): {mae_mean:.4f} dBm")

if __name__ == "__main__":
    test_nf_radio()