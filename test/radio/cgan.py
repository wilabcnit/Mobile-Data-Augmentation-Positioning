import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import calculate_radio_mae
from models.radio.definitions import RadioGenerator

def test_cgan_radio(n_draws=100):
    torch.manual_seed(0)  # reproducible random draws
    print("Testing Radio cGAN...")
    
    if not os.path.exists('models/radio/cgan.pth'):
        print("Error: Model not found.")
        return

    checkpoint = torch.load('models/radio/cgan.pth')
    latent_dim = checkpoint['latent_dim']
    
    G = RadioGenerator(condition_dim=2, latent_dim=latent_dim, output_dim=len(checkpoint['mean_out']))
    G.load_state_dict(checkpoint['model_state_dict'])
    G.eval()
    
    coords, real_rsrps = load_mdt_data('data/test.csv', mode='radio')
    
    coords_norm = (coords - checkpoint['mean_in']) / checkpoint['std_in']
    
    # cGAN is stochastic: report the MAE of one draw (what the augmentation uses) and the
    # MAE of the mean over n_draws (its radio-map estimate, comparable to KNN/RF/GPR/MLP)
    with torch.no_grad():
        draws = torch.stack([G(torch.randn(coords.size(0), latent_dim), coords_norm) for _ in range(n_draws)])
        
    draws_rsrp = draws * checkpoint['std_out'] + checkpoint['mean_out']
    
    mae_single = calculate_radio_mae(real_rsrps.numpy(), draws_rsrp[0].numpy())
    mae_mean = calculate_radio_mae(real_rsrps.numpy(), draws_rsrp.mean(dim=0).numpy())
    print(f"cGAN Radio MAE (single draw): {mae_single:.4f} dB")
    print(f"cGAN Radio MAE (mean of {n_draws} draws): {mae_mean:.4f} dB")

if __name__ == "__main__":
    test_cgan_radio()