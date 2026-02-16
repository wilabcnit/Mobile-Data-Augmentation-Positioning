import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import calculate_radio_mae
from models.radio.definitions import RadioGenerator

def test_cgan_radio():
    print("Testing Radio cGAN...")
    
    if not os.path.exists('models/radio/cgan.pth'):
        print("Error: Model not found.")
        return

    checkpoint = torch.load('models/radio/cgan.pth')
    latent_dim = checkpoint['latent_dim']
    
    G = RadioGenerator(condition_dim=2, latent_dim=latent_dim, output_dim=30)
    G.load_state_dict(checkpoint['model_state_dict'])
    G.eval()
    
    coords, real_rsrps = load_mdt_data('data/test.csv', mode='radio')
    
    coords_norm = (coords - checkpoint['mean_in']) / checkpoint['std_in']
    
    with torch.no_grad():
        z = torch.randn(coords.size(0), latent_dim)
        preds_norm = G(z, coords_norm)
        
    preds_rsrp = preds_norm * checkpoint['std_out'] + checkpoint['mean_out']
    
    mae = calculate_radio_mae(real_rsrps.numpy(), preds_rsrp.numpy())
    print(f"cGAN Radio MAE: {mae:.4f} dBm")

if __name__ == "__main__":
    test_cgan_radio()