import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import calculate_radio_mae
from models.radio.definitions import RadioNF

def test_nf_radio():
    torch.manual_seed(0)  # reproducible random draws
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
    
    with torch.no_grad():
        # Sample z from prior
        z = torch.randn(coords.size(0), n_cells)
        # Inverse: z, c -> x
        preds_norm = flow.inverse(z, coords_norm)
        
    preds_rsrp = preds_norm * checkpoint['std_x'] + checkpoint['mean_x']
    
    mae = calculate_radio_mae(real_rsrps.numpy(), preds_rsrp.numpy())
    print(f"NF Radio MAE: {mae:.2f} dB")

if __name__ == "__main__":
    test_nf_radio()