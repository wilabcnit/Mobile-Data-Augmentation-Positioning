import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import calculate_radio_mae
from models.radio.definitions import RadioMLP

def test_mlp_radio():
    print("Testing Radio MLP...")
    
    if not os.path.exists('models/radio/mlp.pth'):
        print("Error: Model not found.")
        return

    checkpoint = torch.load('models/radio/mlp.pth')
    model = RadioMLP(input_dim=2, output_dim=30) # Assuming 30 cells
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    coords, real_rsrps = load_mdt_data('data/test.csv', mode='radio')
    
    # Normalize Input
    coords_norm = (coords - checkpoint['mean_in']) / checkpoint['std_in']
    
    with torch.no_grad():
        preds_norm = model(coords_norm)
        
    # Denormalize Output
    preds_rsrp = preds_norm * checkpoint['std_out'] + checkpoint['mean_out']
    
    mae = calculate_radio_mae(real_rsrps.numpy(), preds_rsrp.numpy())
    print(f"MLP Radio MAE: {mae:.4f} dBm")

if __name__ == "__main__":
    test_mlp_radio()