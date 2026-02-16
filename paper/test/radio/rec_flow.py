import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import calculate_radio_mae
from models.radio.definitions import VelocityField

def test_reflow_radio():
    print("Testing Radio Rectified Flow...")
    
    if not os.path.exists('models/radio/rec_flow.pth'):
        print("Error: Model not found.")
        return

    checkpoint = torch.load('models/radio/rec_flow.pth')
    
    model = VelocityField(input_dim=30, condition_dim=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    coords, real_rsrps = load_mdt_data('data/test.csv', mode='radio')
    coords_norm = (coords - checkpoint['mean_c']) / checkpoint['std_c']
    
    # Euler ODE Solver
    # Solve dx/dt = v(x,t) from t=0 to t=1
    steps = 20 # Number of Euler steps
    dt = 1.0 / steps
    
    with torch.no_grad():
        # Start from Noise x0
        x = torch.randn(coords.size(0), 30)
        
        for i in range(steps):
            t_value = i * dt
            t = torch.ones(coords.size(0), 1) * t_value
            
            v = model(x, t, coords_norm)
            x = x + v * dt
            
    preds_rsrp = x * checkpoint['std_x'] + checkpoint['mean_x']
    
    mae = calculate_radio_mae(real_rsrps.numpy(), preds_rsrp.numpy())
    print(f"ReFlow Radio MAE: {mae:.4f} dBm")

if __name__ == "__main__":
    test_reflow_radio()