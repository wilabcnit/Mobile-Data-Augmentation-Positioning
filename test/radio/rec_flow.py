import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from utils.metrics import calculate_radio_mae
from models.radio.definitions import VelocityField

def test_reflow_radio(n_draws=100):
    torch.manual_seed(0)  # reproducible random draws
    print("Testing Radio Rectified Flow...")
    
    if not os.path.exists('models/radio/rec_flow.pth'):
        print("Error: Model not found.")
        return

    checkpoint = torch.load('models/radio/rec_flow.pth')
    
    n_cells = len(checkpoint['mean_x'])
    model = VelocityField(input_dim=n_cells, condition_dim=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    coords, real_rsrps = load_mdt_data('data/test.csv', mode='radio')
    coords_norm = (coords - checkpoint['mean_c']) / checkpoint['std_c']
    
    # Euler ODE Solver
    # Solve dx/dt = v(x,t) from t=0 to t=1
    steps = 20 # Number of Euler steps
    dt = 1.0 / steps
    
    # ReFlow is stochastic: report the MAE of one draw (what the augmentation uses) and the
    # MAE of the mean over n_draws (its radio-map estimate, comparable to KNN/RF/GPR/MLP)
    draws = []
    with torch.no_grad():
        for _ in range(n_draws):
            # Start from Noise x0
            x = torch.randn(coords.size(0), n_cells)
            
            for i in range(steps):
                t_value = i * dt
                t = torch.ones(coords.size(0), 1) * t_value
                
                v = model(x, t, coords_norm)
                x = x + v * dt
            draws.append(x)
            
    draws_rsrp = torch.stack(draws) * checkpoint['std_x'] + checkpoint['mean_x']
    
    mae_single = calculate_radio_mae(real_rsrps.numpy(), draws_rsrp[0].numpy())
    mae_mean = calculate_radio_mae(real_rsrps.numpy(), draws_rsrp.mean(dim=0).numpy())
    print(f"ReFlow Radio MAE (single draw): {mae_single:.4f} dB")
    print(f"ReFlow Radio MAE (mean of {n_draws} draws): {mae_mean:.4f} dB")

if __name__ == "__main__":
    test_reflow_radio()