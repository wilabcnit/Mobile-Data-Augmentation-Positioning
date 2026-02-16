import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from models.radio.definitions import VelocityField

def train_reflow_radio(epochs=300, batch_size=64, lr=0.001):
    print("Training Radio Rectified Flow...")
    
    # 1. Data
    coords, rsrps = load_mdt_data('data/train.csv', mode='radio')
    
    mean_c = coords.mean(dim=0)
    std_c = coords.std(dim=0)
    coords_norm = (coords - mean_c) / std_c
    
    mean_x = rsrps.mean(dim=0)
    std_x = rsrps.std(dim=0)
    rsrps_norm = (rsrps - mean_x) / std_x
    
    dataset = TensorDataset(coords_norm, rsrps_norm)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Model (Velocity Field)
    model = VelocityField(input_dim=rsrps.shape[1], condition_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    # 3. Training Loop
    for epoch in range(epochs):
        epoch_loss = 0
        for c, x1 in dataloader:
            batch_len = x1.size(0)
            
            optimizer.zero_grad()
            
            # x0 = N(0, I) (Noise)
            x0 = torch.randn_like(x1)
            
            # t = U[0, 1]
            t = torch.rand(batch_len, 1)
            
            # Interpolation: x_t = t * x1 + (1-t) * x0
            xt = t * x1 + (1 - t) * x0
            
            # Target Velocity: x1 - x0
            target_v = x1 - x0
            
            # Predict Velocity
            pred_v = model(xt, t, c)
            
            loss = criterion(pred_v, target_v)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch} | MSE Loss: {epoch_loss/len(dataloader):.4f}")
            
    # 4. Save
    os.makedirs('models/radio', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'mean_c': mean_c, 'std_c': std_c,
        'mean_x': mean_x, 'std_x': std_x
    }, 'models/radio/rec_flow.pth')
    print("Model saved to models/radio/rec_flow.pth")

if __name__ == "__main__":
    train_reflow_radio()