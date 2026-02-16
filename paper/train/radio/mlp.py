import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from models.radio.definitions import RadioMLP

def train_mlp_radio(epochs=200, batch_size=64, lr=0.001):
    print("Training Radio MLP...")
    
    # 1. Load Data
    coords, rsrps = load_mdt_data('data/train.csv', mode='radio')
    
    # Normalize Inputs (Coords)
    mean_in = coords.mean(dim=0)
    std_in = coords.std(dim=0)
    coords_norm = (coords - mean_in) / std_in
    
    # Normalize Targets (RSRP)
    mean_out = rsrps.mean(dim=0)
    std_out = rsrps.std(dim=0)
    rsrps_norm = (rsrps - mean_out) / std_out
    
    dataset = TensorDataset(coords_norm, rsrps_norm)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Model
    model = RadioMLP(input_dim=2, output_dim=rsrps.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 3. Train
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Loss: {epoch_loss/len(dataloader):.4f}")
            
    # 4. Save
    os.makedirs('models/radio', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'mean_in': mean_in, 'std_in': std_in,
        'mean_out': mean_out, 'std_out': std_out
    }, 'models/radio/mlp.pth')
    print("Model saved to models/radio/mlp.pth")

if __name__ == "__main__":
    train_mlp_radio()