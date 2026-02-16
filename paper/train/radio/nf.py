import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from models.radio.definitions import RadioNF

def train_nf_radio(epochs=300, batch_size=64, lr=0.001):
    print("Training Radio NF (Conditional)...")
    
    # 1. Data Loading & Normalization
    coords, rsrps = load_mdt_data('data/train.csv', mode='radio')
    
    mean_c = coords.mean(dim=0)
    std_c = coords.std(dim=0)
    coords_norm = (coords - mean_c) / std_c
    
    mean_x = rsrps.mean(dim=0)
    std_x = rsrps.std(dim=0)
    rsrps_norm = (rsrps - mean_x) / std_x
    
    dataset = TensorDataset(coords_norm, rsrps_norm)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Model
    flow = RadioNF(input_dim=rsrps.shape[1], condition_dim=2, n_flows=8)
    optimizer = optim.Adam(flow.parameters(), lr=lr)
    
    # 3. Training Loop
    for epoch in range(epochs):
        epoch_loss = 0
        for c, x in dataloader:
            optimizer.zero_grad()
            
            # Forward: x, c -> z
            z, log_det = flow(x, c)
            
            # Loss = -LogLikelihood = - (log p(z) + log_det)
            # Prior p(z) ~ N(0, I)
            log_pz = -0.5 * torch.sum(z**2, dim=1)
            loss = -(torch.mean(log_pz + log_det))
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch} | NLL Loss: {epoch_loss/len(dataloader):.4f}")
            
    # 4. Save
    os.makedirs('models/radio', exist_ok=True)
    torch.save({
        'model_state_dict': flow.state_dict(),
        'mean_c': mean_c, 'std_c': std_c,
        'mean_x': mean_x, 'std_x': std_x
    }, 'models/radio/nf.pth')
    print("Model saved to models/radio/nf.pth")

if __name__ == "__main__":
    train_nf_radio()