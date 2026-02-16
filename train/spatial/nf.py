import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from models.spatial.definitions import SpatialNF

def train_nf(epochs=500, batch_size=64, lr=0.001):
    print("Training Spatial Normalizing Flow...")
    
    # 1. Data
    data = load_mdt_data('data/train.csv', mode='spatial')
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data_norm = (data - mean) / std
    
    dataset = TensorDataset(torch.FloatTensor(data_norm))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Model (Input dim = 2 for Lat/Lon)
    flow = SpatialNF(input_dim=2, n_flows=6)
    optimizer = optim.Adam(flow.parameters(), lr=lr)
    
    # 3. Training Loop (Maximize Log Likelihood)
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            
            optimizer.zero_grad()
            
            # Forward: x -> z
            z, log_det = flow(x)
            
            # Loss = -LogLikelihood
            # LL = log_p(z) + log_det
            # Prior p(z) is standard normal: log_p(z) = -0.5 * z^2 - C
            log_pz = -0.5 * torch.sum(z**2, dim=1) 
            loss = -(torch.mean(log_pz + log_det))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch} | NLL Loss: {total_loss / len(dataloader):.4f}")
            
    # 4. Save
    os.makedirs('models/spatial', exist_ok=True)
    torch.save({
        'model_state_dict': flow.state_dict(),
        'mean': mean,
        'std': std
    }, 'models/spatial/nf.pth')
    print("Model saved to models/spatial/nf.pth")

if __name__ == "__main__":
    train_nf()