import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from models.radio.definitions import RadioGenerator, RadioDiscriminator

def train_cgan_radio(epochs=500, batch_size=64, lr=0.0002):
    print("Training Radio cGAN...")
    
    # 1. Load & Normalize Data
    coords, rsrps = load_mdt_data('data/train.csv', mode='radio')
    
    mean_in = coords.mean(dim=0)
    std_in = coords.std(dim=0)
    coords_norm = (coords - mean_in) / std_in
    
    mean_out = rsrps.mean(dim=0)
    std_out = rsrps.std(dim=0)
    rsrps_norm = (rsrps - mean_out) / std_out
    
    dataset = TensorDataset(coords_norm, rsrps_norm)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Models
    latent_dim = 10
    n_cells = rsrps.shape[1]
    
    G = RadioGenerator(condition_dim=2, latent_dim=latent_dim, output_dim=n_cells)
    D = RadioDiscriminator(condition_dim=2, input_dim=n_cells)
    
    opt_G = optim.Adam(G.parameters(), lr=lr)
    opt_D = optim.Adam(D.parameters(), lr=lr)
    criterion = nn.MSELoss() 
    
    # 3. Train
    for epoch in range(epochs):
        for real_cond, real_data in dataloader:
            batch_len = real_cond.size(0)
            
            # --- Train Plain Discriminator ---
            opt_D.zero_grad()
            
            valid = torch.ones(batch_len, 1)
            fake = torch.zeros(batch_len, 1)
            
            d_real = D(real_data, real_cond)
            d_loss_real = criterion(d_real, valid)
            z = torch.randn(batch_len, latent_dim)
            fake_data = G(z, real_cond)
            d_fake = D(fake_data.detach(), real_cond)
            d_loss_fake = criterion(d_fake, fake)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_loss.backward()
            opt_D.step()
            
            # --- Train Plain Generator ---
            opt_G.zero_grad()
            g_fake = D(fake_data, real_cond)
            g_loss = criterion(g_fake, valid)
            g_loss.backward()
            opt_G.step()
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    # 4. Save
    os.makedirs('models/radio', exist_ok=True)
    torch.save({
        'model_state_dict': G.state_dict(),
        'mean_in': mean_in, 'std_in': std_in,
        'mean_out': mean_out, 'std_out': std_out,
        'latent_dim': latent_dim
    }, 'models/radio/cgan.pth')
    print("Model saved to models/radio/cgan.pth")

if __name__ == "__main__":
    train_cgan_radio()