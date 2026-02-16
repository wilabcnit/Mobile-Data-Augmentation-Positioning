import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Adjust path to find modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data
from models.spatial.definitions import SpatialGenerator, SpatialDiscriminator

def train_gan(epochs=1000, batch_size=64, lr=0.0002):
    print("Training Spatial GAN...")
    
    # 1. Load Data
    data = load_mdt_data('data/train.csv', mode='spatial')
    # Normalize data for easier GAN training
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data_norm = (data - mean) / std
    
    dataset = TensorDataset(torch.FloatTensor(data_norm))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Init Models
    latent_dim = 10
    G = SpatialGenerator(latent_dim=latent_dim)
    D = SpatialDiscriminator()
    
    # Optimizers
    opt_G = optim.Adam(G.parameters(), lr=lr)
    opt_D = optim.Adam(D.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # 3. Training Loop
    for epoch in range(epochs):
        for real_imgs in dataloader:
            real_imgs = real_imgs[0]
            batch_len = real_imgs.size(0)
            
            # --- Train Discriminator ---
            opt_D.zero_grad()
            
            # Real loss
            real_labels = torch.ones(batch_len, 1)
            output_real = D(real_imgs)
            loss_real = criterion(output_real, real_labels)
            
            # Fake loss
            z = torch.randn(batch_len, latent_dim)
            fake_imgs = G(z)
            fake_labels = torch.zeros(batch_len, 1)
            output_fake = D(fake_imgs.detach()) # Detach to stop G gradients
            loss_fake = criterion(output_fake, fake_labels)
            
            d_loss = loss_real + loss_fake
            d_loss.backward()
            opt_D.step()
            
            # --- Train Generator ---
            opt_G.zero_grad()
            output_fake_for_G = D(fake_imgs)
            # Generator wants Discriminator to think these are real
            g_loss = criterion(output_fake_for_G, real_labels) 
            g_loss.backward()
            opt_G.step()
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
            
    # 4. Save Model & Normalization Stats
    os.makedirs('models/spatial', exist_ok=True)
    torch.save({
        'model_state_dict': G.state_dict(),
        'mean': mean,
        'std': std,
        'latent_dim': latent_dim
    }, 'models/spatial/gan.pth')
    print("Model saved to models/spatial/gan.pth")

if __name__ == "__main__":
    train_gan()