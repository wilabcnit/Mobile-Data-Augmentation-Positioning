import torch
import torch.nn as nn
import numpy as np

# --- GAN COMPONENTS ---
class SpatialGenerator(nn.Module):
    def __init__(self, latent_dim=10, output_dim=2):
        super(SpatialGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim) # Outputs (Lat, Lon)
        )

    def forward(self, z):
        return self.net(z)

class SpatialDiscriminator(nn.Module):
    def __init__(self, input_dim=2):
        super(SpatialDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid() # Probability real
        )

    def forward(self, x):
        return self.net(x)

# --- NORMALIZING FLOW COMPONENTS (RealNVP-like) ---
class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(AffineCouplingLayer, self).__init__()
        self.input_dim = input_dim
        # Split input into two halves
        self.split_dim = input_dim // 2
        
        # Scale and Translation networks (MLPs)
        # Note: We clamp scale to avoid exploding gradients
        self.scale_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.split_dim),
            nn.Tanh() 
        )
        self.trans_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.split_dim)
        )

    def forward(self, x, reverse=False):
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        
        if not reverse:
            # Forward: x -> z
            s = self.scale_net(x1)
            t = self.trans_net(x1)
            z1 = x1
            z2 = x2 * torch.exp(s) + t
            log_det = torch.sum(s, dim=1)
            return torch.cat([z1, z2], dim=1), log_det
        else:
            # Inverse: z -> x
            z1, z2 = x1, x2
            s = self.scale_net(z1)
            t = self.trans_net(z1)
            x1 = z1
            x2 = (z2 - t) * torch.exp(-s)
            return torch.cat([x1, x2], dim=1)

class SpatialNF(nn.Module):
    def __init__(self, input_dim=2, n_flows=4):
        super(SpatialNF, self).__init__()
        self.flows = nn.ModuleList([AffineCouplingLayer(input_dim) for _ in range(n_flows)])
        
    def forward(self, x):
        # x -> z (latent)
        log_det_sum = 0
        for flow in self.flows:
            x, log_det = flow(x)
            log_det_sum += log_det
            # Flip dimensions to mix information between layers
            x = x.flip(dims=(1,))
        return x, log_det_sum

    def inverse(self, z):
        # z -> x (data)
        for i in range(len(self.flows) - 1, -1, -1):
            z = z.flip(dims=(1,))
            z = self.flows[i](z, reverse=True)
        return z