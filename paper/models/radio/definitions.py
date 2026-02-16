import torch
import torch.nn as nn

# --- MLP COMPONENT ---
class RadioMLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=30):
        super(RadioMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- cGAN COMPONENTS ---
class RadioGenerator(nn.Module):
    def __init__(self, condition_dim=2, latent_dim=10, output_dim=30):
        super(RadioGenerator, self).__init__()
        # Input: Noise (z) + Condition (Lat, Lon)
        self.net = nn.Sequential(
            nn.Linear(condition_dim + latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, z, condition):
        x = torch.cat([z, condition], dim=1)
        return self.net(x)

class RadioDiscriminator(nn.Module):
    def __init__(self, condition_dim=2, input_dim=30):
        super(RadioDiscriminator, self).__init__()
        # Input: Real/Fake Data (RSRP) + Condition (Lat, Lon)
        self.net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, data, condition):
        x = torch.cat([data, condition], dim=1)
        return self.net(x)
        
# --- CONDITIONAL NORMALIZING FLOW (cNF) ---
class ConditionalAffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim=64):
        super(ConditionalAffineCouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.split_dim = input_dim // 2
        
        # Scale and Translation networks take (x_half + condition) as input
        self.net_input_dim = self.split_dim + condition_dim
        
        self.scale_net = nn.Sequential(
            nn.Linear(self.net_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.split_dim),
            nn.Tanh()
        )
        self.trans_net = nn.Sequential(
            nn.Linear(self.net_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.split_dim)
        )

    def forward(self, x, c, reverse=False):
        # x: input data (RSRP), c: condition (Lat, Lon)
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        
        # Concatenate x1 with condition c for the coupling
        net_input = torch.cat([x1, c], dim=1)
        
        if not reverse:
            # Forward: x -> z
            s = self.scale_net(net_input)
            t = self.trans_net(net_input)
            z1 = x1
            z2 = x2 * torch.exp(s) + t
            log_det = torch.sum(s, dim=1)
            return torch.cat([z1, z2], dim=1), log_det
        else:
            # Inverse: z -> x
            z1, z2 = x1, x2
            s = self.scale_net(net_input)
            t = self.trans_net(net_input)
            x1 = z1
            x2 = (z2 - t) * torch.exp(-s)
            return torch.cat([x1, x2], dim=1)

class RadioNF(nn.Module):
    def __init__(self, input_dim=30, condition_dim=2, n_flows=6):
        super(RadioNF, self).__init__()
        self.flows = nn.ModuleList([
            ConditionalAffineCouplingLayer(input_dim, condition_dim) 
            for _ in range(n_flows)
        ])
        
    def forward(self, x, c):
        log_det_sum = 0
        for flow in self.flows:
            x, log_det = flow(x, c)
            log_det_sum += log_det
            x = x.flip(dims=(1,)) # Permute
        return x, log_det_sum

    def inverse(self, z, c):
        for i in range(len(self.flows) - 1, -1, -1):
            z = z.flip(dims=(1,))
            z = self.flows[i](z, c, reverse=True)
        return z

# --- RECTIFIED FLOW (ReFlow) ---
class VelocityField(nn.Module):
    def __init__(self, input_dim=30, condition_dim=2, time_dim=1):
        super(VelocityField, self).__init__()
        # Inputs: x (state), t (time), c (condition)
        self.net = nn.Sequential(
            nn.Linear(input_dim + condition_dim + time_dim, 128),
            nn.SiLU(), # Swish activation
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, input_dim) # Outputs velocity vector
        )

    def forward(self, x, t, c):
        # Ensure t has shape [Batch, 1]
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        net_input = torch.cat([x, t, c], dim=1)
        return self.net(net_input)