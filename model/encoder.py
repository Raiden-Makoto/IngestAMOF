import torch #type: ignore
import torch.nn as nn #type: ignore
import math
from layers.EGNNLayer import EGNNLayer
from layers.RBFExpansion import RBFExpansion
from layers.SineEmbed import SinusoidalTimeEmbeddings

class CrystalEncoder(nn.Module):
    def __init__(self, hidden_dim=64, latent_dim=64, num_layers=3):
        super().__init__()
        self.atom_embedding = nn.Embedding(100, hidden_dim)
        
        # Use EGNN Layers
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(num_layers)
        ])
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, atom_types, frac_coords, lattice, mask):
        # 1. Embed Types
        h = self.atom_embedding(atom_types)
        
        # 2. Run EGNN
        # We pass coordinates so the model can SEE the geometry,
        # but we don't care if the encoder "moves" atoms.
        for layer in self.layers:
            h, _ = layer(h, frac_coords, lattice, mask)
            
        # 3. Global Pooling (Average over atoms)
        # Mask out padding before averaging
        mask_exp = mask.unsqueeze(-1).float()
        h_masked = h * mask_exp
        h_global = h_masked.sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        
        mu = self.fc_mu(h_global)
        log_var = self.fc_var(h_global)
        
        return mu, log_var