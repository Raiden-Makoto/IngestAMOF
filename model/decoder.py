import torch #type: ignore
import torch.nn as nn #type: ignore
from layers.EGNNLayer import EGNNLayer

class DenoisingDecoder(nn.Module):
    def __init__(self, hidden_dim=64, latent_dim=64, num_layers=3):
        super().__init__()
        self.atom_embedding = nn.Embedding(100, hidden_dim)
        
        # Condition Injectors
        self.time_mlp = nn.Sequential(
            nn.Linear(64, hidden_dim), # Time embedding size is usually 64
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.latent_mlp = nn.Linear(latent_dim, hidden_dim)
        
        # Use EGNN Layers
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, atom_types, frac_coords, lattice, mask, t_emb, z):
        # 1. Embed Types
        h = self.atom_embedding(atom_types)
        
        # 2. Inject Conditioning (Add to every node)
        # t_emb: (Batch, 64) -> (Batch, Hidden)
        # z:     (Batch, Latent) -> (Batch, Hidden)
        cond = self.time_mlp(t_emb) + self.latent_mlp(z)
        
        # Expand conditioning to all atoms
        # (Batch, 1, Hidden)
        cond = cond.unsqueeze(1) 
        h = h + cond 
        
        # 3. Run EGNN to update Coordinates
        total_coord_shift = 0
        
        for layer in self.layers:
            h, coord_shift = layer(h, frac_coords, lattice, mask)
            total_coord_shift = total_coord_shift + coord_shift
            
        # The output is the total CARTESIAN displacement
        return total_coord_shift, None # (No type prediction for now)