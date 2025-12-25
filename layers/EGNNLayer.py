import torch
import torch.nn as nn
from layers.RBFExpansion import RBFExpansion

def get_periodic_diff(frac_coords, lattice):
    """
    Calculates the vector arrows (dx, dy, dz) between all atoms
    respecting Periodic Boundary Conditions.
    """
    # 1. Difference Matrix: pos[i] - pos[j]
    # Shape: (Batch, N, N, 3)
    diff = frac_coords.unsqueeze(2) - frac_coords.unsqueeze(1)
    
    # 2. Wrap around the box (Minimum Image Convention)
    diff = diff - torch.round(diff)
    
    # 3. Project to Cartesian (Angstroms)
    # This is vital: The model must learn in real physical space
    # diff: (B, N, N, 3), lattice: (B, 3, 3)
    # We need to do batch matrix multiplication: (B, N, N, 3) @ (B, 3, 3) -> (B, N, N, 3)
    # Reshape diff to (B*N*N, 3), multiply with lattice, then reshape back
    B, N, _, _ = diff.shape
    diff_flat = diff.view(B * N * N, 3)  # (B*N*N, 3)
    lattice_flat = lattice.view(B, 3, 3)  # (B, 3, 3)
    # For each batch, multiply all (N*N, 3) with (3, 3)
    cart_diff_flat = torch.bmm(
        diff_flat.view(B, N * N, 3),
        lattice_flat
    )  # (B, N*N, 3)
    cart_diff = cart_diff_flat.view(B, N, N, 3)  # (B, N, N, 3)
    
    # 4. Get Distances
    dist_sq = torch.sum(cart_diff**2, dim=-1, keepdim=True)
    dist = torch.sqrt(dist_sq + 1e-6)
    
    return cart_diff, dist

class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        # We need to initialize the RBF expander here (or pass it in)
        # Let's assume we use the same settings as before (vmax=8.0, bins=32)
        # You can import RBFExpansion from your utils
        self.rbf_fn = RBFExpansion(vmin=0, vmax=8.0, bins=32)
        
        # 1. Message Net Input: 
        # Node_i (H) + Node_j (H) + RBF_Distance (32)
        input_dim = (hidden_dim * 2) + 32 
        
        self.message_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. Coordinate Net Input: Message (H)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # Outputs Vector Weight
        )
        
        # 3. Node Net Input: Node (H) + Message_Sum (H)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 4. Layer Normalization (Critical for deep networks)
        # Normalizes node features to prevent signal degradation in deep GNNs
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, frac_coords, lattice, mask):
        # A. Get Vectors and Distances
        # vectors: (B, N, N, 3) <-- These have length = distance
        # dist:    (B, N, N)
        vectors, dist = get_periodic_diff(frac_coords, lattice)
        dist = dist.squeeze(-1)
        
        # --- FIX 1: NORMALIZE VECTORS ---
        # Turn "Distance Vectors" into "Unit Direction Vectors"
        # We add 1e-6 to avoid dividing by zero if atoms overlap
        unit_vectors = vectors / (dist.unsqueeze(-1) + 1e-6)
        # --------------------------------
        
        # B. RBF Expansion (Scalar Info)
        rbf_feat = self.rbf_fn(dist) # (B, N, N, 32)
        
        # C. Create Messages
        N = h.size(1)
        h_i = h.unsqueeze(2).repeat(1, 1, N, 1)
        h_j = h.unsqueeze(1).repeat(1, N, 1, 1)
        
        # Concatenate Features + RBF
        edge_input = torch.cat([h_i, h_j, rbf_feat], dim=-1)
        m_ij = self.message_mlp(edge_input) * mask.unsqueeze(1).unsqueeze(-1)
        
        # D. Update Coords (Using Vectors with Softmax Attention)
        coord_weights = self.coord_mlp(m_ij)  # (B, N, N, 1)
        
        # Softmax Attention (Keep this from last time)
        mask_matrix = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
        coord_weights = coord_weights.masked_fill(mask_matrix.unsqueeze(-1) == 0, -1e9)
        attn = torch.softmax(coord_weights, dim=2)
        
        # --- USE UNIT VECTORS HERE ---
        # Old: sum(attn * vectors)  <-- The "Lever Arm" Bug
        # New: sum(attn * unit_vectors)
        coord_update = torch.sum(attn * unit_vectors, dim=2)
        # -----------------------------
        
        # Still clamp it for safety (physical constraint: max 0.5 Å per layer)
        coord_update = torch.clamp(coord_update, min=-0.5, max=0.5)
        # Apply mask to ensure padding atoms don't move
        coord_update = coord_update * mask.unsqueeze(-1).float()
        
        # E. Update Features (Using Messages)
        m_i = torch.sum(m_ij, dim=2)
        h_new = self.node_mlp(torch.cat([h, m_i], dim=-1))
        h = h + h_new  # Residual connection
        
        # F. Layer Normalization (Critical for deep networks)
        # Normalize node features to maintain signal-to-noise ratio
        # LayerNorm normalizes over the feature dimension (last dim)
        h = self.layer_norm(h)
        
        return h, coord_update