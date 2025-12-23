import torch
import torch.nn as nn
import numpy as np
import os
import sys
from tqdm import tqdm
from pymatgen.core import Structure, Lattice, Element

# Add parent directory to path so we can import from utils and model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import your model
from model.cdvae import CrystalDiffusionVAE
from utils.dataloader import get_dataloader

# --- CONFIG ---
CHECKPOINT_PATH = "checkpoints/cdvae_epoch_50.pt"  # Or your latest checkpoint
OUTPUT_DIR = "generated_mofs"
NUM_SAMPLES = 5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Model Config (Must match training!)
HIDDEN_DIM = 128
LATENT_DIM = 64
NUM_LAYERS = 2
TIMESTEPS = 1000

def inverse_diffusion(model, atom_types, shape, lattice, mask, z, device):
    """
    The Reverse Process: Starts from pure noise and removes it step-by-step.
    """
    # 1. Start with pure Gaussian noise (x_T)
    x = torch.randn(shape, device=device)
    
    # Apply mask to keep padding zero
    mask_3d = mask.unsqueeze(-1).float()
    x = x * mask_3d
    
    # 2. Reverse Loop (T -> 0)
    for t in tqdm(reversed(range(0, model.num_timesteps)), desc="Sampling", total=model.num_timesteps):
        # Create time tensor (batch of 't')
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        # Predict the noise using the model
        pred_noise, _ = model.decoder(
            atom_types,
            x, lattice, mask, t_batch, z
        )
        
        # 3. Math: Remove the noise (DDPM reverse process)
        # Get alpha values
        alpha_t = model.alphas_cumprod[t]
        alpha_t_prev = model.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=device)
        
        # Compute beta_t (noise schedule)
        beta_t = 1 - (alpha_t / alpha_t_prev)
        
        # Predict x_0 from x_t and predicted noise
        sqrt_recip_alpha_t = 1.0 / torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        pred_x0 = sqrt_recip_alpha_t * (x - sqrt_one_minus_alpha_t * pred_noise)
        
        # Compute coefficients for x_{t-1}
        pred_coeff1 = torch.sqrt(alpha_t_prev) * beta_t / (1.0 - alpha_t)
        pred_coeff2 = torch.sqrt(1.0 - beta_t) * (1.0 - alpha_t_prev) / (1.0 - alpha_t)
        
        # Sample x_{t-1}
        if t > 0:
            noise = torch.randn_like(x)
            posterior_variance = beta_t * (1.0 - alpha_t_prev) / (1.0 - alpha_t)
            x = pred_coeff1 * pred_x0 + pred_coeff2 * x + torch.sqrt(posterior_variance) * noise
        else:
            x = pred_x0
            
        # Re-mask to keep geometry clean
        x = x * mask_3d
    
    return x

def save_cif(frac_coords, atom_types, lattice, filename):
    """
    Converts tensors back to a .cif file
    Args:
        frac_coords: (N, 3) fractional coordinates tensor
        atom_types: (N,) atomic numbers tensor
        lattice: (3, 3) lattice matrix tensor
    """
    # Convert to numpy
    frac_coords_np = frac_coords.cpu().numpy()
    atom_types_np = atom_types.cpu().numpy()
    lattice_np = lattice.cpu().numpy()
    
    # Filter out padding (0)
    valid_indices = atom_types_np > 0
    if not valid_indices.any():
        print(f"   Skipped {filename}: No valid atoms")
        return
        
    clean_types = atom_types_np[valid_indices]
    clean_frac_coords = frac_coords_np[valid_indices]
    
    # Convert Atomic Numbers to Symbols
    species = [Element.from_Z(int(z)) for z in clean_types]
    
    # Create Pymatgen Structure (takes fractional coords)
    struct = Structure(Lattice(lattice_np), species, clean_frac_coords, coords_are_cartesian=False)
    
    # Save
    struct.to(filename=filename, fmt="cif")
    print(f"   Saved: {filename}")

def evaluate():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        print("   Available checkpoints:")
        if os.path.exists("checkpoints"):
            for f in os.listdir("checkpoints"):
                if f.endswith(".pt"):
                    print(f"     - checkpoints/{f}")
        return
        
    print(f"Loading Model from {CHECKPOINT_PATH}...")
    
    # 1. Load Model
    model = CrystalDiffusionVAE(
        hidden_dim=HIDDEN_DIM, 
        latent_dim=LATENT_DIM, 
        num_layers=NUM_LAYERS,
        num_timesteps=TIMESTEPS,
        use_checkpoint=False  # Disable checkpointing for inference
    ).to(DEVICE)
    
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✓ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    except Exception as e:
        print(f"   ❌ Error loading checkpoint: {e}")
        return
    
    model.eval()
    
    # 2. Get Templates (Lattices + Atom Counts)
    # We use real data shapes as containers for our new generated crystals
    data_dir = "processed_graphs"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
        
    loader = get_dataloader(data_dir, batch_size=NUM_SAMPLES, shuffle=True)
    if len(loader.dataset) == 0:
        print(f"❌ No data files found in {data_dir}")
        return
        
    batch = next(iter(loader))
    
    atom_types = batch['atom_types'].to(DEVICE)
    lattice = batch['lattice'].to(DEVICE)
    mask = batch['mask'].to(DEVICE)
    
    print(f"Generating {NUM_SAMPLES} new structures...")
    
    with torch.no_grad():
        # A. Sample Random Latent Vector (z)
        # "Dream" a new blueprint
        z = torch.randn(NUM_SAMPLES, LATENT_DIM).to(DEVICE)
        
        # B. Run Inverse Diffusion
        # We reuse the atom_types from the template for now (Rearrangement Task)
        # This asks the model: "Find a STABLE configuration for these atoms."
        shape = batch['frac_coords'].shape
        generated_coords = inverse_diffusion(model, atom_types, shape, lattice, mask, z, DEVICE)
        
        # C. Save Outputs
        for i in range(NUM_SAMPLES):
            filename = os.path.join(OUTPUT_DIR, f"gen_mof_{i}.cif")
            # Shift coordinates back from [-0.5, 0.5] to [0, 1] for CIF file format
            coords_for_cif = generated_coords[i] + 0.5
            save_cif(coords_for_cif, atom_types[i], lattice[i], filename)

    print(f"\n✅ Generation Complete. Check the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    evaluate()