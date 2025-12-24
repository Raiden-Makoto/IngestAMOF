from mp_api.client import MPRester #type: ignore
import pickle
import os
from tqdm import tqdm #type: ignore
from dotenv import load_dotenv #type: ignore

load_dotenv()

# --- CONFIG ---
API_KEY = os.getenv("MP_API_KEY")  # Paste your key
MAX_SITES = 50                 # Keep graphs small and fast

def download_battery_materials():
    print("🔋 Connecting to Materials Project for Battery Data...")
    
    with MPRester(API_KEY) as mpr:
        # Search for Lithium-containing compounds
        # We exclude single elements (pure Li) to find complex crystals
        print("   Querying Li-containing structures...")
        
        docs = mpr.materials.summary.search(
            elements=["Li"],
            num_elements=(2, 4),     # Binary, Ternary, or Quaternary (e.g., Li7La3Zr2O12)
            nsites=(0, MAX_SITES),   # Strict size limit for speed
            is_stable=True,          # Train on ground-truth stable phases
            fields=["structure", "material_id", "formula_pretty", "volume"]
        )
        
        print(f"   Found {len(docs)} candidate structures!")
        
        # Save locally
        os.makedirs("data", exist_ok=True)
        structures = []
        
        for doc in tqdm(docs, desc="Processing"):
            structures.append({
                "id": doc.material_id,
                "formula": doc.formula_pretty,
                "structure": doc.structure
            })
            
        output_path = "data/battery_materials.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(structures, f)
            
        print(f"✅ Saved {len(structures)} crystals to {output_path}")
        print("   Ready for training! (Remember to update your dataloader path)")

if __name__ == "__main__":
    download_battery_materials()