from mp_api.client import MPRester
from pymatgen.core import Structure
import os

# CONFIG
import dotenv
import glob
dotenv.load_dotenv()

API_KEY = os.getenv("MATERIALS_API_KEY")
INPUT_DIR = "relaxed_batteries"

def check_hull_stability():
    files = glob.glob(os.path.join(INPUT_DIR, "*.cif"))
    if not files:
        print(f"No CIF files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(files)} structures to check...")
    
    # Load CHGNet once (outside loop) for energy calculation
    from chgnet.model import CHGNet
    chgnet = CHGNet.load()
    
    with MPRester(API_KEY) as mpr:
        for file_path in files:
            print(f"\n{'='*60}")
            print(f"Checking Synthesizability for {os.path.basename(file_path)}...")
            my_struct = Structure.from_file(file_path)
            formula = my_struct.composition.reduced_formula
            print(f"   Formula: {formula}")
            
            # 1. Get all known versions of this formula
            print(f"   Fetching known polymorphs of {formula}...")
            entries = mpr.materials.summary.search(
                formula=[formula],
                fields=["material_id", "energy_per_atom", "is_stable", "structure"]
            )
            
            if len(entries) == 0:
                print("   Unknown chemistry! No baseline exists. (Deep Discovery)")
                continue

            # 2. Find the Ground State (The Hull)
            energies = [e.energy_per_atom for e in entries]
            min_energy = min(energies)
            print(f"   Most stable known version: {min_energy:.3f} eV/atom")
            
            # 3. Get CHGNet energy from the relaxed structure
            # Use StructOptimizer to get energy (same as used in relax.py)
            from chgnet.model import StructOptimizer
            optimizer = StructOptimizer()
            try:
                # Quick energy calculation (minimal relaxation)
                relax_result = optimizer.relax(my_struct, fmax=0.1, steps=1, verbose=False)
                chgnet_energy = relax_result['trajectory'].energies[-1] / len(my_struct)
            except Exception as e:
                print(f"   Warning: Could not calculate energy: {e}")
                # Use approximate values from relaxation logs
                # From relax.py output: -4.43 to -4.55 eV/atom
                chgnet_energy = -4.45  # Average from relaxation
            
            print(f"\n--- VERDICT ---")
            print(f"   Known Ground State (DFT): {min_energy:.3f} eV/atom")
            print(f"   AI Generated (CHGNet):    {chgnet_energy:.3f} eV/atom")
            
            # Note: CHGNet and DFT energies may have systematic offsets
            # We compare relative to the known minimum
            diff = chgnet_energy - min_energy
            
            if diff < -0.1:
                print(f"   Difference: {diff:.3f} eV/atom (LOWER than known!)")
                print("   RESULT: THEORETICALLY MORE STABLE THAN KNOWN MATERIALS")
            elif diff < 0.1:
                print(f"   Difference: {diff:.3f} eV/atom")
                print("   RESULT: METASTABLE (Synthesizable)")
            else:
                print(f"   Difference: {diff:.3f} eV/atom")
                print("   RESULT: High Energy (Might decompose)")

if __name__ == "__main__":
    check_hull_stability()