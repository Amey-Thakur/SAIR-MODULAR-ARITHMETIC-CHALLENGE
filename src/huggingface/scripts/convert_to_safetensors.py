import torch
from safetensors.torch import save_file
import sys
import os

def convert_pt_to_safetensors(pt_path: str, st_path: str):
    """Converts a PyTorch .pt or .pth checkpoint to a .safetensors file."""
    print(f"Loading weights from {pt_path}...")
    if not os.path.exists(pt_path):
        print("Error: Input file does not exist.")
        return False
        
    state_dict = torch.load(pt_path, map_location='cpu')
    
    print(f"Saving to {st_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(st_path)), exist_ok=True)
    save_file(state_dict, st_path)
    print("Conversion complete!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_to_safetensors.py <input.pt> <output.safetensors>")
        sys.exit(1)
        
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    
    success = convert_pt_to_safetensors(in_file, out_file)
    sys.exit(0 if success else 1)
