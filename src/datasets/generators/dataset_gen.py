# ==============================================================================
# File: dataset_gen.py
# Description: Core module for SAIR Modular Arithmetic Challenge.
# Tech Stack: PyTorch 2.0+, Python 3.10+
# Author: Amey Thakur
# Profile: https://github.com/Amey-Thakur
# Repository: https://github.com/Amey-Thakur/SAIR-MODULAR-ARITHMETIC-CHALLENGE
# License: CC-BY-4.0
# Date: 2026-07-15
# ==============================================================================

import json
import random
import os
from prime_gen import get_primes_for_tier

def generate_curriculum_triplet(P: int, op_type: str):
    """
    Generates a single data point based on the curriculum type.
    op_type can be 'mul_mod', 'add_mod', 'mul_only', 'mod_only'
    """
    A = random.randrange(0, P)
    B = random.randrange(0, P)
    
    if op_type == 'mul_mod':
        ans = (A * B) % P
        return f"{A}*{B}%{P}={ans}"
    elif op_type == 'add_mod':
        ans = (A + B) % P
        return f"{A}+{B}%{P}={ans}"
    elif op_type == 'mul_only':
        ans = A * B
        return f"{A}*{B}={ans}"
    elif op_type == 'mod_only':
        ans = A % P
        return f"{A}%{P}={ans}"
    else:
        raise ValueError("Invalid operation type")

def generate_dataset(bits: int, count: int, op_type: str, output_file: str):
    """Generates a dataset and saves it as JSONL."""
    primes = get_primes_for_tier(bits, count=10) # Use 10 primes for the tier
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for _ in range(count):
            P = random.choice(primes)
            equation = generate_curriculum_triplet(P, op_type)
            f.write(json.dumps({"text": equation}) + "\n")

if __name__ == "__main__":
    print("Generating Stage 3 Curriculum (16-bit)...")
    generate_dataset(16, 10000, 'mul_only', 'data/curriculum/16bit_mul.jsonl')
    generate_dataset(16, 10000, 'mul_mod', 'data/curriculum/16bit_mul_mod.jsonl')
    print("Done!")
