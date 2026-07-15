import json
import random
import os
from prime_gen import get_primes_for_tier

def generate_scratchpad(A: int, B: int, P: int) -> str:
    """
    Generates the step-by-step scratchpad for Horner's method
    multiplying A * B (mod P) in binary.
    """
    binary_B = bin(B)[2:] # e.g. '101'
    
    tokens = [f"<BOS>A={A},B={B},P={P}"]
    acc = 0
    
    for bit in binary_B:
        acc = (acc * 2) % P
        tokens.append(f"ACC2={acc}")
        if bit == '1':
            acc = (acc + A) % P
            tokens.append(f"ADD={acc}")
            
    tokens.append(f"ANS={acc}<EOS>")
    return " | ".join(tokens)

def generate_scratchpad_dataset(bits: int, count: int, output_file: str):
    """Generates a dataset of scratchpads and saves it as JSONL."""
    primes = get_primes_for_tier(bits, count=5)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for _ in range(count):
            P = random.choice(primes)
            A = random.randrange(0, P)
            B = random.randrange(0, P)
            scratchpad = generate_scratchpad(A, B, P)
            f.write(json.dumps({"scratchpad": scratchpad}) + "\n")

if __name__ == "__main__":
    print("Generating Stage 4 Scratchpad Dataset (32-bit)...")
    generate_scratchpad_dataset(32, 5000, 'data/scratchpads/32bit_scratch.jsonl')
    print("Done!")
