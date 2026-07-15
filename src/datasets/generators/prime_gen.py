# ==============================================================================
# File: prime_gen.py
# Description: Core module for SAIR Modular Arithmetic Challenge.
# Tech Stack: PyTorch 2.0+, Python 3.10+
# Author: Amey Thakur
# Profile: https://github.com/Amey-Thakur
# Repository: https://github.com/Amey-Thakur/SAIR-MODULAR-ARITHMETIC-CHALLENGE
# License: CC-BY-4.0
# Date: 2026-07-15
# ==============================================================================

import random
from typing import List

def miller_rabin(n: int, k: int = 40) -> bool:
    """
    Miller-Rabin primality test.
    k determines the accuracy of the test.
    """
    if n == 2 or n == 3:
        return True
    if n < 2 or n % 2 == 0:
        return False

    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1
        s //= 2
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, s, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def generate_n_bit_prime(bits: int) -> int:
    """Generates a random prime number of exactly `bits` length."""
    while True:
        # Generate a random odd number of `bits` length
        p = random.randrange(2**(bits - 1) + 1, 2**bits, 2)
        if miller_rabin(p):
            return p

def get_primes_for_tier(bits: int, count: int = 10) -> List[int]:
    """Generates a list of prime numbers for a specific competition tier."""
    return [generate_n_bit_prime(bits) for _ in range(count)]

if __name__ == "__main__":
    print("Testing Prime Generator...")
    print(f"Random 64-bit prime: {generate_n_bit_prime(64)}")
    print(f"Random 256-bit prime: {generate_n_bit_prime(256)}")
