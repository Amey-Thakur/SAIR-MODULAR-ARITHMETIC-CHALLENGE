# Dataset Generation Roadmap

Neural networks require massive amounts of data to overcome the memorization barrier and grok the underlying mathematics. We cannot use standard text corpora; the data must be entirely synthetic and highly structured.

## 1. Distribution Requirements
The dataset must uniformly sample the space of:
- **Modulus ($P$)**: Prime numbers of varying bit lengths.
- **Operands ($A, B$)**: Integers drawn uniformly from $[0, P-1]$.

## 2. Generation Stages

### Stage 1: The Prime Generator
We need a highly optimized script (using `sympy.randprime` or a custom Miller-Rabin implementation) to generate primes at exact bit boundaries (e.g., exactly 64 bits, exactly 256 bits).
- **Deliverable**: `datasets/generators/prime_gen.py`

### Stage 2: The Direct Formulation Dataset
For the naive models (Tier 1).
- **Format**: `Input: A * B % P = | Output: R`
- **Volume**: ~10 Million unique triplets $(A, B, P)$ per bit-length tier.
- **Storage**: Parquet files or compressed JSONL to minimize disk I/O during dataloading.

### Stage 3: The Curriculum Dataset
To help the model learn the basic operations before doing full modular multiplication.
- **Sub-task 1 (Addition)**: `A + B % P = R`
- **Sub-task 2 (Multiplication)**: `A * B = R` (No modulo)
- **Sub-task 3 (Modulo)**: `X % P = R`

### Stage 4: The Scratchpad Dataset
This is the most complex generation task. The script must natively execute Horner's method and log every intermediate step into a sequence of tokens.
- **Algorithm**:
  ```python
  def generate_scratchpad(A, B, P):
      binary_B = bin(B)[2:]
      history = []
      acc = 0
      for bit in binary_B:
          acc = (acc * 2) % P
          history.append(f"ACC2={acc}")
          if bit == '1':
              acc = (acc + A) % P
              history.append(f"ADD={acc}")
      history.append(f"ANS={acc}")
      return history
  ```
- **Deliverable**: `datasets/generators/scratchpad_gen.py`

## 3. Tokenizer Integration
The dataset loader must be tightly coupled with the tokenizer. We will implement a custom PyTorch `Dataset` that bypasses standard NLP tokenizers (like Hugging Face `tokenizers`) and directly maps characters/bytes to integer IDs to ensure $O(1)$ tokenization time during training.
