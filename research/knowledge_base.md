# Research Knowledge Base: Modular Arithmetic in Neural Networks

This document serves as the foundational theory reference for the repository.

## 1. Mathematical Formulation

The core problem is computing:
$R = (A \times B) \pmod{P}$

Where $A, B$, and $P$ are large integers (e.g., up to 2048 bits), and $P$ is a prime.

### Why is this hard for Neural Networks?
Standard feedforward or attention mechanisms perform continuous, floating-point geometry. Modular arithmetic is highly discontinuous. A change of 1 in the input $A$ can completely change every bit of the output $R$ due to the wrapping effect of the modulo $P$. This destroys the smooth gradients that backpropagation relies on.

## 2. Potential Mathematical Angles for Learning

### A. The Discrete Logarithm / Primitive Roots
Since $P$ is prime, the multiplicative group of integers modulo $P$, $\mathbb{Z}^*_P$, is cyclic. It has a generator $g$. 
Any element $X$ can be written as $g^x \pmod{P}$. 
Multiplication becomes addition in the exponent:
$A \times B \equiv g^a \times g^b \equiv g^{a+b} \pmod{P}$

**Neural Application**: If the network can learn the discrete logarithm mapping (input $\to$ exponent), it can perform addition, and then map back. However, computing the discrete log is notoriously hard (basis of cryptographic hardness). It's unlikely a small model can learn this for large $P$.

### B. Fourier Representations
As seen in Nanda et al.'s analysis of grokking, networks often learn continuous circle representations for addition mod $P$. Multiplication is much harder to represent this way without an exponential number of frequencies.

### C. Algorithmic Emulation (Horner's Method)
The most promising approach for large integers is teaching the network to simulate classical algorithms step-by-step.
Let $B$ be represented in binary: $B = b_n b_{n-1} \dots b_0$.
$A \times B = A \times \sum (b_i \cdot 2^i)$.
Modulo $P$ can be distributed:
$R = \sum ((A \cdot b_i \cdot 2^i) \pmod{P}) \pmod{P}$.

By emitting intermediate scratchpad tokens, the network acts as a state machine. It only needs to learn "shift by 1 (multiply by 2)" and "conditional add", followed by "conditional subtract $P$".

## 3. The "Learnability Wall"
Empirically, transformers can grok arithmetic directly (without scratchpads) up to roughly 5-7 digit numbers. Beyond that, the required internal dimension and precision explode. This boundary is the "learnability wall." 

**Consequence for the Challenge**: We cannot rely on blind, end-to-end training for the hardest tiers (e.g., 256-bit or 1024-bit). We MUST enforce a curriculum and likely a scratchpad format.

## 4. Tokenization Strategies
- **Base-10 Digit Level**: `[1, 2, 3]` -> Familiar, but mathematically arbitrary (base 10 is not special).
- **Binary/Bit Level**: `[1, 1, 1, 1, 0, 1, 1]` -> The most mathematically pure, aligns with hardware and Horner's method. Highly recommended for scratchpad generation. Sequence lengths become long ($3.3\times$ longer than base 10).
- **Character/Byte Level**: Treating inputs as raw bytes strings.

**Recommendation**: Bit-level or Byte-level, strongly right-aligned to preserve positional meaning (the $i$-th token from the right is always the $i$-th power of the base).
