# Open Research Questions

While the roadmap provides a structured path forward, several fundamental research questions remain unresolved. Addressing these could provide a decisive advantage in the SAIR Modular Arithmetic Challenge.

## 1. What is the optimal base/radix for scratchpads?
- **Binary (Base-2)**: Mathematically simplest. Horner's method requires only shift (multiply by 2) and conditional add. However, sequence lengths explode. A 1024-bit prime requires thousands of tokens per calculation.
- **Base-10**: Familiar, but mathematically arbitrary. Does the model waste capacity learning the intricacies of base-10 carrying?
- **Base-256 (Byte-level)**: Reduces sequence length by $8\times$ compared to binary. Can a transformer easily learn the $256 \times 256$ multiplication table in its early layers?
- **Question**: Where is the Pareto frontier between algorithmic simplicity and sequence length?

## 2. Does grokking scale to large widths/depths?
Most literature on grokking (e.g., Power et al.) focuses on very small, 1-2 layer transformers with low hidden dimensions. 
- **Question**: If we train a 100M parameter network on 16-bit primes, will it grok faster, slower, or just overfit permanently? Does massive overparameterization hinder the discovery of the low-rank structures needed for modular arithmetic?

## 3. Can synthetic data induce algorithmic discovery without explicit scratchpads?
Currently, we assume Tier 3 (256+ bits) requires forcing the model to output a scratchpad.
- **Question**: If we provide an implicitly curriculum-structured dataset (e.g., mixing addition, small primes, and large primes), can a deep enough network learn to maintain the scratchpad *internally* within its hidden states, without emitting the tokens?

## 4. How effective are complex-valued representations?
Given that addition modulo $P$ maps cleanly to rotations on a circle (Fourier basis), would a network explicitly designed with complex-valued weights and activations drastically accelerate learning?
- **Question**: Does replacing standard linear layers with complex-linear layers (where $W \in \mathbb{C}^{d \times d}$) lower the learnability wall for modular multiplication?

## 5. Security of the Inference Contract
- **Question**: Are there edge cases in the AST analysis where standard tensor operations (e.g., utilizing `torch.fmod` on floats) might technically bypass the rules while still being executed on GPU? (Note: The competition explicitly tests against this, but the exact boundaries of the sandbox require empirical testing).
