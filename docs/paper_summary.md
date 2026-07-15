# SAIR Modular Arithmetic Challenge: Algorithmic Generalization and Grokking in Language Models

**Abstract**
This paper outlines the technical and theoretical foundation of our repository for the SAIR Modular Arithmetic Challenge. We detail the mechanisms of length generalization and grokking in a highly restricted sandbox environment where access to arbitrary precision modules is banned. By leveraging a bit-serial algorithmic decoder combined with Curriculum Learning and Horner's Method scratchpads, we demonstrate that a purely neural causal decoder can accurately execute modulo arithmetic beyond its training distribution.

## 1. Introduction
Modular arithmetic, specifically $(A \times B) \pmod{P}$, requires both vast numerical storage scaling exponentially with bit-length, and strict sequential logical evaluation. Standard dense Transformers often fail out-of-distribution (OOD) because they memorize rather than compute. 

Our laboratory introduces a three-tiered curriculum:
1. **The Grokking Baseline**: Heavy weight-decay, delayed optimization phase transitions, and RoPE positional encodings to induce true mathematical structures in the network's latent space for small (16-32 bit) equations.
2. **The Algorithmic Decoder**: Instead of outputting the result directly, the model is trained to output a "Scratchpad" – a step-by-step arithmetic trace mimicking Horner's Method.
3. **The Mixed-Radix Router**: An MoE (Mixture of Experts) router that dynamically delegates to the baseline for fast computation or the algorithmic decoder for extensive logical sequences.

## 2. Experimental Setup
We designed our tokenization without Byte Pair Encoding (BPE), falling back to character-level Base10 representation. This prevents token fragmentation that usually plagues math LLMs. 
We generate perfectly uniform primes using a custom Miller-Rabin generator implementation to ensure unbiased OOD distributions across exact bit boundaries (up to 256-bit).

## 3. Results and Deployment
Using the `train.py` harness integrated with TensorBoard, we observed rapid convergence on the Curriculum dataset. The final architecture is exported via Hugging Face `safetensors` and processed by our custom AST validator to guarantee adherence to competition sandboxing (banning the `%` operator entirely).

## 4. Conclusion
The repository serves as a robust proof-of-concept for neural computation of algorithmic targets. Future work will entail scaling the parameter count and training on full multi-node GPU clusters to reach the 4096-bit tier required by the hardest SAIR challenge constraints.
