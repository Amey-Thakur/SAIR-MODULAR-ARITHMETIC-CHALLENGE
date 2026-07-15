---
language:
- en
pipeline_tag: text-generation
tags:
- mathematics
- modular-arithmetic
- grokking
- scratchpad
- length-generalization
- transformers
license: cc-by-4.0
---

# SAIR Modular Arithmetic Challenge: Abacus Generalization

[![GitHub Repository](https://img.shields.io/badge/GitHub-SAIR_Modular_Arithmetic-181717?logo=github)](https://github.com/Amey-Thakur/SAIR-MODULAR-ARITHMETIC-CHALLENGE)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

This repository hosts the canonical weights for the **SAIR Modular Arithmetic Challenge**. The model is an autoregressive transformer purpose-built to solve exact modular multiplication `(A × B) mod P` entirely organically, strictly avoiding hardcoded arithmetic logic, symbolic parsers, or external computation modules.

By enforcing rigid adherence to state-machine decoding, this research directly targets the "learnability wall" of transformer mathematics, achieving theoretical infinite length generalization.

---

## Architectural Innovations

To bypass the inherent spatial permutation failures of standard transformers, this model integrates three distinct architectural principles:

### 1. Abacus Significance Embeddings
Standard transformers track coordinate positions (e.g., Token 3 is in Position 3). This model entirely strips coordinate embeddings, replacing them with **Mathematical Significance Injections**. Digits are routed based on their place-value (e.g., 100s, 10s, 1s), ensuring 1024-bit primes are processed through the exact same logic gates as 16-bit primes.

### 2. Algorithmic Scratchpads (Bit-Serial Decoding)
The network operates as a recurrent state machine. By forcing the model to generate intermediate computational traces (Horner's method) autoregressively, the network allocates computation proportionally to integer complexity, mimicking a Turing machine's tape.

### 3. Grokking Phase Transitions
The weights deployed here are captured *after* the grokking phase transition. The model was trained through thousands of delayed gradient steps beyond the initial validation plateau with extreme weight decay (WD=1.0), coercing the network to collapse memorization circuits into sparse mathematical algorithms.

---

## Inference Contract

This model is constrained by a strict execution sandbox. 
If prompted with an equation, it will sequentially emit the step-by-step logic trace before terminating natively with the final matrix node.

**Input Format**: Character-level ASCII equations.
```text
123*456
```

**Output Format**: State machine algorithmic trace terminating in `ANS=`.
```text
(scratchpad logic...) ANS=56088<EOS>
```

---

## Technical Specifications

- **Architecture Layer**: Bit-Serial Autoregressive Transformer
- **Embedding Mechanism**: Significance Input Injection (Abacus)
- **Tokenization**: Discrete Character-level (`Base10Tokenizer`)
- **Framework Ecosystem**: PyTorch 2.0+ 

---

## Citations & Laboratory

This model is an isolated deployment artifact. The complete research laboratory—containing the synthetic data generators, the curriculum training loops, the AST sandbox validators, and the `AbacusLayer` PyTorch implementations—is entirely open-source.

**Official Research Repository**:  
[SAIR-MODULAR-ARITHMETIC-CHALLENGE](https://github.com/Amey-Thakur/SAIR-MODULAR-ARITHMETIC-CHALLENGE)

*Prepared by Amey Thakur for the SAIR Foundation AI Benchmark.*
