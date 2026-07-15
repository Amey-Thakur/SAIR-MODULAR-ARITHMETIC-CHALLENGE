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

This repository hosts the weights for the **SAIR Modular Arithmetic Challenge**. The model is an autoregressive transformer built to solve exact modular multiplication `(A × B) mod P` organically, without hardcoded arithmetic logic, symbolic parsers, or external computation modules.

Using state-machine decoding, this research targets the learnability wall of transformer mathematics, achieving theoretical infinite length generalization.

---

## Architectural Methods

To bypass the spatial limitations of standard transformers, this model uses three methods:

### 1. Abacus Significance Embeddings
Standard transformers track coordinate positions. This model strips coordinate embeddings, replacing them with Mathematical Significance Injections. Digits route based on their place-value, ensuring 1024-bit primes process through the exact same logic gates as 16-bit primes.

### 2. Algorithmic Scratchpads (Bit-Serial Decoding)
The network operates as a recurrent state machine. By forcing the model to generate intermediate computational traces autoregressively, the network allocates computation proportionally to integer complexity, mimicking a Turing machine tape.

### 3. Grokking Phase Transitions
The weights deployed here were captured after the grokking phase transition. The model trained through thousands of delayed gradient steps beyond the initial validation plateau with extreme weight decay, forcing the network to collapse memorization circuits into sparse mathematical algorithms.

---

## Inference Format

This model requires a specific execution format. 
If prompted with an equation, it sequentially emits the step-by-step logic trace before terminating natively with the final matrix node.

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

## Citations and Laboratory

This model is an isolated artifact. The complete research laboratory - containing the synthetic data generators, the training loops, the sandbox validators, and the PyTorch implementations - is open-source.

**Official Research Repository**:  
[SAIR-MODULAR-ARITHMETIC-CHALLENGE](https://github.com/Amey-Thakur/SAIR-MODULAR-ARITHMETIC-CHALLENGE)

*Prepared by Amey Thakur for the SAIR Foundation AI Benchmark.*
