# Experiment Roadmap

This roadmap defines the specific, isolated experiments we will conduct to validate our hypotheses regarding modular arithmetic learning.

## Experiment Track A: The Grokking Threshold
**Goal**: Identify exactly where the "learnability wall" occurs for a standard, direct-prediction transformer.
- **EXP-A1**: Train on 4-bit (1-2 digit) primes. Measure steps to grokking.
- **EXP-A2**: Train on 8-bit primes.
- **EXP-A3**: Train on 16-bit primes.
- **EXP-A4**: Train on 32-bit primes.
- **Key Metric**: Total optimization steps before validation loss drops to zero. We expect exponential scaling, leading to failure around A4.

## Experiment Track B: Regularization and Weight Decay
**Goal**: Verify Power et al.'s findings that weight decay induces grokking for modular arithmetic.
- **EXP-B1**: Run EXP-A2 with Weight Decay = 0.0. (Expectation: Overfitting, no grokking).
- **EXP-B2**: Run EXP-A2 with Weight Decay = 0.1. (Expectation: Grokking).
- **EXP-B3**: Run EXP-A2 with Weight Decay = 1.0. (Expectation: Underfitting).
- **Key Metric**: Gap between training loss and validation loss over $10^5$ steps.

## Experiment Track C: Positional Encodings
**Goal**: Determine which positional encoding scheme best supports arithmetic.
- **EXP-C1**: Absolute Positional Encodings (Standard).
- **EXP-C2**: RoPE (Rotary Positional Embeddings).
- **EXP-C3**: ALiBi (Attention with Linear Biases).
- **EXP-C4**: No Positional Encodings (Relying solely on causal masking and order).
- **Key Metric**: Length generalization. Train on $N$-bit numbers, evaluate on $(N+1)$-bit numbers.

## Experiment Track D: Scratchpad Emulation
**Goal**: Validate the Horner's method algorithmic emulation on Tier 2 difficulty.
- **EXP-D1**: Train a base-2 (binary) scratchpad on 16-bit primes.
- **EXP-D2**: Train a base-256 (byte-level) scratchpad on 16-bit primes.
- **EXP-D3**: Scale the winning approach from D1/D2 to 64-bit primes.
- **Key Metric**: Exact match accuracy on the final parsed output versus the number of intermediate tokens required.

## Documentation Standard
Every experiment must be logged in `experiments/archive/` with its exact config (learning rate, batch size, model dimensions) and a plot of the loss curves.
