# Future Milestones

This document tracks the high-level project trajectory for the SAIR Modular Arithmetic Challenge.

## Milestone 1: Repository Foundation (Current)
- [x] Analyze Mathematics Distillation reference repository.
- [x] Establish architecture, roadmaps, and knowledge base.
- [x] Define branding and visual identity.
- [x] Initialize git, configure pre-commit hooks, and publish README.

## Milestone 2: Data & Tokenization (Weeks 3-4)
- [x] Implement `datasets/generators/prime_gen.py`.
- [x] Implement custom tokenizers (Digit, Byte, Bit-level).
- [x] Write PyTorch training loops with WandB/TensorBoard integration.
- [x] Generate 10M sample dataset for Tier 1 (16-32 bit).

## Milestone 3: Base Model & Grokking (Weeks 5-6)
- [x] Train naive transformer on Tier 1.
- [x] Observe and document the grokking phase transition.
- [x] Export weights and write a baseline model card.

## Milestone 4: Algorithmic Generalization (Weeks 7-9)
- [x] Implement Horner's method dataset generator.
- [x] Train encoder-decoder or causal model on scratchpad data.
- [x] Implement KV-caching inference wrapper.
- [x] Validate on 128-bit primes.

## Milestone 5: The Mixed-Radix Router (Week 10)
- [x] Train a lightweight classifier to route inputs based on bit length.
- [x] Combine Tier 1 and Tier 3 experts.
- [x] Pass all local static analysis and sandbox simulated constraints.
- [x] Package for official SAIR Submission.

## Milestone 6: Release (Week 11)
- [x] Push final weights to Hugging Face Hub.
- [x] Publish an ArXiv-style summary of the training dynamics, curriculum learning successes, and grokking observations in the `docs/` folder.
