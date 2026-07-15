# Future Milestones

This document tracks the high-level project trajectory for the SAIR Modular Arithmetic Challenge.

## Milestone 1: Repository Foundation (Current)
- [x] Analyze Mathematics Distillation reference repository.
- [x] Establish architecture, roadmaps, and knowledge base.
- [x] Define branding and visual identity.
- [ ] Initialize git, configure pre-commit hooks, and publish README.

## Milestone 2: Infrastructure & Data
- [ ] Implement `datasets/generators/prime_gen.py`.
- [ ] Implement custom tokenizers (Digit, Byte, Bit-level).
- [ ] Write PyTorch training loops with WandB/TensorBoard integration.
- [ ] Generate 10M sample dataset for Tier 1 (16-32 bit).

## Milestone 3: The Grokking Baseline (Tier 1)
- [ ] Train naive transformer on Tier 1.
- [ ] Observe and document the grokking phase transition.
- [ ] Export weights and write a baseline model card.

## Milestone 4: Scratchpad Algorithmic Learning (Tier 2/3)
- [ ] Implement Horner's method dataset generator.
- [ ] Train encoder-decoder or causal model on scratchpad data.
- [ ] Implement KV-caching inference wrapper.
- [ ] Validate on 128-bit primes.

## Milestone 5: The Router Architecture (Final Submission)
- [ ] Train a lightweight classifier to route inputs based on bit length.
- [ ] Combine Tier 1 and Tier 3 experts.
- [ ] Pass all local static analysis and sandbox simulated constraints.
- [ ] Package for official SAIR Submission.

## Milestone 6: Publication
- [ ] Push final weights to Hugging Face Hub.
- [ ] Publish an ArXiv-style summary of the training dynamics, curriculum learning successes, and grokking observations in the `docs/` folder.
