# SAIR Modular Arithmetic Challenge

![SAIR Modular Arithmetic Challenge Research Repository](./social_preview.png)

**Prepared by**: [Amey Thakur](https://github.com/Amey-Thakur)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

Welcome to the long-term research repository for the **SAIR Modular Arithmetic Challenge**. This repository acts as a comprehensive laboratory for investigating how neural networks can organically learn mathematically rigid operations like modular multiplication ($(a \times b) \pmod{p}$) without relying on hard-coded symbolic logic.

> **Related Challenge**: See my other formal reasoning solution repository: [SAIR-MATHEMATICS-DISTILLATION-CHALLENGE](https://github.com/Amey-Thakur/SAIR-MATHEMATICS-DISTILLATION-CHALLENGE).

---

## Repository Purpose & Research Goals

Our core objective is to bypass the "learnability wall" where standard transformers fail at arithmetic logic. We are specifically investigating:
1. **Grokking**: Phase transitions in validation loss during prolonged training.
2. **Abacus Embeddings**: Implementing state-of-the-art significance embedding injection for 100% theoretical length generalization.
3. **Algorithmic Emulation**: Teaching models to act as state machines by generating autoregressive "scratchpads" (e.g., Horner's method).
4. **Dynamic Routing**: Building expert models that can adaptively solve 16-bit primes vs 1024-bit primes based on computational budget.

---

## Directory Architecture

The repository is organized directly into `src` for operational code and `docs` for research.

*   **[`docs/`](./docs/)**: Core architectural decisions, competition constraints, literature reviews, and published summaries.
*   **[`src/architectures/`](./src/architectures/)**: Neural network definitions (e.g., standard transformers, the Abacus Transformer, and Routers).
*   **[`src/datasets/`](./src/datasets/)**: Synthetic data generators for large prime arithmetic.
*   **[`src/tokenization/`](./src/tokenization/)**: Digit, byte, and significance-level tokenization for Abacus mappings.
*   **[`src/training/`](./src/training/)**: PyTorch execution loops, TensorBoard logging, and Dataset streaming.
*   **[`src/evaluation/`](./src/evaluation/)**: Autoregressive exact-match decoding metrics.
*   **[`src/sandbox/`](./src/sandbox/)**: Local mock of the strict SAIR judge (AST behavioral checks blocking `%` operators).
*   **[`src/submission/`](./src/submission/)**: Artifacts formatted explicitly for the competition constraints.
*   **[`src/huggingface/`](./src/huggingface/)**: Scripts for deployment and model cards for final community release.

---

## Model Development Workflow

1. **Generate Data**: Use `src/datasets/generators/` to create the appropriate curriculum.
2. **Train & Ablate**: Run `src/training/train.py` to observe phase transitions.
3. **Evaluate Locally**: Pass the model through `src/sandbox/simulate_judge.py` to ensure it adheres to the strict inference contract.
4. **Publish**: Prepare weights via `src/submission/` and document on the Hugging Face hub.

---

## Acknowledgements
- **SAIR**: For hosting the Modular Arithmetic Challenge.
- **Power et al. (OpenAI)**: For the foundational work on Grokking.
- **McLeish et al.**: For the breakthrough Abacus Embeddings architecture.

---
*Built with rigor, designed for discovery.*
