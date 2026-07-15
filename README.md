# SAIR Modular Arithmetic Challenge

![SAIR Modular Arithmetic Challenge Research Repository](./social_preview.png)

**Prepared by**: [Amey Thakur](https://github.com/Amey-Thakur)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗_Model_Hub-SAIR_Modular_Arithmetic-FFD21E.svg)](https://huggingface.co/ameythakur/SAIR-Modular-Arithmetic-Challenge)

Repository for the **SAIR Modular Arithmetic Challenge**. A laboratory investigating neural network induction of exact arithmetic logic, specifically modular multiplication ($(a \times b) \pmod{p}$), isolated from external symbolic modules or arbitrary precision math libraries.

> **Related Challenge**: [SAIR-MATHEMATICS-DISTILLATION-CHALLENGE](https://github.com/Amey-Thakur/SAIR-MATHEMATICS-DISTILLATION-CHALLENGE)

---

## Research Scope

Standard transformers fail at exact arithmetic due to position and scale permutation. We investigate:
1. **Grokking**: Validation loss phase transitions over delayed gradient trajectories.
2. **Abacus Embeddings**: Eliminating absolute spatial coordinates in favor of mathematical place-value significance to achieve perfect length generalization.
3. **Algorithmic Emulation**: Forcing multi-layer state machines via autoregressive trace decoding (Scratchpads/Horner's method).
4. **Dynamic Routing**: Mixture of Experts (MoE) dispatch logic bridging small and large modulo tiers.

---

## Directory Architecture

*   **[`docs/`](./docs/)**: Theoretical constraints, literature reviews, and publications.
*   **[`src/architectures/`](./src/architectures/)**: PyTorch module definitions (Abacus, RoPE Transformers, Routers).
*   **[`src/datasets/`](./src/datasets/)**: Curriculum and prime data generation pipelines.
*   **[`src/tokenization/`](./src/tokenization/)**: Custom significance-aware tokenization protocols.
*   **[`src/training/`](./src/training/)**: Execution loops, loss functions, and dataset wrappers.
*   **[`src/evaluation/`](./src/evaluation/)**: Exact-match inference metrics.
*   **[`src/sandbox/`](./src/sandbox/)**: AST validation simulating the strict SAIR sandbox contract.
*   **[`src/submission/`](./src/submission/)**: Isolated predict.py inference targets.
*   **[`src/huggingface/`](./src/huggingface/)**: Export and SafeTensors serialization scripts.

---

## Development Workflow

1. **Generation**: Execute `src/datasets/generators/` for specific bit-width curriculums.
2. **Execution**: Run `src/training/train.py` to compile checkpoints.
3. **Validation**: Test AST boundaries via `src/sandbox/ast_validator.py` and `simulate_judge.py`.
4. **Serialization**: Convert binaries via `src/huggingface/scripts/convert_to_safetensors.py`.

---

## Acknowledgements
- **SAIR**: Challenge infrastructure.
- **Power et al.**: Grokking foundations.
- **McLeish et al.**: Abacus Embeddings logic.
