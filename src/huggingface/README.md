---
language:
- en
pipeline_tag: text-generation
tags:
- mathematics
- modular-arithmetic
- grokking
- scratchpad
license: cc-by-4.0
---

# SAIR Modular Arithmetic Challenge - Baseline Model

## Model Details
This is the baseline Grokking/Algorithmic model trained for the **SAIR Modular Arithmetic Challenge**. It is an autoregressive decoder-only Transformer designed to solve $(A \times B) \pmod{P}$ without utilizing any external mathematical libraries or hardcoded arithmetic operators.

- **Architecture:** Transformer with RoPE / Bit-Serial Algorithmic Decoder
- **Framework:** PyTorch
- **Tokenization:** Custom Character-level (`Base10Tokenizer`)

## Intended Use
This model is intended purely for research into algorithmic generalization, grokking, and mathematical reasoning in language models. 
Send a string like `123*456` and the model will generate the Scratchpad trace and the final answer.

## Limitations
As an algorithmic model, the context window bounds the maximum size of the integer that can be processed. If the multiplication trace exceeds the maximum sequence length, the model will fail to output `<EOS>`.

## Citation
If you use this model or the dataset generation logic, please cite the original SAIR Modular Arithmetic Challenge repository.
