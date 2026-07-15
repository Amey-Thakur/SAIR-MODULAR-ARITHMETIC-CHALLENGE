# Literature Review: Neural Arithmetic and Grokking

This document summarizes key academic papers and community research that form the foundation for our approach to the SAIR Modular Arithmetic Challenge.

## 1. Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets
**Authors**: Power et al. (OpenAI, 2022)
- **Problem Addressed**: How neural networks learn algorithmic tasks (including modular arithmetic).
- **Methodology**: Training small transformers on binary operations like $(a \diamond b)$. 
- **Key Findings**: Models often heavily overfit the training data initially (achieving ~0% validation accuracy) but suddenly generalize ("grok") after prolonged training (e.g., $10^5$ optimization steps). 
- **Mathematical Insights**: Weight decay is critical to encourage grokking by forcing the network to discover low-complexity representations (e.g., Fourier bases).
- **Relevance**: Direct proof that transformers *can* learn modular arithmetic given enough training steps and appropriate regularization, without hard-coded rules.

## 2. A Mechanistic Interpretability Analysis of Grokking
**Authors**: Nanda et al. (2023)
- **Problem Addressed**: Understanding *how* models grok modular addition.
- **Methodology**: Reverse-engineering the weights of a small 1-layer transformer trained on $a + b \pmod{p}$.
- **Key Findings**: The network learns a Discrete Fourier Transform (DFT). It projects integers onto a circle, performs rotation (addition), and projects back.
- **Relevance**: Suggests that for modular multiplication, the network might attempt to learn discrete logarithms (mapping multiplication to addition in the exponent) or complex Fourier-like bases. This informs our architectural bias: maybe we should use complex-valued activations or specialized positional encodings.

## 3. Teaching Algorithmic Reasoning via In-context Learning
**Authors**: Various (e.g., Zhou et al., 2022, "Teaching Algorithmic Reasoning")
- **Problem Addressed**: LLMs struggle with large arithmetic operations because they process left-to-right without intermediate memory.
- **Methodology**: Chain-of-Thought (CoT) and Scratchpad emulation.
- **Key Findings**: Emulating a step-by-step scratchpad (e.g., Horner's method, digit-by-digit multiplication with carry) vastly improves accuracy over direct answer generation.
- **Relevance**: To multiply very large numbers modulo $p$, predicting the final digits directly is likely impossible (hitting a "learnability wall"). The model must be trained to output intermediate steps.

## 4. Length Generalization in Transformers
**Authors**: Anil et al. (2022)
- **Problem Addressed**: Transformers trained on $N$-digit numbers fail catastrophically on $(N+1)$-digit numbers.
- **Methodology**: Analyzing position embeddings and tokenization.
- **Key Findings**: Relative positional encodings (like RoPE or ALiBi) and strict right-aligned character/digit-level tokenization improve length generalization.
- **Relevance**: Our tokenization strategy must explicitly handle alignment. Standard subword tokenizers (like BPE) group digits arbitrarily (e.g., `1234` -> `12`, `34`), destroying the mathematical structure.

## 5. Routing and Specialist Models
**Authors**: Various mixture-of-experts literature.
- **Relevance to Competition**: Given the difficulty tiers in the challenge based on bit lengths, using a router mechanism to classify the bit length of the inputs and direct them to specialized sub-networks (or dynamically allocated compute budgets) is a highly viable strategy.

## Summary of Lessons
1. **Tokenize explicitly**: 1 digit = 1 token (or byte-level). No BPE.
2. **Regularize heavily**: Weight decay is necessary for grokking.
3. **Use Scratchpads**: For large integers, the model must output intermediate steps (e.g. shift and add).
4. **Beware length generalization**: Train on distributions that force the model to handle variable lengths cleanly.
