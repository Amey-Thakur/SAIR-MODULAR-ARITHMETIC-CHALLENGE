# Model Research Roadmap

This roadmap outlines the progression of model architectures we will investigate. We progress from naive baselines to complex algorithmic emulators.

## Phase 1: The Naive Baseline
**Objective**: Establish the "learnability wall" for a standard architecture.
- **Architecture**: Standard Decoder-only Transformer (e.g., GPT-style, ~10M-50M parameters).
- **Tokenization**: Base-10 characters (0-9).
- **Task**: Direct prediction. `Input: A * B % P = Output: R`.
- **Expected Result**: Will fail catastrophically beyond 5-6 digits.

## Phase 2: Encoding and Alignment
**Objective**: Improve length generalization and positional understanding.
- **Architecture**: Transformer with ALiBi or RoPE (Relative Positional Encodings).
- **Tokenization**: Right-aligned padding. 
- **Task**: Direct prediction, but with reversed outputs. Generating the least significant digit first is algorithmically easier because carries propagate right-to-left.
- **Expected Result**: Slight improvement, pushing the boundary to ~10 digits.

## Phase 3: The Bit-Serial Algorithmic Model
**Objective**: Force the model to learn the $O(\log B)$ multiplication algorithm.
- **Architecture**: Encoder-Decoder or Decoder-only.
- **Tokenization**: Binary (0, 1).
- **Task**: Scratchpad generation. The model must output intermediate steps:
  - Step 1: $A \times b_0 \pmod{P}$
  - Step 2: $2A \pmod{P}$
  - Step 3: Accumulate...
- **Expected Result**: Significant breakthrough. Slower inference, but capable of handling Tier 2 (64-128 bit) problems.

## Phase 4: The Mixed-Radix / Hybrid Router
**Objective**: Maximize score across all tiers within time limits.
- **Architecture**: A gating network (Router) + Tier-specific experts.
- **Expert 1**: Fast, direct-prediction network for small bits (relies on grokking).
- **Expert 2**: Base-256 (Byte-level) scratchpad model. Doing binary scratchpads for 1024 bits takes thousands of tokens. Operating in base-256 reduces the sequence length by $8\times$, balancing token cost vs. algorithmic complexity.
- **Expected Result**: The final submission candidate.

## Ongoing: Curriculum Learning
For all phases, we will not train on random data distributions immediately.
1. Train on $A \times B$ without modulo.
2. Train on $A \pmod{P}$ without multiplication.
3. Train on $A \times B \pmod{P}$ where $P$ is small.
4. Gradually increase the bit length of $P$.
