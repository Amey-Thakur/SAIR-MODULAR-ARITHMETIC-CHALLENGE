# Recommendations for Achieving a Top Leaderboard Position

To win the SAIR Modular Arithmetic Challenge, brute force scaling is insufficient. The winner will be the team that best balances algorithmic emulation with the sandbox's time and compute limits.

## 1. Implement a Dynamic Router
Do not try to force a single model to solve 16-bit and 1024-bit problems simultaneously. 
- **Strategy**: Train a lightweight 1M parameter classifier that looks at the length of $P$ and routes the input.
- **Why**: Time is a critical factor on the leaderboard. If a 16-bit problem arrives, you should not waste time generating a 500-token scratchpad. Use a grokked, direct-prediction expert that answers in 1 token. Save the time budget for the Tier 3 problems.

## 2. Byte-Level Tokenization is the Sweet Spot
Standard BPE tokenizers will destroy the mathematical structure. Binary tokenization will exceed the time limit due to autoregressive generation length.
- **Strategy**: Treat the inputs as raw bytes (Base-256).
- **Why**: A 256-bit number is only 32 tokens. A Horner's method scratchpad in Base-256 is manageable within the typical 65,536 token limit. The network can easily memorize the $256 \times 256$ multiplication table in its MLP layers.

## 3. Emulate Hardware Multipliers
Instead of emulating human scratchpads (which carry left-to-right), emulate hardware Wallace trees or bit-serial multipliers.
- **Strategy**: Train the model to generate outputs starting from the Least Significant Byte (LSB).
- **Why**: Modular reduction and multiplication naturally flow from the LSB. Forcing a network to predict the Most Significant Byte first requires it to hold all intermediate carries in its hidden state, which severely caps scalability.

## 4. Master the Curriculum
The model will never learn if you start it on random 256-bit inputs.
- **Strategy**: Build a multi-stage data curriculum.
  1. Multiplication without Modulo.
  2. Modulo without Multiplication.
  3. $A \times B \pmod{P}$ for $P < 100$.
  4. Gradually scale bit length.
- **Why**: Transformers learn compositional rules. If the components (multiplication and wrapping) are not learned independently first, the combined task represents an impossible loss landscape.

## 5. Aggressive Weight Decay
- **Strategy**: Use significantly higher weight decay than standard NLP tasks (e.g., `0.1` to `1.0`).
- **Why**: Generalization in algorithmic tasks requires the network to find low-rank, sparse circuits. High weight decay forces the network to prune memorization circuits and rely on generalizing algorithmic circuits.
