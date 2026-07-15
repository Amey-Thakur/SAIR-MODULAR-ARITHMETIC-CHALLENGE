# Competition Analysis

This document analyzes the strict constraints and evaluation pipeline of the SAIR Modular Arithmetic Challenge.

## 1. The Inference Contract

The core constraint of the competition is that the arithmetic must be *learned*.
- **Forbidden**: Calling `a % p` or `a * b` using native Python big-int math in the final inference step.
- **Forbidden**: Importing `sympy`, `gmpy2`, or using cryptographic libraries to execute Montgomery/Barrett reduction on the CPU.
- **Allowed**: Using native math *during data generation* and *training*.
- **Allowed**: The model using its parameters to output the result.

### The Sandbox Environment
The organizers evaluate submissions inside an isolated sandbox. 
- The submission must provide a `predict_digits()` or equivalent interface.
- Static analysis (AST parsing) and behavioral hooks trace the execution graph. If the compute graph delegates the heavy lifting to CPU integer math instead of tensor operations, the submission is disqualified.

## 2. Input/Output Encoding

Models must consume strings/arrays of characters and output strings/arrays.
- **Inputs**: `$a$, $b$, $p$` (likely formatted with separators).
- **Outputs**: `$r$` (the remainder).

If using a scratchpad, the `predict_digits()` function must run the autoregressive generation loop until the final answer is produced, then parse out the final answer to return to the judge.

## 3. Difficulty Tiers

The competition evaluates across multiple scales:
1. **Tier 1 (Easy)**: ~16 to 32 bits. Small enough that grokking and direct memorization/interpolation might work.
2. **Tier 2 (Medium)**: ~64 to 128 bits. The learnability wall. Requires structural biases (like relative positional encodings).
3. **Tier 3 (Hard)**: ~256+ bits. Requires algorithmic reasoning (Scratchpads / Chain of Thought).

## 4. Evaluation Metrics
- **Exact Match**: The primary metric. $(a \times b) \pmod{p}$ is chaotic; being off by 1 is entirely wrong.
- **Timing/Determinism**: Inference must complete within a reasonable time limit. The model should ideally use greedy decoding (temperature=0.0) to ensure deterministic outputs.

## 5. Security & Static Analysis

The organizers use Python `ast` and likely `sys.settrace` to monitor execution.
- Do not attempt to obfuscate `eval()` or `exec()`.
- Do not attempt to shell out to another process.
- Keep the `predict_digits` wrapper purely focused on tensor manipulations and token decoding.

## 6. Optimization Opportunities

- **Routing Models**: Because different tiers require vastly different compute, a valid strategy is a lightweight "Router" that checks the length of $P$, and directs the input to:
  - Model A (Fast, direct inference) for Tier 1.
  - Model B (Heavy, scratchpad autoregressive) for Tier 3.
- **Batch Inference**: Ensure the `predict_digits` function utilizes batching effectively to maximize throughput within the time limits.
