# Submission Preparation Roadmap

The SAIR Modular Arithmetic Challenge requires submissions to execute within a strictly monitored sandbox. This roadmap details the pipeline for converting a trained PyTorch model into a compliant submission artifact.

## 1. The Inference Contract
The submission must consist of a `predict_digits()` interface that accepts strings (or arrays) and returns the string representation of the modulo result.

### Banned Operations
The judge performs AST analysis on the submission. If it detects:
- `%` operator on the operands.
- Calls to `sympy`, `gmpy2`, `math`, or other arbitrary precision libraries for the core logic.
- Execution of `exec()` or `eval()`.
The submission will be disqualified.

## 2. The Sandbox Wrapper
We must develop a lightweight, self-contained Python wrapper for the model.
- **No external dependencies**: Other than `torch` (and potentially `numpy` if allowed by the specific challenge manifest).
- **Weight Loading**: The script must load the `.pt` or `.safetensors` file locally without internet access (the sandbox is air-gapped).

## 3. The Decoding Loop (Scratchpad Parsing)
For advanced tiers, the model will output a scratchpad. The wrapper must:
1. Initialize the input tensor.
2. Run autoregressive generation loop (`torch.argmax` for greedy decoding).
3. Monitor for the `<EOS>` or `<ANS>` token.
4. Extract the final digits from the sequence.
5. Convert the digit tokens back into the required string format.
6. Return the string.

## 4. Performance Optimization
Because of time limits on the evaluation server:
- **Batching**: The wrapper must support batch inference if the judge provides multiple problems simultaneously.
- **KV-Caching**: Ensure past Key-Value states are cached during the autoregressive scratchpad generation to prevent $O(N^2)$ compute blowup.
- **Precision**: Export the model weights in `bfloat16` or `float16` to double inference speed and reduce memory limits.

## 5. Local Validation Pipeline
Before submitting, we must run:
1. `evaluation_pipeline/behavioral_checks/ast_validator.py` to ensure our code passes the rule checks.
2. `sandbox/simulate_judge.py` to run the model in an isolated, timed environment on a holdout test set.
