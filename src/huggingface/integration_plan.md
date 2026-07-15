# Hugging Face Integration Plan

While the primary focus is on the GitHub repository for research and development, a successful model will eventually be published to the Hugging Face Hub to ensure community reproducibility.

## 1. Repository Mapping
The local `huggingface/` directory serves as a staging ground. When ready for release, this directory will be initialized as a separate Git repository linked to a Hugging Face Space/Model hub.

## 2. Directory Structure for HF
```text
huggingface/
├── README.md               # The Model Card
├── model.safetensors       # The final weights (converted from PyTorch)
├── config.json             # Transformer configuration
├── tokenizer.json          # Custom vocabulary mapping
├── handler.py              # Custom inference code (if using Inference Endpoints)
└── manifest/               # SAIR-specific metadata
```

## 3. The Model Card
A world-class model card must be prepared, detailing:
- **Model Description**: The architecture (e.g., 50M parameter Byte-level ALiBi Transformer).
- **Intended Use**: Solving $(a \times b) \pmod{p}$.
- **Training Data**: Overview of the synthetic prime distribution used.
- **Evaluation Results**: Performance across bit-length tiers.
- **Limitations**: The maximum bit length the model can handle before accuracy degrades.

## 4. Checkpoint Conversion
We will use the `safetensors` library to save the weights, as it is faster to load and more secure than standard PyTorch pickle files. A script (`scripts/convert_to_safetensors.py`) will automate this translation from our local `experiments/` output.

## 5. Custom Inference Handlers
Because our model likely uses a custom Scratchpad decoding loop rather than standard NLP `generate()`, we will provide a `handler.py` that implements the logic required by Hugging Face Inference Endpoints. This ensures anyone can test the model via the HF API using a standard JSON payload.
