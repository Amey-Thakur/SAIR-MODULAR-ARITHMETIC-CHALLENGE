# ==============================================================================
# File: config.py
# Description: Core module for SAIR Modular Arithmetic Challenge.
# Tech Stack: PyTorch 2.0+, Python 3.10+
# Author: Amey Thakur
# Profile: https://github.com/Amey-Thakur
# Repository: https://github.com/Amey-Thakur/SAIR-MODULAR-ARITHMETIC-CHALLENGE
# License: CC-BY-4.0
# Date: 2026-07-15
# ==============================================================================

from dataclasses import dataclass

@dataclass
class TransformerConfig:
    """Configuration for the Baseline Decoder-only Transformer."""
    vocab_size: int
    max_seq_len: int = 512
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    weight_decay: float = 0.1 # High weight decay for grokking
    pad_token_id: int = 0
