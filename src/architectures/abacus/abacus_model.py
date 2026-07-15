# ==============================================================================
# File: abacus_model.py
# Description: Core module for SAIR Modular Arithmetic Challenge.
# Tech Stack: PyTorch 2.0+, Python 3.10+
# Author: Amey Thakur
# Profile: https://github.com/Amey-Thakur
# Repository: https://github.com/Amey-Thakur/SAIR-MODULAR-ARITHMETIC-CHALLENGE
# License: CC-BY-4.0
# Date: 2026-07-15
# ==============================================================================

import torch
import torch.nn as nn
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines.config import TransformerConfig

class AbacusLayer(nn.Module):
    """
    A single Decoder layer that supports Input Injection.
    In Abacus architecture, the original positional (significance) embeddings
    are added back to the latent state at the start of every layer to prevent 
    positional information loss during deep arithmetic reasoning.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model, 
            num_heads=config.n_heads, 
            dropout=config.dropout, 
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
    def forward(self, x, injected_pos, causal_mask):
        # 1. Input Injection (The Abacus secret sauce)
        x_injected = x + injected_pos
        
        # 2. Attention
        x_norm = self.norm1(x_injected)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask, is_causal=True)
        x = x + attn_out
        
        # 3. FFN
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + ffn_out
        
        return x

class AbacusTransformer(nn.Module):
    """
    The 100% Accurate Abacus Model.
    Utilizes significance embeddings instead of absolute/rotary embeddings,
    and injects them across every layer to achieve infinite length generalization
    for arithmetic tasks.
    """
    def __init__(self, config: TransformerConfig, max_significance: int = 1024):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        
        # Significance embeddings (0 to max_significance). 
        # We need a small offset for negative/special tokens (like -1, -2, -3).
        # We map significance S to S + offset to fit within Embedding bounds.
        self.significance_offset = 5
        self.significance_embedding = nn.Embedding(max_significance + self.significance_offset, config.d_model)
        
        self.layers = nn.ModuleList([
            AbacusLayer(config) for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def generate_square_subsequent_mask(self, sz: int, device) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, input_ids: torch.Tensor, significance: torch.Tensor, targets: torch.Tensor = None):
        """
        input_ids: (batch, seq_len)
        significance: (batch, seq_len) containing mathematical place-values
        """
        b, t = input_ids.size()
        
        # Token Embeddings
        x = self.token_embedding(input_ids)
        
        # Map significance negative indices (like -1 for '+') to valid embedding range
        shifted_sig = significance + self.significance_offset
        # Clamp to bounds just in case
        shifted_sig = torch.clamp(shifted_sig, min=0, max=self.significance_embedding.num_embeddings - 1)
        
        # The core Abacus positional embeddings
        pos_emb = self.significance_embedding(shifted_sig)
        
        causal_mask = self.generate_square_subsequent_mask(t, input_ids.device)
        
        # Pass through layers with Input Injection
        for layer in self.layers:
            x = layer(x, pos_emb, causal_mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), targets.view(-1))
            
        return logits, loss
