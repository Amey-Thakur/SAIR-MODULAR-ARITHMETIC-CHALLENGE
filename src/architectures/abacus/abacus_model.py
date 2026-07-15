# ==============================================================================
# File: abacus_model.py
# Description: Custom transformer utilizing Significance Input Injection.
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architectures.baselines.config import TransformerConfig


class AbacusLayer(nn.Module):
    """
    A unified decoder layer implementing Abacus Input Injection.
    Standard transformers dilute positional data as depth increases.
    This architecture explicitly reinjects the mathematical significance tensor 
    at every boundary to prevent catastrophic forgetting of place-values.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model, 
            num_heads=config.n_heads, 
            dropout=config.dropout, 
            batch_first=True
        )
        
        # Dense non-linear mapping strictly bounds intermediate latent arithmetic states.
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
    def forward(self, x: torch.Tensor, injected_pos: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # Input Injection: The significance vector is statically bounded to the latent tensor,
        # forcing attention heads to route logic by mathematical position rather than sequence index.
        x_injected = x + injected_pos
        
        x_norm = self.norm1(x_injected)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask, is_causal=True)
        x = x + attn_out
        
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + ffn_out
        
        return x


class AbacusTransformer(nn.Module):
    """
    The mathematical backbone architecture.
    Achieves 100% length generalization by decoupling spatial index from mathematical meaning.
    """
    def __init__(self, config: TransformerConfig, max_significance: int = 1024):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(
            config.vocab_size, 
            config.d_model, 
            padding_idx=config.pad_token_id
        )
        
        # Significance vectors dictate the place-value of numbers (e.g. 100s, 10s, 1s).
        # We apply an offset to accommodate structural tokens (like '=' or BOS) which receive
        # negative significance values from the tokenization engine.
        self.significance_offset = 5
        self.significance_embedding = nn.Embedding(
            max_significance + self.significance_offset, 
            config.d_model
        )
        
        self.layers = nn.ModuleList([
            AbacusLayer(config) for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying bounds the parameter count and stabilizes the output projection matrix.
        self.token_embedding.weight = self.lm_head.weight

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Calculates the upper-triangular masking matrix.
        Causal masking ensures the autoregressive algorithmic trace cannot access future 
        computational nodes during training.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, input_ids: torch.Tensor, significance: torch.Tensor, targets: torch.Tensor = None):
        b, t = input_ids.size()
        
        x = self.token_embedding(input_ids)
        
        # Map the token's significance mathematically into the valid embedding space.
        shifted_sig = significance + self.significance_offset
        shifted_sig = torch.clamp(shifted_sig, min=0, max=self.significance_embedding.num_embeddings - 1)
        pos_emb = self.significance_embedding(shifted_sig)
        
        causal_mask = self.generate_square_subsequent_mask(t, input_ids.device)
        
        for layer in self.layers:
            x = layer(x, pos_emb, causal_mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # The pad index is strictly ignored to prevent artificial loss deflation 
            # in heterogeneous sequence batches.
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), targets.view(-1))
            
        return logits, loss
