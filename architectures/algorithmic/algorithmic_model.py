import torch
import torch.nn as nn
import os
import sys

# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines.config import TransformerConfig
from baselines.transformer import PositionalEncoding

class AlgorithmicDecoder(nn.Module):
    """
    Phase 3 Architecture: The Bit-Serial Algorithmic Model.
    This model consumes Binary tokens (0, 1) and generates a Scratchpad.
    It simulates an algorithmic state machine.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Vocabulary is much smaller (Binary + Special Tokens)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        
        # Deep narrow network is better for sequential algorithmic reasoning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers * 2) # Deeper
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        b, t = idx.size()
        
        x = self.token_embedding(idx)
        x = self.positional_encoding(x)
        
        causal_mask = self.generate_square_subsequent_mask(t).to(idx.device)
        
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), targets.view(-1))
            
        return logits, loss
