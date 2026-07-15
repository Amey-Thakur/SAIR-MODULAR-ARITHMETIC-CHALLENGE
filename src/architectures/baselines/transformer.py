import torch
import torch.nn as nn
import math
from .config import TransformerConfig

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class BaselineTransformer(nn.Module):
    """
    A standard Decoder-only Transformer used to establish the
    'learnability wall' for modular arithmetic.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        
        # We use TransformerEncoder natively but apply a causal mask, effectively making it a decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True # Pre-LN is better for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a causal mask for the attention."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass for the decoder model.
        idx shape: (batch_size, seq_len)
        """
        b, t = idx.size()
        assert t <= self.config.max_seq_len, f"Cannot forward sequence of length {t}, max_seq_len is {self.config.max_seq_len}"

        # Embedding and Positional Encoding
        x = self.token_embedding(idx)
        x = self.positional_encoding(x)
        
        # Causal mask for autoregressive training
        causal_mask = self.generate_square_subsequent_mask(t).to(idx.device)
        
        # Transformer blocks
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.ln_f(x)
        
        # Logits
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Shifted targets (predict next token)
            # Flatten to calculate cross entropy
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), targets.view(-1))
            
        return logits, loss
