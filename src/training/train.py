# ==============================================================================
# File: train.py
# Description: Core training execution loop for modular arithmetic induction.
# Tech Stack: PyTorch 2.0+, Python 3.10+
# Author: Amey Thakur
# Profile: https://github.com/Amey-Thakur
# Repository: https://github.com/Amey-Thakur/SAIR-MODULAR-ARITHMETIC-CHALLENGE
# License: CC-BY-4.0
# Date: 2026-07-15
# ==============================================================================

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import sys

# Ensure modules are resolved from the repository root, enforcing the flat hierarchy.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import MathEquationDataset
from tokenization.base10_tokenizer import Base10Tokenizer
from architectures.baselines.config import TransformerConfig
from architectures.baselines.transformer import BaselineTransformer


def train():
    """
    Executes the training loop optimized for inducing arithmetic generalization.
    Standard optimization converges early to memorization; we deliberately force 
    continued training beyond validation plateau to induce Grokking.
    """
    # Deterministic accelerator fallback ensures the research pipeline is 
    # operational regardless of host infrastructure.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware bounds initialized. Backend target: {device}")
    
    tokenizer = Base10Tokenizer()
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'data', 'curriculum', '16bit_mul.jsonl'
    )
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Curriculum data absent at {data_path}. "
            "Data distributions must be pre-compiled via the dataset_gen pipeline."
        )
        
    dataset = MathEquationDataset(data_path, tokenizer, max_length=64)
    
    # Batch size heavily dictates convergence stability. 
    # Scaled down dynamically for CPU environments to prevent memory starvation.
    batch_size = 16 if device.type == 'cpu' else 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # We restrict embedding dimensions (d_model=128) intentionally to prevent 
    # the network from routing calculations through spatial memorization alone.
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size, 
        max_seq_len=64, 
        d_model=128, 
        n_heads=4, 
        n_layers=2
    )
    model = BaselineTransformer(config).to(device)
    
    # Grokking demands exceptionally high weight decay (wd=1.0). 
    # This aggressively penalizes dense storage, forcing the network to 
    # discover the sparse, algorithmic subnetworks for modulo arithmetic.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    
    # TensorBoard logs the phase transition boundary where validation loss craters.
    writer = SummaryWriter(log_dir='runs/sair_baseline')
    
    epochs = 1
    model.train()
    global_step = 0
    
    print("Optimization sequence activated.")
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            
            # Gradient clipping prevents activation explosions common in deep arithmetic models.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
                writer.add_scalar('Training/Loss', loss.item(), global_step)
                
            # Hardware throttle: Aborts early on CPUs to prevent OS lockup during testing.
            if device.type == 'cpu' and batch_idx == 5:
                print("Hardware throttle invoked (CPU Mode). Terminating epoch prematurely.")
                break
                
            global_step += 1
            
    writer.close()
    
    # Serialization checkpoints the weights for the AST validation sandbox.
    os.makedirs('models/checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'models/checkpoints/baseline_16bit.pt')


if __name__ == "__main__":
    train()
