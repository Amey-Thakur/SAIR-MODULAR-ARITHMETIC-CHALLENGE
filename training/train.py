import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset import MathEquationDataset
from tokenization.base10_tokenizer import Base10Tokenizer
from architectures.baselines.config import TransformerConfig
from architectures.baselines.transformer import BaselineTransformer

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Setup Data
    tokenizer = Base10Tokenizer()
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'curriculum', '16bit_mul.jsonl')
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Please run datasets generators first.")
        return
        
    dataset = MathEquationDataset(data_path, tokenizer, max_length=64)
    # Use a small batch size for CPU testing, scale up for GPU
    batch_size = 16 if device.type == 'cpu' else 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Setup Model
    config = TransformerConfig(vocab_size=tokenizer.vocab_size, max_seq_len=64, d_model=128, n_heads=4, n_layers=2)
    model = BaselineTransformer(config).to(device)
    
    # 3. Setup Optimizer & Logger
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0) # High WD for grokking
    writer = SummaryWriter(log_dir='runs/sair_baseline')
    
    # 4. Training Loop (Just 1 epoch for verification)
    epochs = 1
    model.train()
    
    print("Starting training loop...")
    global_step = 0
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
                writer.add_scalar('Training/Loss', loss.item(), global_step)
                
            # For CPU verification, we'll break early so we don't hang the system
            if device.type == 'cpu' and batch_idx == 5:
                print("CPU verification run complete. Stopping early.")
                break
                
            global_step += 1
                
        print(f"Epoch {epoch} Average Loss: {total_loss / min(len(dataloader), 6 if device.type == 'cpu' else len(dataloader)):.4f}")
        
    writer.close()
    
    # Save checkpoint
    os.makedirs('models/checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'models/checkpoints/baseline_16bit.pt')
    print("Saved checkpoint to models/checkpoints/baseline_16bit.pt")

if __name__ == "__main__":
    train()
