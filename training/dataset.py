import torch
from torch.utils.data import Dataset
import json
import os
import sys

# Add parent directory to path to import tokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenization.base10_tokenizer import Base10Tokenizer

class MathEquationDataset(Dataset):
    """
    A PyTorch Dataset for loading JSONL files containing equations or scratchpads.
    """
    def __init__(self, file_path: str, tokenizer: Base10Tokenizer, max_length: int = 512, is_scratchpad: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_scratchpad = is_scratchpad
        self.data = []
        
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                text = obj['scratchpad'] if self.is_scratchpad else obj['text']
                self.data.append(text)
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        
        # In a true seq2seq model we'd split encoder/decoder inputs.
        # For our decoder-only baseline, we just train on the whole sequence.
        # Input: tokens[:-1]
        # Target: tokens[1:]
        
        tokens = self.tokenizer.encode(text)
        
        # Truncate if necessary
        if len(tokens) > self.max_length + 1:
            tokens = tokens[:self.max_length + 1]
            
        # Pad sequence
        pad_len = (self.max_length + 1) - len(tokens)
        tokens.extend([self.tokenizer.pad_id] * pad_len)
        
        tensor = torch.tensor(tokens, dtype=torch.long)
        
        x = tensor[:-1]
        y = tensor[1:]
        
        return x, y
