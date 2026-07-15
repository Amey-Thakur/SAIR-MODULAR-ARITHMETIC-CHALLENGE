import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenization.base10_tokenizer import Base10Tokenizer
from architectures.baselines.config import TransformerConfig
from architectures.baselines.transformer import BaselineTransformer

def evaluate_exact_match(model: torch.nn.Module, tokenizer: Base10Tokenizer, prompt: str, max_new_tokens: int = 16, device: str = 'cpu') -> str:
    """
    Evaluates the model on a specific prompt by generating tokens autoregressively.
    """
    model.eval()
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(idx)
            # Take the logits for the last token
            next_token_logits = logits[0, -1, :]
            # Greedily pick the max probability token
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
            
            idx = torch.cat((idx, next_token), dim=1)
            
            if next_token.item() == tokenizer.pad_id or next_token.item() == tokenizer.char_to_id.get('=', -1):
                # Optionally stop if we generate a stop token, but for now we'll just run to max_new_tokens
                pass
                
    generated_text = tokenizer.decode(idx[0].tolist())
    return generated_text

if __name__ == "__main__":
    device = 'cpu'
    tokenizer = Base10Tokenizer()
    config = TransformerConfig(vocab_size=tokenizer.vocab_size, max_seq_len=64, d_model=128, n_heads=4, n_layers=2)
    model = BaselineTransformer(config).to(device)
    
    # In a real scenario, we would load the trained checkpoint here:
    # model.load_state_dict(torch.load('../models/checkpoints/baseline_16bit.pt'))
    
    prompt = "123*45="
    print(f"Prompt: {prompt}")
    output = evaluate_exact_match(model, tokenizer, prompt, device=device)
    print(f"Generated Output: {output}")
