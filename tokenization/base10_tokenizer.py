import torch
from typing import List, Union

class Base10Tokenizer:
    """
    A custom character-level tokenizer for modular arithmetic.
    Designed to process strings like 'A*B%P=R' mapping each character
    to a unique integer token ID.
    """
    def __init__(self):
        # Define the exact vocabulary for the Modular Arithmetic Challenge
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        self.chars = [
            self.pad_token, self.bos_token, self.eos_token,
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "*", "%", "=", "A", "B", "P", "R", " ", ",", "M", "O", "D"
        ]
        
        self.vocab_size = len(self.chars)
        
        # Create mappings
        self.char_to_id = {c: i for i, c in enumerate(self.chars)}
        self.id_to_char = {i: c for i, c in enumerate(self.chars)}
        
        self.pad_id = self.char_to_id[self.pad_token]
        self.bos_id = self.char_to_id[self.bos_token]
        self.eos_id = self.char_to_id[self.eos_token]
        
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Convert a string to a list of token IDs."""
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
            
        for char in text:
            if char not in self.char_to_id:
                raise ValueError(f"Character '{char}' not in vocabulary.")
            tokens.append(self.char_to_id[char])
            
        if add_eos:
            tokens.append(self.eos_id)
            
        return tokens

    def encode_batch(self, texts: List[str], max_length: int = None) -> torch.Tensor:
        """Encode a batch of strings and pad them to the same length."""
        encoded = [self.encode(text) for text in texts]
        
        if max_length is None:
            max_length = max(len(seq) for seq in encoded)
            
        padded = []
        for seq in encoded:
            if len(seq) > max_length:
                padded.append(seq[:max_length])
            else:
                pad_len = max_length - len(seq)
                padded.append(seq + [self.pad_id] * pad_len)
                
        return torch.tensor(padded, dtype=torch.long)

    def decode(self, ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Convert a list of token IDs back to a string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            
        chars = []
        for token_id in ids:
            if token_id in [self.pad_id, self.bos_id, self.eos_id] and skip_special_tokens:
                continue
            chars.append(self.id_to_char.get(token_id, ""))
            
        return "".join(chars)

    def calculate_significance(self, text: str) -> List[int]:
        """
        Calculates the place-value significance of each character in a math string.
        Example: '123*45' ->
        1 (100s -> 2), 2 (10s -> 1), 3 (1s -> 0)
        * (-1)
        4 (10s -> 1), 5 (1s -> 0)
        """
        significance = []
        
        # Split string by non-digit characters to isolate numbers
        import re
        # Find all numbers and their spans
        spans = [(m.start(), m.end()) for m in re.finditer(r'\d+', text)]
        
        # Default all to -1 (special characters)
        significance = [-1] * len(text)
        
        for start, end in spans:
            num_len = end - start
            for i in range(num_len):
                # Significance is length - 1 - index
                # E.g. length 3, index 0 -> significance 2
                significance[start + i] = num_len - 1 - i
                
        return significance

    def encode_with_significance(self, text: str, add_bos: bool = True, add_eos: bool = True) -> dict:
        """Encodes text and returns both token IDs and significance indices."""
        tokens = []
        sigs = []
        
        char_sigs = self.calculate_significance(text)
        
        if add_bos:
            tokens.append(self.bos_id)
            sigs.append(-2) # Special significance for BOS
            
        for idx, char in enumerate(text):
            if char not in self.char_to_id:
                raise ValueError(f"Character '{char}' not in vocabulary.")
            tokens.append(self.char_to_id[char])
            sigs.append(char_sigs[idx])
            
        if add_eos:
            tokens.append(self.eos_id)
            sigs.append(-3) # Special significance for EOS
            
        return {"input_ids": tokens, "significance": sigs}
