"""
NanoChat Data Loader
Supports both Shakespeare (character-level) and Alpaca (instruction fine-tuning with tiktoken).
"""

import os
import json
import pickle
import urllib.request
import numpy as np
import jax.numpy as jnp
from typing import List, Tuple, Optional

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================================
# Prompt Template for Instruction Fine-Tuning
# ============================================================================

SYSTEM_PROMPT = "You are NanoChat, a friendly and helpful AI assistant."

def format_alpaca_prompt(instruction: str, input_text: str = "", response: str = "") -> str:
    """
    Format instruction data using Alpaca template.
    
    Template:
    ### System: {system_prompt}
    ### Instruction: {instruction}
    ### Input: {input} (optional)
    ### Response: {response}
    """
    prompt = f"### System: {SYSTEM_PROMPT}\n"
    prompt += f"### Instruction: {instruction}\n"
    
    if input_text:
        prompt += f"### Input: {input_text}\n"
    
    prompt += f"### Response: {response}"
    return prompt


def format_chat_prompt(conversation: List[dict]) -> str:
    """
    Format multi-turn conversation for inference.
    
    conversation: List of {"role": "user"|"assistant", "content": str}
    """
    prompt = f"### System: {SYSTEM_PROMPT}\n"
    
    for turn in conversation:
        if turn["role"] == "user":
            prompt += f"### Instruction: {turn['content']}\n"
        else:
            prompt += f"### Response: {turn['content']}\n"
    
    # Add response prefix for generation
    if conversation[-1]["role"] == "user":
        prompt += "### Response:"
    
    return prompt


# ============================================================================
# Tiktoken Tokenizer (BPE)
# ============================================================================

class TiktokenWrapper:
    """Wrapper around tiktoken for NanoChat."""
    
    def __init__(self, encoding_name: str = "gpt2"):
        """
        Initialize tiktoken encoder.
        
        Args:
            encoding_name: "gpt2" (50257 vocab) or "cl100k_base" (100277 vocab)
        """
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding(encoding_name)
            self.vocab_size = self.enc.n_vocab
            self.encoding_name = encoding_name
        except ImportError:
            raise ImportError("Please install tiktoken: pip install tiktoken")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        return self.enc.encode(text, allowed_special="all")
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token ids to text."""
        return self.enc.decode(tokens)
    
    def save(self, path: str):
        """Save tokenizer config."""
        with open(path, 'wb') as f:
            pickle.dump({
                'type': 'tiktoken',
                'encoding_name': self.encoding_name,
                'vocab_size': self.vocab_size
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'TiktokenWrapper':
        """Load tokenizer from config."""
        with open(path, 'rb') as f:
            meta = pickle.load(f)
        return cls(meta['encoding_name'])


# ============================================================================
# Character-Level Tokenizer (for Shakespeare)
# ============================================================================

class CharTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, chars: Optional[List[str]] = None):
        if chars:
            self.chars = chars
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}
            self.vocab_size = len(chars)
    
    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]
    
    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.itos[i] for i in tokens if i in self.itos])
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'type': 'char',
                'chars': self.chars,
                'vocab_size': self.vocab_size
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'CharTokenizer':
        with open(path, 'rb') as f:
            meta = pickle.load(f)
        tokenizer = cls()
        tokenizer.chars = meta['chars']
        tokenizer.stoi = {ch: i for i, ch in enumerate(meta['chars'])}
        tokenizer.itos = {i: ch for i, ch in enumerate(meta['chars'])}
        tokenizer.vocab_size = meta['vocab_size']
        return tokenizer
    
    @classmethod
    def from_text(cls, text: str) -> 'CharTokenizer':
        chars = sorted(list(set(text)))
        return cls(chars)


# ============================================================================
# Dataset Loaders
# ============================================================================

def download_shakespeare() -> str:
    """Download TinyShakespeare dataset."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    candidate_paths = [
        os.path.join(DATA_DIR, "shakespeare.txt"),
        os.path.join(DATA_DIR, "input.txt"),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()

    path = candidate_paths[0]
    print("Downloading TinyShakespeare...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:
        raise RuntimeError(
            "TinyShakespeare dataset not found locally and download failed. "
            f"Expected one of: {candidate_paths}"
        ) from exc

    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def download_alpaca() -> List[dict]:
    """Download Alpaca-Cleaned dataset."""
    url = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"
    path = os.path.join(DATA_DIR, "alpaca_cleaned.json")
    
    if not os.path.exists(path):
        print("Downloading Alpaca-Cleaned dataset...")
        urllib.request.urlretrieve(url, path)
        print("Download complete!")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_shakespeare(val_split: float = 0.1) -> Tuple:
    """
    Prepare Shakespeare dataset with character-level tokenizer.
    Returns: (train_data, val_data, tokenizer)
    """
    text = download_shakespeare()
    tokenizer = CharTokenizer.from_text(text)
    tokenizer.save(os.path.join(DATA_DIR, "tokenizer.pkl"))
    
    data = np.array(tokenizer.encode(text), dtype=np.int32)
    split_idx = int(len(data) * (1 - val_split))
    
    print(f"Shakespeare: {len(data):,} chars, vocab={tokenizer.vocab_size}")
    return data[:split_idx], data[split_idx:], tokenizer


def prepare_alpaca(val_split: float = 0.05, max_length: int = 512) -> Tuple:
    """
    Prepare Alpaca dataset with tiktoken BPE tokenizer.
    Returns: (train_data, val_data, tokenizer)
    """
    data = download_alpaca()
    tokenizer = TiktokenWrapper("gpt2")
    tokenizer.save(os.path.join(DATA_DIR, "tokenizer.pkl"))
    
    # Format all examples
    print(f"Processing {len(data)} Alpaca examples...")
    all_tokens = []
    
    for example in data:
        prompt = format_alpaca_prompt(
            instruction=example.get("instruction", ""),
            input_text=example.get("input", ""),
            response=example.get("output", "")
        )
        tokens = tokenizer.encode(prompt + "\n\n")  # Add separator
        all_tokens.extend(tokens)
    
    all_tokens = np.array(all_tokens, dtype=np.int32)
    split_idx = int(len(all_tokens) * (1 - val_split))
    
    print(f"Alpaca: {len(all_tokens):,} tokens, vocab={tokenizer.vocab_size}")
    return all_tokens[:split_idx], all_tokens[split_idx:], tokenizer


# ============================================================================
# Legacy Compatibility Functions
# ============================================================================

def load_tokenizer():
    """Load saved tokenizer (legacy compatibility)."""
    path = os.path.join(DATA_DIR, "tokenizer.pkl")
    
    if not os.path.exists(path):
        # Fallback to meta.pkl for old Shakespeare models
        meta_path = os.path.join(DATA_DIR, "meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            return meta['stoi'], meta['itos'], meta['vocab_size']
        raise FileNotFoundError("No tokenizer found. Run prepare_data() first.")
    
    with open(path, 'rb') as f:
        meta = pickle.load(f)
    
    if meta['type'] == 'tiktoken':
        tokenizer = TiktokenWrapper.load(path)
        return tokenizer.encode, tokenizer.decode, tokenizer.vocab_size
    else:
        tokenizer = CharTokenizer.load(path)
        return tokenizer.stoi, tokenizer.itos, tokenizer.vocab_size


def decode(tokens, itos_or_func):
    """Decode tokens (supports both dict and function)."""
    if callable(itos_or_func):
        return itos_or_func(list(tokens))
    return ''.join([itos_or_func[i] for i in tokens])


def encode(text, stoi_or_func):
    """Encode text (supports both dict and function)."""
    if callable(stoi_or_func):
        return stoi_or_func(text)
    return [stoi_or_func[ch] for ch in text if ch in stoi_or_func]


# Legacy function for old models
def prepare_data(val_split=0.1):
    """Legacy function - prepares Shakespeare data."""
    train, val, tokenizer = prepare_shakespeare(val_split)
    return train, val, tokenizer.vocab_size, tokenizer.stoi, tokenizer.itos


# ============================================================================
# DataLoader
# ============================================================================

class DataLoader:
    """Batch generator for training."""
    
    def __init__(self, data: np.ndarray, block_size: int, batch_size: int, seed: int = 42):
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
    
    def __len__(self):
        return len(self.data) // (self.block_size * self.batch_size)
    
    def get_batch(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        max_start = len(self.data) - self.block_size - 1
        starts = self.rng.integers(0, max_start, size=(self.batch_size,))
        
        tokens = np.stack([self.data[i:i + self.block_size] for i in starts])
        targets = np.stack([self.data[i + 1:i + 1 + self.block_size] for i in starts])
        
        return jnp.array(tokens), jnp.array(targets)


def get_batch_for_eval(data, block_size, batch_size, rng):
    """Get a batch for evaluation."""
    max_start = len(data) - block_size - 1
    starts = rng.integers(0, max_start, size=(batch_size,))
    
    tokens = np.stack([data[i:i + block_size] for i in starts])
    targets = np.stack([data[i + 1:i + 1 + block_size] for i in starts])
    
    return jnp.array(tokens), jnp.array(targets)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Testing Shakespeare loader...")
    train, val, tokenizer = prepare_shakespeare()
    print(f"Sample: {tokenizer.decode(train[:100].tolist())[:80]}...")
    
    print("\nTesting Alpaca loader...")
    try:
        train, val, tokenizer = prepare_alpaca()
        print(f"Sample: {tokenizer.decode(train[:100].tolist())[:80]}...")
    except ImportError as e:
        print(f"Skipping Alpaca test: {e}")
