from dataclasses import dataclass

@dataclass
class Config:
    block_size: int = 256     # Context window (max tokens)
    vocab_size: int = 50257    # GPT-2 vocabulary size
    n_layer: int = 4           # Number of transformer blocks
    n_head: int = 4            # Number of attention heads
    n_embd: int = 128          # Embedding dimension
    dropout: float = 0.0       # Regularization
    learning_rate: float = 6e-4
    