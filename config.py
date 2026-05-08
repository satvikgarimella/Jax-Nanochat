"""
NanoChat Configuration
Supports both Shakespeare (small) and Alpaca (instruction) modes.
"""

from dataclasses import dataclass
from typing import Literal
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    """Model configuration (frozen for JAX JIT compatibility)."""
    # Architecture
    block_size: int = 512         # Context window
    vocab_size: int = 50257       # GPT-2 vocab (tiktoken)
    n_layer: int = 6              # Transformer layers
    n_head: int = 6               # Attention heads
    n_embd: int = 384             # Embedding dimension
    dropout: float = 0.0          # Dropout (unused in inference)
    dtype: str = "float32"        # "float32" or "bfloat16"


# Preset configurations
SHAKESPEARE_CONFIG = Config(
    block_size=256,
    vocab_size=65,        # Character-level
    n_layer=4,
    n_head=4,
    n_embd=128,
    dtype="float32",      # Small model, float32 is fine
)

ALPACA_CONFIG = Config(
    block_size=512,
    vocab_size=50257,     # GPT-2 BPE
    n_layer=6,
    n_head=6,
    n_embd=384,
    dtype="bfloat16",     # Use bfloat16 for speed
)

# Default config (for backward compatibility)
config = ALPACA_CONFIG
