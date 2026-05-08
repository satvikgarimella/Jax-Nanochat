"""
NanoChat Text Generation
Fast inference with Top-P sampling and temperature control.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional, List

from model import forward


# ============================================================================
# Sampling Utilities
# ============================================================================

def top_p_filter(logits: jnp.ndarray, p: float) -> jnp.ndarray:
    """
    Top-P (Nucleus) Sampling.
    Keep smallest set of tokens with cumulative probability >= p.
    """
    if p >= 1.0:
        return logits
    if p <= 0.0:
        return top_k_filter(logits, 1)
    
    # Sort logits descending
    sorted_indices = jnp.argsort(-logits, axis=-1)
    sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
    
    # Compute cumulative probabilities
    sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
    cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
    
    # Mask tokens above threshold
    sorted_mask = cumulative_probs - sorted_probs < p
    sorted_logits = jnp.where(sorted_mask, sorted_logits, -jnp.inf)
    
    # Unsort
    unsort_indices = jnp.argsort(sorted_indices, axis=-1)
    return jnp.take_along_axis(sorted_logits, unsort_indices, axis=-1)


def top_k_filter(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    """Keep only top-k logits."""
    if k <= 0:
        return logits

    k = min(int(k), logits.shape[-1])
    top_k_values = jax.lax.top_k(logits, k)[0]
    threshold = top_k_values[..., -1:]
    return jnp.where(logits < threshold, -jnp.inf, logits)


@partial(jax.jit, static_argnames=['temperature', 'top_p', 'top_k'])
def sample_next_token(key, logits, temperature=0.8, top_p=0.9, top_k=0):
    """Sample next token with temperature and nucleus sampling."""
    # Apply temperature
    logits = logits / jnp.maximum(temperature, 1e-8)
    
    # Apply top-k if specified
    if top_k > 0:
        logits = top_k_filter(logits, top_k)
    
    # Apply top-p
    logits = top_p_filter(logits, top_p)
    
    return jax.random.categorical(key, logits, axis=-1)


# ============================================================================
# JIT-Compiled Forward
# ============================================================================

@partial(jax.jit, static_argnames=['config'])
def forward_jit(params, tokens, config, mask, cos, sin):
    """JIT-compiled forward pass."""
    return forward(params, tokens, config, mask, cos, sin)


# ============================================================================
# Generation Functions
# ============================================================================

def generate(
    params,
    prompt_tokens: List[int],
    config,
    mask: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 0,
    key: Optional[jax.random.PRNGKey] = None,
    stop_tokens: Optional[List[int]] = None,
) -> List[int]:
    """
    Generate text autoregressively.
    
    Args:
        params: Model parameters
        prompt_tokens: List of token ids for the prompt
        config: Model configuration
        mask, cos, sin: Precomputed attention mask and RoPE
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling (0 = disabled)
        key: Random key for sampling
        stop_tokens: Optional list of tokens to stop generation
    
    Returns:
        List of generated token ids (including prompt)
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    if stop_tokens is None:
        stop_tokens = []
    
    generated = list(prompt_tokens)
    
    for _ in range(max_new_tokens):
        # Prepare input
        context = generated[-config.block_size:]
        seq_len = len(context)
        tokens = jnp.array([context], dtype=jnp.int32)
        
        # Forward pass
        logits = forward_jit(
            params, tokens, config,
            mask[:seq_len, :seq_len],
            cos[:seq_len],
            sin[:seq_len]
        )
        
        # Sample next token
        key, subkey = jax.random.split(key)
        next_logits = logits[0, -1, :]
        next_token = sample_next_token(subkey, next_logits, temperature, top_p, top_k)
        next_token = int(next_token)
        
        generated.append(next_token)
        
        # Check for stop tokens
        if next_token in stop_tokens:
            break
    
    return generated


def generate_greedy(
    params,
    prompt_tokens: List[int],
    config,
    mask: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    max_new_tokens: int = 100,
    stop_tokens: Optional[List[int]] = None,
) -> List[int]:
    """Greedy generation - always pick highest probability token."""
    if stop_tokens is None:
        stop_tokens = []
    
    generated = list(prompt_tokens)
    
    for _ in range(max_new_tokens):
        context = generated[-config.block_size:]
        seq_len = len(context)
        tokens = jnp.array([context], dtype=jnp.int32)
        
        logits = forward_jit(
            params, tokens, config,
            mask[:seq_len, :seq_len],
            cos[:seq_len],
            sin[:seq_len]
        )
        
        next_token = int(jnp.argmax(logits[0, -1, :]))
        generated.append(next_token)
        
        if next_token in stop_tokens:
            break
    
    return generated


def generate_stream(
    params,
    prompt_tokens: List[int],
    config,
    mask: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    key: Optional[jax.random.PRNGKey] = None,
):
    """
    Streaming generation - yields tokens one at a time.
    Useful for real-time display.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    generated = list(prompt_tokens)
    
    for _ in range(max_new_tokens):
        context = generated[-config.block_size:]
        seq_len = len(context)
        tokens = jnp.array([context], dtype=jnp.int32)
        
        logits = forward_jit(
            params, tokens, config,
            mask[:seq_len, :seq_len],
            cos[:seq_len],
            sin[:seq_len]
        )
        
        key, subkey = jax.random.split(key)
        next_logits = logits[0, -1, :]
        next_token = int(sample_next_token(subkey, next_logits, temperature, top_p))
        
        generated.append(next_token)
        yield next_token
