"""
NanoChat Transformer Model
Clean implementation with RoPE, causal attention, and bfloat16 support.
"""

import jax
import jax.numpy as jnp
from functools import partial


# ============================================================================
# Precision Utilities
# ============================================================================

def cast_to_dtype(params, dtype):
    """Cast all parameters to specified dtype."""
    def cast_fn(x):
        if x.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]:
            return x.astype(dtype)
        return x
    return jax.tree.map(cast_fn, params)


# ============================================================================
# Core Operations
# ============================================================================

def rms_norm(x, gamma, eps=1e-6):
    """RMSNorm: simpler and faster than LayerNorm."""
    # Compute in float32 for stability, then cast back
    x_f32 = x.astype(jnp.float32)
    ms = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    normed = x_f32 * jax.lax.rsqrt(ms + eps)
    return (normed * gamma).astype(x.dtype)


def apply_rope(xq, xk, cos, sin):
    """Apply Rotary Position Embeddings."""
    def rotate_half(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)
    
    cos = cos[None, None, :, :].astype(xq.dtype)
    sin = sin[None, None, :, :].astype(xq.dtype)
    
    return (xq * cos) + (rotate_half(xq) * sin), (xk * cos) + (rotate_half(xk) * sin)


def self_attention(x, params, config, mask=None, cos=None, sin=None):
    """Multi-head self-attention with RoPE."""
    batch, seq_len, _ = x.shape
    n_head = config.n_head
    head_dim = config.n_embd // n_head
    
    # Project to Q, K, V
    q = jnp.dot(x, params['wq'])
    k = jnp.dot(x, params['wk'])
    v = jnp.dot(x, params['wv'])
    
    # Reshape to multi-head
    q = q.reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
    
    # Apply RoPE
    if cos is not None and sin is not None:
        q, k = apply_rope(q, k, cos, sin)
    
    # Attention scores
    scale = jnp.sqrt(jnp.array(head_dim, dtype=x.dtype))
    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / scale
    
    if mask is not None:
        scores = scores + mask.astype(scores.dtype)
    
    # Softmax and weighted sum
    weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
    output = jnp.matmul(weights, v)
    
    # Reshape back
    output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    
    return jnp.dot(output, params['wo']), weights


def transformer_block(x, params, config, mask=None, cos=None, sin=None):
    """Single transformer block: attention + MLP."""
    # Pre-norm attention
    h = rms_norm(x, params['norm1'])
    attn_out, weights = self_attention(h, params['attn'], config, mask, cos, sin)
    x = x + attn_out
    
    # Pre-norm MLP
    h = rms_norm(x, params['norm2'])
    mlp_out = jnp.dot(jax.nn.gelu(jnp.dot(h, params['mlp']['w1'])), params['mlp']['w2'])
    x = x + mlp_out
    
    return x, weights


def forward(params, tokens, config, mask, cos, sin, return_attn=False):
    """Full forward pass through the transformer."""
    # Embed tokens
    x = params['token_emb'][tokens]
    
    # Cast to model dtype if needed
    if hasattr(config, 'dtype') and config.dtype == "bfloat16":
        x = x.astype(jnp.bfloat16)
    
    all_weights = []
    
    # Transformer blocks
    for block_params in params['blocks']:
        x, weights = transformer_block(x, block_params, config, mask, cos, sin)
        if return_attn:
            all_weights.append(weights)
    
    # Final norm and output projection
    x = rms_norm(x, params['final_norm'])
    logits = jnp.dot(x, params['output_head'])
    
    # Cast logits to float32 for stable softmax
    logits = logits.astype(jnp.float32)
    
    return (logits, all_weights) if return_attn else logits


# ============================================================================
# JIT-compiled forward for inference
# ============================================================================

@partial(jax.jit, static_argnames=['config'])
def forward_jit(params, tokens, config, mask, cos, sin):
    """JIT-compiled forward pass."""
    return forward(params, tokens, config, mask, cos, sin, return_attn=False)
