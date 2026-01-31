import jax
import jax.numpy as jnp

def rms_norm(x, gamma, eps=1e-6):
    ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(ms + eps) * gamma

def apply_rope(xq, xk, cos, sin):
    """Apply rotary position embeddings. Inputs: (batch, n_head, seq_len, head_dim)"""
    def rotate_half(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)
    # cos/sin are (seq_len, head_dim), need to broadcast to (1, 1, seq_len, head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (xq * cos) + (rotate_half(xq) * sin), (xk * cos) + (rotate_half(xk) * sin)

def self_attention(x, params, config, mask=None, cos=None, sin=None):
    batch, seq_len, _ = x.shape
    n_head = config.n_head
    head_dim = config.n_embd // n_head
    
    # Project to q, k, v
    q, k, v = jnp.dot(x, params['wq']), jnp.dot(x, params['wk']), jnp.dot(x, params['wv'])
    
    # Reshape to multi-head: (batch, seq_len, n_embd) -> (batch, n_head, seq_len, head_dim)
    q = q.reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
    
    if cos is not None and sin is not None:
        q, k = apply_rope(q, k, cos, sin)
    
    # Attention: (batch, n_head, seq_len, head_dim) @ (batch, n_head, head_dim, seq_len)
    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(head_dim)
    if mask is not None: scores = scores + mask
    
    weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(weights, v)  # (batch, n_head, seq_len, head_dim)
    
    # Reshape back: (batch, n_head, seq_len, head_dim) -> (batch, seq_len, n_embd)
    output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    return jnp.dot(output, params['wo']), weights

def transformer_block(x, params, config, mask=None, cos=None, sin=None):
    h = rms_norm(x, params['norm1'])
    attn_out, weights = self_attention(h, params['attn'], config, mask, cos, sin)
    x = x + attn_out
    h = rms_norm(x, params['norm2'])
    x = x + jnp.dot(jax.nn.gelu(jnp.dot(h, params['mlp']['w1'])), params['mlp']['w2'])
    return x, weights

def forward(params, tokens, config, mask, cos, sin, return_attn=False):
    x = params['token_emb'][tokens]
    all_weights = []
    for block_params in params['blocks']:
        x, weights = transformer_block(x, block_params, config, mask, cos, sin)
        if return_attn: all_weights.append(weights)
    logits = jnp.dot(rms_norm(x, params['final_norm']), params['output_head'])
    return (logits, all_weights) if return_attn else logits