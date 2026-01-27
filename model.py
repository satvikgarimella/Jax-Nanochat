import jax
import jax.numpy as jnp

def rms_norm(x, gamma, eps=1e-6):
    ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(ms + eps) * gamma

def apply_rope(xq, xk, cos, sin):
    def rotate_half(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)

    # Apply rotation: (x * cos) + (rotated_x * sin)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out, xk_out

def self_attention(x, params, config, mask=None, cos=None, sin=None):
    # 1. Project to Q, K, V
    q = jnp.dot(x, params['wq'])
    k = jnp.dot(x, params['wk'])
    v = jnp.dot(x, params['wv'])

    # 2. Apply RoPE to Q and K
    if cos is not None and sin is not None:
        q, k = apply_rope(q, k, cos, sin)

    # 3. Scaled Dot-Product Attention
    # (batch, seq, heads, dim) logic usually goes here, 
    # but for the "atomic" version:
    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(config['d_model'])
    
    if mask is not None:
        scores = scores + mask

    weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(weights, v)
    
    # 4. Final projection
    return jnp.dot(output, params['wo'])

def transformer_block(x, params, config, mask=None, cos=None, sin=None):
    # Layer 1: Attention + Residual
    h = rms_norm(x, params['norm1'])
    x = x + self_attention(h, params['attn'], config, mask, cos, sin)
    
    # Layer 2: MLP + Residual
    h = rms_norm(x, params['norm2'])
    # mlp_block usually = jnp.dot(gelu(jnp.dot(h, w1)), w2)
    
    return x

def init_transformer_params(key, config):
    """
    Initialize all parameters for the Transformer model.
    Returns a Pytree (nested dictionary) of weights.
    """
    # Split keys for different parts of the model
    k_embed, k_blocks, k_final = jax.random.split(key, 3)
    
    params = {
        'token_embedding': jax.random.normal(k_embed, (config.vocab_size, config.d_model)) * 0.02,
        # List comprehension to init each block with a unique key
        'blocks': [
            init_block_params(jax.random.fold_in(k_blocks, i), config) 
            for i in range(config.num_layers)
        ],
        'final_norm_g': jnp.ones((config.d_model,))
    }
    return params

def forward(params, tokens, config, mask, cos, sin):
    """
    The full data flow.
    x starts as token IDs and ends as vocab logits.
    """
    # 1. Embedding lookup
    x = params['token_emb'][tokens] 
    
    # 2. Sequential Blocks
    for block_params in params['blocks']:
        x = transformer_block(x, block_params, config, mask, cos, sin)
        
    # 3. Final Output Prep
    x = rms_norm(x, params['final_norm'])
    logits = jnp.dot(x, params['output_head'])
    
    return logits