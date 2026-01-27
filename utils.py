import jax.numpy as jnp

def create_causal_mask(seq_len):
    """
    Creates a causal attention mask to prevent attending to future tokens.
    Returns a (seq_len, seq_len) mask where mask[i, j] = 1 if j <= i, else 0.
    """
    return jnp.tril(jnp.ones((seq_len, seq_len)))

def precompute_rope(seq_len, head_dim, theta=10000.0):
    """
    Precomputes the cos and sin angles for RoPE.
    head_dim: dimension of a single attention head (must be even).
    """
    # 1. Create the frequencies for each dimension pair
    # We only need dim/2 because RoPE rotates 2D pairs
    dims = jnp.arange(0, head_dim, 2)
    freqs = 1.0 / (theta ** (dims / head_dim))
    
    # 2. Create the position indices [0, 1, ..., seq_len-1]
    pos = jnp.arange(seq_len)
    
    # 3. Outer product to get angles for every (position, dimension)
    # Resulting shape: (seq_len, head_dim / 2)
    angles = jnp.outer(pos, freqs)
    
    # 4. Duplicate to match the full head_dim
    # We want [a, b, c] -> [a, a, b, b, c, c]
    angles = jnp.repeat(angles, 2, axis=-1)
    
    return jnp.cos(angles), jnp.sin(angles)