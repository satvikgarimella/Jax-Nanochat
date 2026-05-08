import jax.numpy as jnp

def create_causal_mask(seq_len):
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    return jnp.where(mask, 0.0, -jnp.inf)

def precompute_rope(seq_len, head_dim, theta=10000.0):
    dims = jnp.arange(0, head_dim, 2)
    freqs = 1.0 / (theta ** (dims / head_dim))
    pos = jnp.arange(seq_len)
    angles = jnp.outer(pos, freqs)
    angles = jnp.repeat(angles, 2, axis=-1)
    return jnp.cos(angles), jnp.sin(angles)
