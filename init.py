"""
NanoChat Parameter Initialization
Proper initialization for stable training at any scale.
"""

import jax
import jax.numpy as jnp
import math


def init_params(key, config):
    """
    Initialize model parameters with proper scaling.
    
    Uses:
    - Normal distribution with std=0.02 for most weights
    - Scaled initialization for residual layers (1/sqrt(2*n_layer))
    - Ones for RMSNorm gamma parameters
    """
    k_embed, k_blocks, k_out = jax.random.split(key, 3)
    
    # Standard deviation for normal initialization
    std = 0.02
    
    # Residual scaling factor (GPT-2 style)
    residual_scale = 1.0 / math.sqrt(2 * config.n_layer)
    
    def init_weight(k, shape, scale=1.0):
        """Initialize weight with normal distribution."""
        return jax.random.normal(k, shape) * std * scale
    
    def init_embedding(k, shape):
        """Initialize embedding with slightly smaller std."""
        return jax.random.normal(k, shape) * 0.01
    
    # Build parameter tree
    params = {
        # Token embeddings
        'token_emb': init_embedding(k_embed, (config.vocab_size, config.n_embd)),
        
        # Transformer blocks
        'blocks': [
            {
                'attn': {
                    'wq': init_weight(jax.random.fold_in(k_blocks, i*4+0), 
                                      (config.n_embd, config.n_embd)),
                    'wk': init_weight(jax.random.fold_in(k_blocks, i*4+1), 
                                      (config.n_embd, config.n_embd)),
                    'wv': init_weight(jax.random.fold_in(k_blocks, i*4+2), 
                                      (config.n_embd, config.n_embd)),
                    # Output projection scaled for residual
                    'wo': init_weight(jax.random.fold_in(k_blocks, i*4+3), 
                                      (config.n_embd, config.n_embd), residual_scale),
                },
                'mlp': {
                    'w1': init_weight(jax.random.fold_in(k_blocks, i*2+100), 
                                      (config.n_embd, config.n_embd * 4)),
                    # Output projection scaled for residual
                    'w2': init_weight(jax.random.fold_in(k_blocks, i*2+101), 
                                      (config.n_embd * 4, config.n_embd), residual_scale),
                },
                # RMSNorm gamma (initialized to 1)
                'norm1': jnp.ones((config.n_embd,)),
                'norm2': jnp.ones((config.n_embd,)),
            }
            for i in range(config.n_layer)
        ],
        
        # Final layer norm
        'final_norm': jnp.ones((config.n_embd,)),
        
        # Output projection (unembedding)
        'output_head': init_weight(k_out, (config.n_embd, config.vocab_size))
    }
    
    return params


def count_params(params):
    """Count total number of parameters."""
    return sum(x.size for x in jax.tree.leaves(params))


def get_param_shapes(params, prefix=""):
    """Get shapes of all parameters (for debugging)."""
    shapes = {}
    
    def _traverse(p, path):
        if isinstance(p, dict):
            for k, v in p.items():
                _traverse(v, f"{path}.{k}" if path else k)
        elif isinstance(p, list):
            for i, v in enumerate(p):
                _traverse(v, f"{path}[{i}]")
        else:
            shapes[path] = p.shape
    
    _traverse(params, prefix)
    return shapes


if __name__ == "__main__":
    # Test initialization
    from dataclasses import dataclass
    
    @dataclass
    class TestConfig:
        vocab_size: int = 50257
        n_layer: int = 6
        n_head: int = 6
        n_embd: int = 384
    
    config = TestConfig()
    key = jax.random.PRNGKey(42)
    params = init_params(key, config)
    
    print(f"Total parameters: {count_params(params):,}")
    print("\nParameter shapes:")
    for name, shape in get_param_shapes(params).items():
        print(f"  {name}: {shape}")
