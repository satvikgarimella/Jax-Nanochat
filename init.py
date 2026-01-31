import jax
import jax.numpy as jnp

def init_params(key, config):
    """
    Creates the 'Fuel Tank' for your model.
    A nested dictionary containing every weight in the system.
    """
    k_embed, k_blocks, k_out = jax.random.split(key, 3)
    
    # Standard initializer (Llama/GPT style)
    def init_weight(k, shape):
        return jax.random.normal(k, shape) * 0.02

    params = {
        'token_emb': init_weight(k_embed, (config.vocab_size, config.n_embd)),
        'blocks': [
            {
                'attn': {
                    'wq': init_weight(jax.random.fold_in(k_blocks, i*4+0), (config.n_embd, config.n_embd)),
                    'wk': init_weight(jax.random.fold_in(k_blocks, i*4+1), (config.n_embd, config.n_embd)),
                    'wv': init_weight(jax.random.fold_in(k_blocks, i*4+2), (config.n_embd, config.n_embd)),
                    'wo': init_weight(jax.random.fold_in(k_blocks, i*4+3), (config.n_embd, config.n_embd)),
                },
                'mlp': {
                    'w1': init_weight(jax.random.fold_in(k_blocks, i*2+100), (config.n_embd, config.n_embd * 4)),
                    'w2': init_weight(jax.random.fold_in(k_blocks, i*2+101), (config.n_embd * 4, config.n_embd)),
                },
                'norm1': jnp.ones((config.n_embd,)),
                'norm2': jnp.ones((config.n_embd,)),
            }
            for i in range(config.n_layer)
        ],
        'final_norm': jnp.ones((config.n_embd,)),
        'output_head': init_weight(k_out, (config.n_embd, config.vocab_size))
    }
    return params