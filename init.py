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
        'token_emb': init_weight(k_embed, (config['vocab_size'], config['d_model'])),
        'blocks': [
            {
                'attn': {
                    'wq': init_weight(jax.random.fold_in(k_blocks, i*4+0), (config['d_model'], config['d_model'])),
                    'wk': init_weight(jax.random.fold_in(k_blocks, i*4+1), (config['d_model'], config['d_model'])),
                    'wv': init_weight(jax.random.fold_in(k_blocks, i*4+2), (config['d_model'], config['d_model'])),
                    'wo': init_weight(jax.random.fold_in(k_blocks, i*4+3), (config['d_model'], config['d_model'])),
                },
                'mlp': {
                    'w1': init_weight(jax.random.fold_in(k_blocks, i*2+100), (config['d_model'], config['d_model'] * 4)),
                    'w2': init_weight(jax.random.fold_in(k_blocks, i*2+101), (config['d_model'] * 4, config['d_model'])),
                },
                'norm1': jnp.ones((config['d_model'],)),
                'norm2': jnp.ones((config['d_model'],)),
            }
            for i in range(config['num_layers'])
        ],
        'final_norm': jnp.ones((config['d_model'],)),
        'output_head': init_weight(k_out, (config['d_model'], config['vocab_size']))
    }
    return params