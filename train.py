import jax
import jax.numpy as jnp
import optax
import numpy as np
from model import forward
from config import config # Corrected: Imports the instance from config.py
from init import init_params
from utils import create_causal_mask, precompute_rope

# 1. Initialize Weights
key = jax.random.PRNGKey(42)
params = init_params(key, config) 
optimizer = optax.adam(learning_rate=config.learning_rate)
opt_state = optimizer.init(params)

# 2. Create training data (simple synthetic example)
seq_len = 32  # sequence length for training
key, data_key = jax.random.split(key)
tokens = jax.random.randint(data_key, (1, seq_len), 0, config.vocab_size)
targets = jnp.roll(tokens, -1, axis=-1)  # next-token prediction

# 3. Create mask and RoPE embeddings
mask = create_causal_mask(seq_len)
head_dim = config.n_embd // config.n_head
cos, sin = precompute_rope(seq_len, head_dim)

@jax.jit
def update_step(params, opt_state, tokens, targets, mask, cos, sin):
    def loss_fn(p):
        logits = forward(p, tokens, config, mask, cos, sin)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
        return -jnp.mean(target_log_probs)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), new_opt_state, loss

# 4. Train and Export
for step in range(500):
    params, opt_state, loss = update_step(params, opt_state, tokens, targets, mask, cos, sin)
    if step % 50 == 0: print(f"Step {step}, Loss: {loss:.4f}")

_, weights = forward(params, tokens, config, mask, cos, sin, return_attn=True)
np.save("real_attn.npy", np.array(weights[0][0, 0])) # Layer 0, Head 0