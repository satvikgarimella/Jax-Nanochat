import jax
import jax.numpy as jnp
from init import init_params
from utils import create_causal_mask, precompute_rope
from model import forward

# Setup Dummy Data
config = {"vocab_size": 1000, "d_model": 512, "num_layers": 4, "head_dim": 64}
key = jax.random.PRNGKey(42)
tokens = jax.random.randint(key, (1, 128), 0, 1000) # Batch of 1, seq len 128

# 1. Initialize Params
params = init_params(key, config)

# 2. Pre-compute static logic (Mask, RoPE)
mask = create_causal_mask(128)
cos, sin = precompute_rope(128, config["d_model"]) # Use d_model since Q/K are (batch, seq, d_model)

# 3. THE JIT TEST
jitted_forward = jax.jit(forward)

print("Compiling and running first batch...")
output = jitted_forward(params, tokens, config, mask, cos, sin)
print(f"Success! Output shape: {output.shape}")