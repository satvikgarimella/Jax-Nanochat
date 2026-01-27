import jax.numpy as jnp
import matplotlib.pyplot as plt

# 1. Run your mask function
def create_causal_mask(seq_len):
    # This matches the code we wrote earlier
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    return mask

# 2. Generate a 10x10 mask for clear visualization
seq_len = 10
mask_data = create_causal_mask(seq_len)

# 3. Plot it
plt.figure(figsize=(8, 6))
plt.imshow(mask_data, cmap='Blues')
plt.title(f"Causal Mask Visualization ({seq_len}x{seq_len})")
plt.xlabel("Key Position (Past Tokens)")
plt.ylabel("Query Position (Current Token)")
plt.colorbar(label="Attention Allowed (1=Yes, 0=No)")
plt.show()