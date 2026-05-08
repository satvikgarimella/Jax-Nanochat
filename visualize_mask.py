import jax.numpy as jnp
import matplotlib.pyplot as plt

def create_causal_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len)))

seq_len = 10
mask = create_causal_mask(seq_len)

plt.figure(figsize=(8, 6))
plt.imshow(mask, cmap='Blues')
plt.title(f"Causal Mask ({seq_len}x{seq_len})")
plt.xlabel("Key Position")
plt.ylabel("Query Position")
plt.colorbar(label="Attention Allowed")
plt.show()
