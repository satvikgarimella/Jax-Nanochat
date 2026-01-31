import jax
import jax.numpy as jnp
from model import forward
from config import config # Using your Config instance

def generate(params, prompt_tokens, config, max_new_tokens=10):
    """
    Greedy generation: always picks the most likely next word.
    """
    generated = list(prompt_tokens)
    
    for _ in range(max_new_tokens):
        # Convert to JAX array and add batch dimension
        x = jnp.array([generated])
        
        # Forward pass to get predictions (logits)
        # Note: We don't need RoPE or Masks for this simple greedy test
        logits = forward(params, x, config, mask=None, cos=None, sin=None)
        
        # Get the logits for the VERY LAST token predicted
        next_token_logits = logits[0, -1, :]
        
        # Pick the winner (highest probability)
        next_token = jnp.argmax(next_token_logits)
        
        generated.append(int(next_token))
        
        # Stop if we hit a limit or a specific end token
        if len(generated) >= config.block_size:
            break
            
    return generated

# Example Run:
# start_tokens = [1, 2, 3]
# result = generate(params, start_tokens, config)
# print(f"Generated sequence: {result}")