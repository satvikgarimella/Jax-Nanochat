import jax
import jax.numpy as jnp

def rms_norm(x, gamma, eps=1e-6):
    """
    Normalize the input vector x using the Root Mean Square. 
    x: input array
    gamma: learnable scale paremeter ( a leaf in our pytree)
    """

    #1. Square every number, take the average, and then the square root
    ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

    #2. Divide x by that root ( using rsrt for speed) and multiply by gamma
    # THis keeps the number from getting to big or too small
    return x * jax.lax.rsqrt(ms + eps) * gamma


    def apply_rope(xq, xk, cos, sin):
        """
        Rotates QUery (xq) and Key (xk) vectors based on their positions.
        """
        
        # Helper: splits the vector in half and flips/negates one half 
        def rotate_half(x):
            x1, x2 = jnp.split(x, 2, axis=-1)
            return jnp.concatenate([-x2, x1], axis=-1)

            # The math: x * cos + (roated_x) * sin
            # This is standard complex number rotation math in real-valued form
            xq_out = (xq * cos) + (rotate_half(xk) * sin)
            xk_out = (rotate_half(xq) * sin) + (xk * cos)

            return xq_out, xk_out
            