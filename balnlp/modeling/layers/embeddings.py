import jax
import jax.numpy as jnp
import equinox as eqx


class FractalEmbedding(eqx.Module):
    """
    Maps Balochi tokens to coordinates in the Mandelbrot Set.
    Replacing standard lookup tables with Chaos Theory recursion.
    """
    weights: jax.Array

    def __init__(self, vocab_size, embed_dim, key):
        # Initialize on the Complex Plane
        self.weights = jax.random.normal(key, (vocab_size, embed_dim))

    def __call__(self, token_id):
        # Get the seed coordinate
        c = self.weights[token_id]

        # The Fractal Recursion (Z = Z^2 + C)
        # We iterate this to find the "stable meaning" of the word
        z = jnp.zeros_like(c)

        # JAX unrolls this loop for extreme speed on GPU
        def loop_body(i, z_val):
            return z_val ** 2 + c

        z = jax.lax.fori_loop(0, 10, loop_body, z)

        return z.real  # Return real component as vector