import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
from modeling.layers.dynamics import HamiltonianFlow
from modeling.layers.embeddings import FractalEmbedding


class BalochiTransformer(eqx.Module):
    embedding: FractalEmbedding
    dynamics: HamiltonianFlow
    decoder: eqx.nn.Linear

    def __init__(self, vocab_size, dim, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.embedding = FractalEmbedding(vocab_size, dim, key=k1)
        self.dynamics = HamiltonianFlow(dim, key=k2)
        self.decoder = eqx.nn.Linear(dim, vocab_size, key=k3)

    def __call__(self, token_ids):
        # 1. Fractalize Inputs (Batch processing)
        # vmap allows parallel processing of all tokens
        y0 = jax.vmap(self.embedding)(token_ids)

        # Pool sentence into a single state (Simplified for prototype)
        y_state = jnp.mean(y0, axis=0)

        # 2. Evolve in Continuous Time (ODE Solver)
        term = diffrax.ODETerm(self.dynamics)
        solver = diffrax.Tsit5()

        # Solve from Time=0 to Time=1
        solution = diffrax.diffeqsolve(
            term, solver, t0=0, t1=1, dt0=0.1, y0=y_state
        )

        final_thought = solution.ys[-1]

        # 3. Project back to Vocabulary
        return self.decoder(final_thought)