import jax
import jax.numpy as jnp
import equinox as eqx

class HamiltonianFlow(eqx.Module):
    """
    Defines the 'Energy Landscape' of the Balochi language.
    Sentences flow like fluids through this network.
    """
    net: eqx.nn.MLP

    def __init__(self, dim, key):
        # A simple field definition
        self.net = eqx.nn.MLP(in_size=dim, out_size=dim, width_size=128, depth=2, key=key)

    def __call__(self, t, y, args):
        """
        Differential Equation: dy/dt = Force(y)
        """
        force = self.net(y)
        return jnp.tanh(force) # Keep energy bounded