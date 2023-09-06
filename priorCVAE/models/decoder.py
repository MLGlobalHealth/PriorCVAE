"""
File contains the Decoder models.
"""
from abc import ABC
from typing import Tuple, Union

from flax import linen as nn
import jax.numpy as jnp
from jaxlib.xla_extension import PjitFunction


class Decoder(ABC, nn.Module):
    """Parent class for decoder model."""
    def __init__(self):
        super().__init__()


class MLPDecoder(Decoder):
    """
    MLP decoder model with the structure:

    for _ in hidden_dims:
        z = Activation(Dense(z))
    y = Dense(z)

    Note: For the same activation functions for all hidden layers, pass a single function rather than a list.

    """
    hidden_dim: Union[Tuple[int], int]
    out_dim: int
    activations: Union[Tuple, PjitFunction] = nn.sigmoid

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        # If a single activation function or single hidden dimension is passed.
        hidden_dims = [self.hidden_dim] if isinstance(self.hidden_dim, int) else self.hidden_dim
        activations = [self.activations] * len(hidden_dims) if not isinstance(self.activations,
                                                                              Tuple) else self.activations

        for i, (hidden_dim, activation_fn) in enumerate(zip(hidden_dims, activations)):
            z = nn.Dense(hidden_dim, name=f"dec_hidden_{i}")(z)
            z = activation_fn(z)
        z = nn.Dense(self.out_dim, name="dec_out")(z)
        return z
