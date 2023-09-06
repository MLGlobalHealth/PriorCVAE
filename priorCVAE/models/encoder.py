"""
File contains the Encoder models.
"""
from abc import ABC
from typing import Tuple, Union

from flax import linen as nn
import jax.numpy as jnp
from jaxlib.xla_extension import PjitFunction


class Encoder(ABC, nn.Module):
    """Parent class for encoder model."""
    def __init__(self):
        super().__init__()


class MLPEncoder(Encoder):
    """
    MLP encoder model with the structure:

    for _ in hidden_dims:
        y = Activation(Dense(y))

    z_m = Dense(y)
    z_logvar = Dense(y)

    Note: For the same activation functions for all hidden layers, pass a single function rather than a list.

    """
    hidden_dim: Union[Tuple[int], int]
    latent_dim: int
    activations: Union[Tuple, PjitFunction] = nn.sigmoid

    @nn.compact
    def __call__(self, y: jnp.ndarray) -> (jnp.ndarray, jnp.ndarray):
        # If a single activation function or single hidden dimension is passed.
        hidden_dims = [self.hidden_dim] if isinstance(self.hidden_dim, int) else self.hidden_dim
        activations = [self.activations] * len(hidden_dims) if not isinstance(self.activations,
                                                                              Tuple) else self.activations

        for i, (hidden_dim, activation_fn) in enumerate(zip(hidden_dims, activations)):
            y = nn.Dense(hidden_dim, name=f"enc_hidden_{i}")(y)
            y = activation_fn(y)
        z_mu = nn.Dense(self.latent_dim, name="z_mu")(y)
        z_logvar = nn.Dense(self.latent_dim, name="z_logvar")(y)
        return z_mu, z_logvar
