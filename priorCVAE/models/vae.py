"""
File contains the variational autoencoder (VAE) class.

The class is baed on the flax VAE example: https://github.com/google/flax/blob/main/examples/vae/train.py.
"""

from jax import random
import jax.numpy as jnp
from flax import linen as nn

from priorCVAE.models.encoder import Encoder
from priorCVAE.models.decoder import Decoder


class VAE(nn.Module):
    """
    Variational autoencoder (VAE) class binding the encoder and decoder model together.
    """
    encoder: Encoder
    decoder: Decoder

    @nn.compact
    def __call__(self, y: jnp.ndarray, z_rng: random.KeyArray, c: jnp.ndarray = None) -> (jnp.ndarray, jnp.ndarray,
                                                                                          jnp.ndarray):
        """

        :parma y: a Jax ndarray of the shape, (N, D_{observed}).
        :param z_rng: a PRNG key used as the random key.
        :param c: a Jax ndarray used for cVAE of the shape, (N, C).

        Returns: a list of three values: output of the decoder, mean of the latent z, logvar of the latent z.

        """
        def reparameterize(z_rng, mean, logvar):
            """Sampling using the reparameterization trick."""
            std = jnp.exp(0.5 * logvar)
            eps = random.normal(z_rng, logvar.shape)
            return mean + eps * std

        if c is not None:
            y = jnp.concatenate([y, c], axis=-1)

        z_mu, z_logvar = self.encoder(y)
        z = reparameterize(z_rng, z_mu, z_logvar)

        if c is not None:
            z = jnp.concatenate([z, c], axis=-1)

        y_hat = self.decoder(z)

        return y_hat, z_mu, z_logvar
