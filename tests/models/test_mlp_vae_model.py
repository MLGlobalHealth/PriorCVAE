"""
Test the MLPEncoder, MLPDecoder, VAE models.
"""
import pytest

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from priorCVAE.models import MLPEncoder, MLPDecoder, VAE


@pytest.fixture(name="hidden_dimension", params=[2, 8, [8, 5], [10, 4, 2]])
def _hidden_dimension_fixture(request):
    return request.param


@pytest.fixture(name="latent_dimension", params=[2, 8])
def _latent_dimension_fixture(request):
    return request.param


@pytest.fixture(name="data_dimension", params=[2, 8])
def _data_dimension_fixture(request):
    return request.param


@pytest.fixture(name="hidden_dim_activations", params=[[2, nn.sigmoid], [[4], [nn.sigmoid]],
                                                       [[10, 4, 2], nn.leaky_relu],
                                                       [[5, 3, 1], [nn.sigmoid, nn.leaky_relu, nn.sigmoid]]])
def _hidden_dim_activations_fixture(request):
    return request.param


def test_encoder_model_shape(num_data, data_dimension, hidden_dimension, latent_dimension):
    """Test the shape of the outputs of Encoder model"""
    encoder = MLPEncoder(hidden_dim=hidden_dimension, latent_dim=latent_dimension)
    x = jnp.zeros((num_data, data_dimension))

    rng = jax.random.PRNGKey(0)
    params = encoder.init(rng, x)['params']

    variables = {"params": params}
    z_m, z_logvar = encoder.apply(variables, x)

    assert z_m.shape == (num_data, latent_dimension)
    assert z_logvar.shape == (num_data, latent_dimension)


def test_decoder_model_shape(num_data, data_dimension, hidden_dimension, latent_dimension):
    """Test the shape of the outputs of Decoder model"""
    decoder = MLPDecoder(hidden_dim=hidden_dimension, out_dim=data_dimension)
    z = jnp.zeros((num_data, latent_dimension))

    rng = jax.random.PRNGKey(0)
    params = decoder.init(rng, z)['params']

    variables = {"params": params}
    x = decoder.apply(variables, z)

    assert x.shape == (num_data, data_dimension)


def test_vae_model_shape(num_data, data_dimension, hidden_dimension, latent_dimension):
    """Test the shape of the outputs of VAE model"""
    encoder = MLPEncoder(hidden_dim=hidden_dimension, latent_dim=latent_dimension)
    decoder = MLPDecoder(hidden_dim=hidden_dimension, out_dim=data_dimension)

    vae = VAE(encoder=encoder, decoder=decoder)
    x = jnp.zeros((num_data, data_dimension))
    rng = jax.random.PRNGKey(0)
    params = vae.init(rng, x, rng)['params']

    variables = {"params": params}
    x, z_m, z_logvar = vae.apply(variables, x, rng)

    assert x.shape == (num_data, data_dimension)
    assert z_m.shape == (num_data, latent_dimension)
    assert z_logvar.shape == (num_data, latent_dimension)


def test_vae_reparameterize(num_data, data_dimension, hidden_dimension, latent_dimension):
    """Test the reparameterization of the VAE."""
    encoder = MLPEncoder(hidden_dim=hidden_dimension, latent_dim=latent_dimension)
    decoder = MLPDecoder(hidden_dim=hidden_dimension, out_dim=data_dimension)

    vae = VAE(encoder=encoder, decoder=decoder)
    x = jnp.zeros((num_data, data_dimension))
    rng = jax.random.PRNGKey(0)
    params = vae.init(rng, x, rng)['params']

    variables = {"params": params}
    x, z_m, z_logvar = vae.apply(variables, x, rng)

    # sample from a Gaussian
    std = jnp.exp(0.5 * z_logvar)
    eps = jax.random.normal(rng, z_logvar.shape)
    z_sample = z_m + eps * std
    # Pass it through the decoder
    expected_x = decoder.apply({"params": params["decoder"]}, z_sample)

    np.testing.assert_array_almost_equal(x, expected_x)


def test_dense_vae_structure(hidden_dim_activations, latent_dimension, data_dimension, num_data):
    """
    Test VAE model structure when multiple hidden layers and activation functions are passed.
    """
    hidden_dimension, activation_fn = hidden_dim_activations
    encoder = MLPEncoder(hidden_dim=hidden_dimension, latent_dim=latent_dimension, activations=activation_fn)
    decoder = MLPDecoder(hidden_dim=hidden_dimension, out_dim=data_dimension, activations=activation_fn)

    vae = VAE(encoder=encoder, decoder=decoder)
    x = jnp.zeros((num_data, data_dimension))
    rng = jax.random.PRNGKey(0)
    params = vae.init(rng, x, rng)['params']

    variables = {"params": params}
    x, z_m, z_logvar = vae.apply(variables, x, rng)

    assert x.shape == (num_data, data_dimension)
    assert z_m.shape == (num_data, latent_dimension)
    assert z_logvar.shape == (num_data, latent_dimension)
