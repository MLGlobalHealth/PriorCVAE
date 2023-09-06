import jax
import jax.numpy as jnp
import numpy as np

from priorCVAE.utility import sq_euclidean_dist, generate_decoder_samples, decode
from priorCVAE.models import MLPDecoder


def true_sq_euclidean_distance(x1, x2):
    """Square Euclidean distance calculated as x1^2 + x2^2 - 2 * x1 * x2"""
    dist = jnp.sum(jnp.square(x1), axis=-1)[..., None] + jnp.sum(jnp.square(x2), axis=-1)[..., None].T - 2 * jnp.dot(x1, x2.T)
    return dist


def test_sq_euclidean_distance(num_data, dimension):
    """
    Test the square Euclidean distance utility function.
    """
    key = jax.random.PRNGKey(123)
    x1 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.01, maxval=1.)
    key, _ = jax.random.split(key)
    x2 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.01, maxval=1.)

    sq_eucliden_dist_val = sq_euclidean_dist(x1, x2)
    expected_val = true_sq_euclidean_distance(x1, x2)

    np.testing.assert_array_almost_equal(sq_eucliden_dist_val, expected_val, decimal=6)


def test_generate_decoder_samples(num_data):
    key = jax.random.PRNGKey(123)
    num_samples = num_data
    latent_dim = 5
    decoder = MLPDecoder(hidden_dim=10, out_dim=5)
    decoder_params = decoder.init(key, jnp.zeros((num_samples, latent_dim)))['params']

    samples = generate_decoder_samples(key, decoder_params, decoder, num_samples, latent_dim)

    assert samples.shape == (num_samples, 5)


def test_generate_decoder_samples_conditional(num_data):
    key = jax.random.PRNGKey(123)
    num_samples = num_data

    c = jax.random.uniform(key=key, shape=(num_samples, 1), minval=0.01, maxval=1.)
    decoder = MLPDecoder(hidden_dim=10, out_dim=5)

    latent_dim = 5
    decoder_params = decoder.init(key, jnp.zeros((num_samples, latent_dim + 1)))['params']

    samples = generate_decoder_samples(key, decoder_params, decoder, num_samples, latent_dim, c=c)

    assert samples.shape == (num_samples, 5)
