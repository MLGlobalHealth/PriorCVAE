"""
Test the loss functions.
"""
import random
import jax
import numpy as np
import jax.numpy as jnp

from priorCVAE.losses import kl_divergence, scaled_sum_squared_loss, mean_squared_loss, square_maximum_mean_discrepancy
from priorCVAE.priors import SquaredExponential


def test_kl_divergence(dimension):
    """Test KL divergence between a Gaussian N(m, S) and a unit Gaussian N(0, I)"""
    key = jax.random.PRNGKey(random.randint(a=0, b=999))
    m = jax.random.uniform(key=key, shape=(dimension, ), minval=0.1, maxval=4.)
    log_S = jax.random.uniform(key=key, shape=(dimension,), minval=0.1, maxval=.9)

    kl_value = kl_divergence(m, log_S)

    expected_kl_value = -0.5 * (1 + log_S - jnp.exp(log_S) - jnp.square(m))
    expected_kl_value = jnp.sum(expected_kl_value)

    np.testing.assert_array_almost_equal(kl_value, expected_kl_value, decimal=6)


def test_scaled_sum_squared_loss(num_data, dimension):
    """Test scaled sum squared loss."""
    key = jax.random.PRNGKey(random.randint(a=0, b=999))
    y = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.1, maxval=4.)
    key, _ = jax.random.split(key)
    y_reconstruction = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.1, maxval=4.)
    vae_variance = jax.random.normal(key=key).item()

    vae_loss_val = scaled_sum_squared_loss(y, y_reconstruction, vae_variance)
    expected_val = jnp.sum(0.5 * (y - y_reconstruction)**2/vae_variance)

    np.testing.assert_array_almost_equal(vae_loss_val, expected_val, decimal=6)


def test_mean_squared_loss(num_data, dimension):
    """Test mean squared loss."""
    key = jax.random.PRNGKey(random.randint(a=0, b=999))
    y = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.1, maxval=4.)
    key, _ = jax.random.split(key)
    y_reconstruction = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.1, maxval=4.)

    vae_loss_val = mean_squared_loss(y, y_reconstruction)
    expected_val = jnp.mean((y_reconstruction - y)**2)

    np.testing.assert_array_almost_equal(vae_loss_val, expected_val, decimal=6)


def _true_sq_mmd_value(kernel, x1, x2, full: bool = True, biased: bool = False):
    """
    True Squared MMD value for testing.
    """
    x1_n = x1.shape[0]
    x2_n = x2.shape[0]
    term1 = 0
    if full:
        K_xx = kernel(x1, x1)
        if biased:
            term1 = (1 / (x1_n * x1_n)) * jnp.sum(K_xx)
        else:
            term1 = (1 / (x1_n * (x1_n - 1))) * (jnp.sum(K_xx) - jnp.trace(K_xx))

    K_yy = kernel(x2, x2)
    K_xy = kernel(x1, x2)

    if biased:
        term2 = (1 / (x2_n * x2_n)) * jnp.sum(K_yy)
    else:
        term2 = (1 / (x2_n * (x2_n - 1))) * (jnp.sum(K_yy) - jnp.trace(K_yy))

    term3 = (2 / (x1_n * x2_n)) * jnp.sum(K_xy)

    return term1 + term2 - term3


def test_square_maximum_mean_discrepancy(num_data, dimension, boolean_variable):
    """
    Test square_maximum_mean_discrepancy
    Here boolean variable is used for biased and unbiased version testing.
    """
    key = jax.random.PRNGKey(random.randint(a=0, b=999))
    x1 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.1, maxval=4.)
    key, _ = jax.random.split(key)

    x2_shape = jax.random.randint(key, (1, ), 2, 10).tolist() + [dimension]
    key, _ = jax.random.split(key)
    x2 = jax.random.uniform(key=key, shape=x2_shape, minval=0.1, maxval=4.)

    kernel = SquaredExponential(lengthscale=2., variance=1.)

    sq_mmd_val_grads = square_maximum_mean_discrepancy(kernel, x1, x2, efficient_grads=True, biased=boolean_variable)
    sq_mmd_val_full = square_maximum_mean_discrepancy(kernel, x1, x2, efficient_grads=False, biased=boolean_variable)

    expected_val_grads = _true_sq_mmd_value(kernel, x1, x2, full=False, biased=boolean_variable)
    expected_val_full = _true_sq_mmd_value(kernel, x1, x2, full=True, biased=boolean_variable)

    np.testing.assert_array_almost_equal(sq_mmd_val_grads, expected_val_grads, decimal=6)
    np.testing.assert_array_almost_equal(sq_mmd_val_full, expected_val_full, decimal=6)
