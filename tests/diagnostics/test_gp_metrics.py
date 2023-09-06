import jax
import jax.numpy as jnp
import numpy as np
import random

from priorCVAE.diagnostics import frobenius_norm_of_diff


def test_frobenius_norm_of_diff(num_data, dimension):
    key = jax.random.PRNGKey(random.randint(a=0, b=999))
    matrix1 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.1, maxval=4.)
    key, _ = jax.random.split(key)
    matrix2 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.1, maxval=4.)
    norm = frobenius_norm_of_diff(matrix1, matrix2)

    expected_val = jnp.linalg.norm(matrix1 - matrix2)
    np.testing.assert_array_almost_equal(norm, expected_val, decimal=6)


def test_frobenius_norm_of_diff_is_zero_when_identical(num_data, dimension):
    key = jax.random.PRNGKey(random.randint(a=0, b=999))
    matrix = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.1, maxval=4.)
    norm = frobenius_norm_of_diff(matrix, matrix)
    np.testing.assert_array_equal(norm, 0.)


def test_frobenius_norm_of_diff_greater_or_equal_to_zero(num_data, dimension):
    key = jax.random.PRNGKey(random.randint(a=0, b=999))
    matrix1 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.1, maxval=4.)
    key, _ = jax.random.split(key)
    matrix2 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.1, maxval=4.)
    norm = frobenius_norm_of_diff(matrix1, matrix2)
    assert jnp.greater_equal(norm, 0.)
