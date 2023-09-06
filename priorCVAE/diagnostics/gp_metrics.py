import jax.numpy as jnp
import jax


@jax.jit
def frobenius_norm_of_diff(mat1: jnp.ndarray, mat2: jnp.ndarray):
    """
    Computes the frobenius norm of the difference of two matrices.

    :param mat1: array of shape (D, D).
    :param mat2: array of shape (D, D).
    :return: norm of the difference.
    """
    diff = mat1 - mat2
    norm = jnp.linalg.norm(diff)
    return norm

@jax.jit
def sample_covariance(samples: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the empirical covariance matrix from samples.

    :param samples: array of shape (num_samples, dimensions).
    :return: covariance matrix of the samples.
    """
    covariance = jnp.cov(samples, rowvar=False)
    return covariance
