"""
File contains the Gaussian process numpyro primitive.
"""
import numpyro
import numpyro.distributions as npdist
import jax.numpy as jnp

from priorCVAE.priors import Kernel, SquaredExponential


def GP(x: jnp.ndarray, kernel: Kernel = SquaredExponential(), jitter: float = 1e-5, y=None, noise: bool = False,
       sample_lengthscale: bool = False):
    """
    Gaussian process numpyro primitive to generate samples from it.

    :param x: Jax ndarray of the shape (N, D)
    :param kernel: Gaussian process kernel, object of the class priorCVAE.priors.Kernel
    :param jitter: Float value added to the kernel matrix.
    :param y: observations.
    :param noise: if True, add noise to the sample. The noise is drawn from the half-normal distribution with
                  variance of 0.1.
    :param sample_lengthscale: if True, sample lenthscale from a Uniform distribution, U(0.01, 0.99).

    """

    if sample_lengthscale:
        kernel.lengthscale = numpyro.sample("lengthscale", npdist.Uniform(0.01, 0.99))

    k = kernel(x, x)
    k += jitter * jnp.eye(x.shape[0])

    if noise is False:
        f = numpyro.sample("y", npdist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k), obs=y)
    else:
        sigma = numpyro.sample("noise", npdist.HalfNormal(0.1))
        f = numpyro.sample("f", npdist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k))
        y = numpyro.sample("y", npdist.Normal(f, sigma), obs=y)

    ls = numpyro.deterministic("ls", jnp.array([kernel.lengthscale]))
