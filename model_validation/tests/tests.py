"""
File contains tests for validating VAE models.
"""

import jax.numpy as jnp
import numpy as np

from priorCVAE.diagnostics import frobenius_norm_of_diff, sample_covariance
from priorCVAE.priors import Kernel, SquaredExponential
from .utils import mean_bootstrap_interval


def bootstrap_mean_test(samples: jnp.ndarray, **kwargs) -> jnp.ndarray:
    """
    Test if zero lies within the 5th and 95th quantiles of the bootstrap distribution of the mean.

    :param samples: samples to be tested.
    :return: Proportion of bootstrap intervals containing zero.
    """
    ci_lower, ci_upper = mean_bootstrap_interval(samples)
    zero_in_interval = (ci_lower <= 0) & (0 <= ci_upper)
    num_valid = jnp.where(zero_in_interval)[0].shape[0]
    return num_valid / samples.shape[1]


def norm_of_kernel_diff(samples: jnp.ndarray, kernel: Kernel, grid: jnp.array, **kwargs) -> jnp.ndarray:
    """
    Calculate the norm of the difference of the empirical covariance matrix and the kernel covariance.

    :param samples: samples used to compute empirical covariance.
    :param kernel: kernel used to compute covariance.
    :param grid: grid used to compute kernel covariance.
    :return: norm of the difference between the sample covariance and the kernel.
    """
    cov = sample_covariance(samples)
    norm = frobenius_norm_of_diff(cov, kernel(grid, grid))
    return norm


def bootstrap_covariance_test(samples: jnp.ndarray, kernel: Kernel, grid: jnp.array, sample_size=4000, num_iterations=1000, **kwargs):
    """
    Test if the kernel matrix lies within the 5th and 95th quantiles of the bootstrap distribution of the sample covariance matrix.

    :param samples: samples to be tested.
    :param kernel: kernel used to compute covariance.
    :param grid: grid used to compute kernel covariance.
    :param sample_size: size of each bootstrap sample.
    :param num_iterations: number of bootstrap samples.
    :return: proportion of points at which the test passes.
    """
    
    stats = []
    n = samples.shape[0]
    for _ in range(num_iterations):
        bootstrap_sample = samples[np.random.choice(n, size=sample_size, replace=True)]
        stat = sample_covariance(bootstrap_sample).flatten()
        stats.append(stat)

    stats = np.array(stats)

    ci_lower = np.percentile(stats, 5, axis=0)
    ci_upper = np.percentile(stats, 95, axis=0)

    K = kernel(grid,grid).flatten()

    in_range = (ci_lower <= K) & (K <= ci_upper)
    valid_idx = np.where(in_range)
    
    num_valid = valid_idx[0].shape[0]
    total = in_range.shape[0]

    return num_valid/total


def mmd_two_sample_test(samples, target_samples, num_permutations=1000, **kwargs):
    """
    Test if the two samples come from the same distribution using the MMD test.

    :param samples: samples to be tested.
    :param target_samples: samples from the target distribution.
    :param num_permutations: number of permutations to use.
    :return: True if the test passes.
    """
    # Compute the observed test statistic.
    observed_mmd = multi_kernel_mmd(samples, target_samples)

    # Pool the distributions.
    pooled = np.concatenate([samples, target_samples])

    # Generate permutations.
    mmd_permutations = []
    for _ in range(num_permutations):
        np.random.shuffle(pooled)
        mmd_permutations.append(multi_kernel_mmd(pooled[:len(samples)], pooled[len(samples):]))

    # Compute the threshold for a significance level of 0.05.
    mmd_permutations = np.array(mmd_permutations)
    threshold = np.percentile(mmd_permutations, 95)

    # Pass test if within threshold.
    return not observed_mmd > threshold


def squared_mmd(X, Y, kernel): 
    """
    Implementation of Empirical Maximum Mean Discrepancy (MMD).
    For details see lemma 6 of https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf

    :param X: first set of samples.
    :param Y: second set of samples.
    :param kernel: Kernel instance.
    """
 
    X = jnp.array(X)
    Y = jnp.array(Y)
    K_XX = kernel(X, X)
    K_YY = kernel(Y, Y)
    K_XY = kernel(X, Y)

    n = K_XX.shape[0]
    m = K_YY.shape[0]

    mmd_squared = (jnp.sum(K_XX) - jnp.trace(K_XX)) / (n * (n - 1)) + (jnp.sum(K_YY) - jnp.trace(K_YY)) / (
                m * (m - 1)) - 2 * jnp.sum(K_XY) / (m * n)

    return mmd_squared


def multi_kernel_mmd(X, Y, kernels=[SquaredExponential(0.2), SquaredExponential(4.0), SquaredExponential(8.0)]):
    """
    Compute the average MMD using multiple kernels.

    :param X: first set of samples.
    :param Y: second set of samples.
    :param kernels: list of kernels.
    """
    total = 0.0
    for kernel in kernels:
        total += squared_mmd(X, Y, kernel)
    return total / len(kernels)