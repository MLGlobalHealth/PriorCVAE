"""
File contains functions for saving visualizations to wandb.
"""

import numpy as np
import jax.numpy as jnp

from priorCVAE.priors import Kernel
from priorCVAE.diagnostics import sample_covariance
from .utils import plot_realizations, plot_heatmap, wandb_log_figure


def plot_vae_realizations(samples: jnp.ndarray, grid: jnp.ndarray, **kwargs):
    """Plot realizations sampled from from a VAE."""

    fig, _ = plot_realizations(grid, samples, "VAE samples")
    wandb_log_figure(fig, "vae_realizations")


def plot_covariance(samples: jnp.ndarray, **kwargs):
    """Plot empirical covariance matrix."""

    cov_matrix = sample_covariance(samples)
    fig, ax = plot_heatmap(cov_matrix, "Empirical covariance")
    wandb_log_figure(fig, "covariance")


def plot_correlation(samples: jnp.ndarray, **kwargs):
    """Plot correlation matrix."""

    corr = np.corrcoef(np.transpose(samples))
    fig, ax = plot_heatmap(corr, "Correlation")
    wandb_log_figure(fig, "correlation")


def plot_kernel(kernel: Kernel, kernel_name: str, grid: jnp.ndarray, **kwargs):
    """Plot a kernel's covariance matrix."""

    K = kernel(grid, grid)
    fig, ax = plot_heatmap(K, kernel_name)
    wandb_log_figure(fig, "kernel")
