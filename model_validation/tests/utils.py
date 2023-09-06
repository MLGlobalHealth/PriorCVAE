import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np
import jax.numpy as jnp
import scipy.stats as stats
from typing import Tuple
import wandb


def plot_realizations(grid: jnp.ndarray, realizations: jnp.ndarray, title='', figsize=(4,3)):
    """Plot realizations sampled from a VAE."""

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(grid, realizations[:15].T)
    ax.plot(grid, realizations.mean(axis=0), c='black')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y=f_{VAE}(x)$')
    ax.set_title(title)
    return fig, ax


def plot_heatmap(matrix: jnp.ndarray, title: str ='', figsize: Tuple[float, float]=(8,6)):
    """Plot a heatmap of the input matrix."""

    assert len(matrix.shape) == 2
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    sns.heatmap(matrix, annot=False, fmt='g', cmap='coolwarm', ax=ax)
    ax.set_title(title)
    return fig, ax


def mean_bootstrap_interval(
    samples: jnp.ndarray, confidence_level: float=0.95, axis=0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the confidence interval for the mean of the samples using bootstrap resampling.

    :param samples: input array of samples.
    :param confidence_level: confidence level for the interval.
    :param axis: Axis along which to compute mean.

    :return: lower and upper confidence interval bounds for the mean of the data.
    """
    sample = (samples,)
    res = stats.bootstrap(
        sample,
        np.mean,
        axis=axis,
        vectorized=True,
        confidence_level=confidence_level,
        method="percentile",
        n_resamples=1000,
    )
    ci_lower, ci_upper = res.confidence_interval

    return ci_lower, ci_upper


def wandb_log_figure(fig: Figure, name: str):
    """
    Log a matplotlib figure to wandb.

    Image metadata is removed from the wandb summary since it does not
    need to be shown in the summary table.
    """
    if wandb.run is not None:
        wandb.log({name: wandb.Image(fig)})

        # remove media metadata from wandb summary
        del wandb.run.summary[name]