"""
File contains utility functions used throughout the package
"""

from typing import Sequence, Union, List, Dict
import random
import orbax
from flax.training import orbax_utils
import jax.numpy as jnp
from jax import random
from jax.random import KeyArray
from flax.core import FrozenDict
import numpy as np
import torch
import torch.utils.data as data

from priorCVAE.models import Decoder


def numpy_collate(batch):
    """
    Used while creating a dataloader.

    Details: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide4/Research_Projects_with_JAX.html
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def create_data_loaders(*datasets: Sequence[data.Dataset], train: Union[bool, Sequence[bool]] = True,
                        batch_size: int = 128, num_workers: int = 4, seed: int = None) -> List[data.DataLoader]:
    """
    Creates data loaders used in JAX for a set of datasets.
    
    :param datasets: Datasets for which data loaders are created.
    :param train: Sequence indicating which datasets are used for training and which not. If single bool, the same value
                  is used for all datasets.
    :param batch_size: Batch size to use in the data loaders.
    :param num_workers: Number of workers for each dataset.
    :param seed: Seed to initialize the workers and shuffling with.

    Details: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide4/Research_Projects_with_JAX.html

    """

    if seed is None:
        seed = random.randint(0, 9999)

    loaders = []
    if not isinstance(train, (list, tuple)):
        train = [train for _ in datasets]
    for dataset, is_train in zip(datasets, train):
        loader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=is_train,
                                 drop_last=is_train,
                                 collate_fn=numpy_collate,
                                 num_workers=num_workers,
                                 persistent_workers=True,
                                 generator=torch.Generator().manual_seed(seed))
        loaders.append(loader)
    return loaders


def sq_euclidean_dist(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate Squared Euclidean distance between the two vectors, x and y.

    d(x, y) = sqrt(x**2 + y**2.T - 2 * <x, transpose(y)>)

    The implementation uses the broadcasting functionality of jax.numpy for mulit-dimensionality calculation.

    :param x: Jax ndarray of the shape, (N_1, D).
    :param y: Jax ndarray of the shape, (N_2, D).

    :returns: the Euclidean distance value.
    """
    assert isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray)
    if len(x.shape) == 1:
        x = x.reshape(x.shape[0], 1)
    if len(y.shape) == 1:
        y = y.reshape(y.shape[0], 1)

    assert x.shape[-1] == y.shape[-1]

    dist = jnp.sum(jnp.square(x), axis=-1)[..., None] + jnp.sum(jnp.square(y), axis=-1)[..., None].T - 2 * jnp.dot(x,
                                                                                                                   y.T)
    return dist


def save_model_params(ckpt_dir: str, params: FrozenDict):
    """Save the model parameters in the specified directory."""
    ckpt = {'params': params}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(ckpt_dir, ckpt, save_args=save_args)


def load_model_params(ckpt_dir: str) -> FrozenDict:
    """Load the model parameters"""
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored_params = orbax_checkpointer.restore(ckpt_dir)['params']
    return restored_params


def generate_decoder_samples(key: KeyArray, decoder_params: Dict, decoder: Decoder, num_samples: int, latent_dim: int, c: jnp.ndarray = None):
    """
    Generate samples from the decoder.
    
    :param key: Jax random key.
    :param decoder_params: decoder parameters.
    :param decoder: decoder model.
    :param num_samples: number of samples to generate.
    :param latent_dim: dimension of the latent space.
    :param c: a Jax ndarray used for cVAE of the shape, (N, C).

    :return: generated samples.
    """
    z = random.normal(key, (num_samples, latent_dim))
    z = jnp.array(z)

    if c is not None:
        z = jnp.concatenate([z, c], axis=-1)

    x = decode(decoder_params, decoder, z)
    return x


def decode(decoder_params: Dict, decoder: Decoder, z: jnp.ndarray):
    """Decode a latent vector z."""
    return decoder.apply({'params': decoder_params}, z)  