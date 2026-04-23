"""Row-wise sequential MNIST loader.

Each 28×28 image is streamed as a 28-step sequence of 28-dim row vectors.
Pixels are normalised to [0, input_gain]. 10 classes, standard MNIST
train/test split (60000 / 10000).
"""

import os

import jax
import jax.numpy as jnp
import numpy as np


T_STEPS = 28
N_CHANNELS = 28
NUM_CLASSES = 10


_MNIST_CACHE = None


def _fetch_mnist():
    global _MNIST_CACHE
    if _MNIST_CACHE is not None:
        return _MNIST_CACHE
    cache_path = os.path.expanduser("~/.cache/dendritic_mnist.npz")
    if os.path.exists(cache_path):
        blob = np.load(cache_path)
        _MNIST_CACHE = (blob["X"], blob["y"])
        return _MNIST_CACHE

    from sklearn.datasets import fetch_openml

    bundle = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = bundle.data.astype(np.float32)       # (70000, 784) in [0, 255]
    y = bundle.target.astype(np.int32)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, X=X, y=y)
    _MNIST_CACHE = (X, y)
    return _MNIST_CACHE


def load(input_gain: float = 3.0, train_size: int | None = None,
         val_size: int | None = None):
    """Returns (train_X, train_y, val_X, val_y) as jnp arrays.

    train_X shape (n_train, 28, 28) — 28 timesteps × 28 features per step.
    """
    X, y = _fetch_mnist()
    X = X.reshape(-1, T_STEPS, N_CHANNELS) / 255.0 * float(input_gain)
    # sklearn's default MNIST is last-10k-are-test
    train_X, val_X = X[:60000], X[60000:]
    train_y, val_y = y[:60000], y[60000:]
    if train_size is not None:
        train_X = train_X[:train_size]; train_y = train_y[:train_size]
    if val_size is not None:
        val_X = val_X[:val_size]; val_y = val_y[:val_size]
    return (
        jnp.asarray(train_X), jnp.asarray(train_y),
        jnp.asarray(val_X), jnp.asarray(val_y),
    )


def batch_iter(inputs, targets, batch_size: int, key):
    n = inputs.shape[0]
    perm = jax.random.permutation(key, n)
    inputs = inputs[perm]; targets = targets[perm]
    for i in range(0, n - batch_size + 1, batch_size):
        yield inputs[i : i + batch_size], targets[i : i + batch_size]
