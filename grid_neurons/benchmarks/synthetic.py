"""Synthetic temporal-pattern-detection dataset (rate-coded).

Each sample: a (T, N) continuous sequence in [0, input_gain]. Channel 0
carries a class-specific temporal profile — a Gaussian bump whose centre
depends on the class. All other channels are Gaussian background.

Rate-coded rather than single-spike: the filter-based dendritic branches
integrate continuous input naturally, and the task still requires learning
timescale-selective responses.
"""

import jax
import jax.numpy as jnp


def class_bump(T_steps: int, cls: int, num_classes: int, width: float = 4.0):
    """Gaussian bump centred at class-specific time. Values in [0, 1]."""
    t = jnp.arange(T_steps, dtype=jnp.float32)
    start = int(T_steps * 0.15)
    end = int(T_steps * 0.85)
    centre = start + (end - start) * (cls + 0.5) / num_classes
    bump = jnp.exp(-0.5 * ((t - centre) / width) ** 2)
    return bump


def generate_batch(
    key,
    batch_size: int,
    num_classes: int,
    T_steps: int,
    N_channels: int,
    input_gain: float = 5.0,
    noise_std: float = 0.2,
):
    """Return (inputs, targets).

    inputs:  (batch, T, N) float32
    targets: (batch,) int32
    """
    k_cls, k_noise = jax.random.split(key)
    targets = jax.random.randint(k_cls, (batch_size,), 0, num_classes)

    # background noise on all channels (positive half-normal-ish via |N(0, σ)|)
    noise = jnp.abs(jax.random.normal(k_noise, (batch_size, T_steps, N_channels))) * noise_std

    # for each sample, put its class bump into channel 0 (scaled by input_gain)
    def make_sample(noise_i, cls):
        bump = class_bump(T_steps, cls, num_classes)  # (T,)
        signal_channel = bump * input_gain            # (T,)
        # replace channel 0 with signal + noise
        combined = noise_i.at[:, 0].set(signal_channel + noise_i[:, 0])
        return combined

    inputs = jax.vmap(make_sample)(noise, targets)
    return inputs, targets
