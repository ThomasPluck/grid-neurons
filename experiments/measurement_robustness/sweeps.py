"""Sweep definitions for the measurement-noise robustness study.

Each sweep is a list of ``Condition`` records. A condition fully specifies
how to construct the ``Perturbation`` object passed into the SnAP-1 rule
for that run.

Layout:
  - 1a leak              : delta_leak ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2}
  - 1b gain symmetric    : msg_gain_pos = msg_gain_neg = 1-δ, δ ∈ {0,0.02,0.05,0.1,0.2}
  - 1b gain asym A       : (gp, gn) = (1-δ, 1+δ)        (shrink positive,  grow negative)
  - 1b gain asym B       : (gp, gn) = (1+δ, 1-δ)        (grow positive,    shrink negative)
  - 1c subtraction       : delta_sub  ∈ {-0.1,-0.05,-0.02,0,0.02,0.05,0.1}
  - 2a eps noise         : sigma_eps_rel ∈ {0,0.01,0.05,0.1,0.2,0.5,1.0}
  - 2b msg noise         : sigma_msg_rel ∈ same
  - 2c combined noise    : both at matched sigma
"""
from typing import Callable, List, NamedTuple

import jax.numpy as jnp

from grid_neurons.block import Perturbation, no_perturbation


class Condition(NamedTuple):
    sweep: str             # e.g. "1c_subtraction"
    param: str             # the perturbation knob name
    value: float           # the swept value
    perturbation: Perturbation


def _f(x):
    return jnp.asarray(x, dtype=jnp.float32)


def _leak(delta):
    p = no_perturbation()
    return p._replace(delta_leak=_f(delta))


def _gain(gp, gn):
    p = no_perturbation()
    return p._replace(msg_gain_pos=_f(gp), msg_gain_neg=_f(gn))


def _sub(delta):
    p = no_perturbation()
    return p._replace(delta_sub=_f(delta))


def _eps_noise(sigma):
    p = no_perturbation()
    return p._replace(sigma_eps_rel=_f(sigma))


def _msg_noise(sigma):
    p = no_perturbation()
    return p._replace(sigma_msg_rel=_f(sigma))


def _both_noise(sigma):
    p = no_perturbation()
    return p._replace(sigma_eps_rel=_f(sigma), sigma_msg_rel=_f(sigma))


def sweep_1a_leak() -> List[Condition]:
    return [
        Condition("1a_leak", "delta_leak", d, _leak(d))
        for d in (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2)
    ]


def sweep_1b_gain_sym() -> List[Condition]:
    return [
        Condition("1b_gain_sym", "delta", d, _gain(1.0 - d, 1.0 - d))
        for d in (0.0, 0.02, 0.05, 0.1, 0.2)
    ]


def sweep_1b_gain_asym_A() -> List[Condition]:
    # shrink positive, grow negative
    return [
        Condition("1b_gain_asym_A", "delta", d, _gain(1.0 - d, 1.0 + d))
        for d in (0.0, 0.02, 0.05, 0.1, 0.2)
    ]


def sweep_1b_gain_asym_B() -> List[Condition]:
    # grow positive, shrink negative
    return [
        Condition("1b_gain_asym_B", "delta", d, _gain(1.0 + d, 1.0 - d))
        for d in (0.0, 0.02, 0.05, 0.1, 0.2)
    ]


def sweep_1c_subtraction() -> List[Condition]:
    return [
        Condition("1c_subtraction", "delta_sub", d, _sub(d))
        for d in (-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1)
    ]


def sweep_2a_eps_noise() -> List[Condition]:
    return [
        Condition("2a_eps_noise", "sigma_rel", s, _eps_noise(s))
        for s in (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
    ]


def sweep_2b_msg_noise() -> List[Condition]:
    return [
        Condition("2b_msg_noise", "sigma_rel", s, _msg_noise(s))
        for s in (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
    ]


def sweep_2c_combined_noise() -> List[Condition]:
    return [
        Condition("2c_combined_noise", "sigma_rel", s, _both_noise(s))
        for s in (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
    ]


ALL_SWEEPS: dict[str, Callable[[], List[Condition]]] = {
    "1a_leak":            sweep_1a_leak,
    "1b_gain_sym":        sweep_1b_gain_sym,
    "1b_gain_asym_A":     sweep_1b_gain_asym_A,
    "1b_gain_asym_B":     sweep_1b_gain_asym_B,
    "1c_subtraction":     sweep_1c_subtraction,
    "2a_eps_noise":       sweep_2a_eps_noise,
    "2b_msg_noise":       sweep_2b_msg_noise,
    "2c_combined_noise":  sweep_2c_combined_noise,
}


# Priority order: bias first (the more important per the spec), starting with
# 1c which is the most likely to show interesting structure.
SWEEP_ORDER = [
    "1c_subtraction",
    "1a_leak",
    "1b_gain_sym",
    "1b_gain_asym_A",
    "1b_gain_asym_B",
    "2a_eps_noise",
    "2b_msg_noise",
    "2c_combined_noise",
]
