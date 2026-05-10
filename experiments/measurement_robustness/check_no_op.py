"""Sanity check: explicit ``no_perturbation()`` and ``perturbation=None``
should give bit-identical gradients. If they don't, the no-op claim of
the Perturbation hooks is suspect.
"""
import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp

from grid_neurons.benchmarks.mnist import N_CHANNELS, NUM_CLASSES
from grid_neurons.block import init_block_params, zero_block_state, no_perturbation
from grid_neurons.topology import build_grid
from grid_neurons.training import local_grads


def main():
    DT = 1e-3
    topo = build_grid(N_CHANNELS, NUM_CLASSES)
    params = init_block_params(jax.random.PRNGKey(0), topology=topo,
                               tau_min=1e-3, tau_max=20e-3)
    s0 = zero_block_state(topo)

    # Synthetic single sequence (28 timesteps × 28 features), random target.
    seq = jax.random.uniform(jax.random.PRNGKey(1), (28, 28)) * 10.0
    tgt = 3
    key = jax.random.PRNGKey(123)

    L_a, g_a, _ = local_grads(params, s0, seq, tgt, topo, DT, "sum")
    L_b, g_b, _ = local_grads(params, s0, seq, tgt, topo, DT, "sum",
                              perturbation=no_perturbation(), key=key)

    def diff(label, a, b):
        d = float(jnp.max(jnp.abs(a - b)))
        n = float(jnp.max(jnp.abs(b)))
        print(f"  {label:12s}  max|d|={d:.3e}  max|b|={n:.3e}  rel={d/max(n,1e-30):.3e}")

    print(f"loss diff: {abs(float(L_a) - float(L_b)):.3e}")
    diff("w_left",  g_a.cells.w_left,  g_b.cells.w_left)
    diff("w_top",   g_a.cells.w_top,   g_b.cells.w_top)
    diff("bias",    g_a.cells.bias,    g_b.cells.bias)
    diff("log_tau", g_a.cells.log_tau, g_b.cells.log_tau)


if __name__ == "__main__":
    main()
