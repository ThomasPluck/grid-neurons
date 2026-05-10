"""Run a single unperturbed baseline condition to gauge timing and the
final val accuracy under the chosen ``--train-size``/``--epochs`` config.

Used to decide what scale gives a clean enough baseline trend before
launching the full sweep.
"""
import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, THIS_DIR)

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax

from grid_neurons.benchmarks.mnist import load, N_CHANNELS, NUM_CLASSES
from grid_neurons.block import init_block_params
from grid_neurons.topology import build_grid

from sweeps import sweep_1a_leak
from run_sweep import make_train_step, make_eval_step, run_one_condition


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--train-size", type=int, default=6000)
    ap.add_argument("--val-size", type=int, default=2000)
    args = ap.parse_args()

    train_X, train_y, val_X, val_y = load(input_gain=10.0,
                                          train_size=args.train_size)
    topo = build_grid(N_CHANNELS, NUM_CLASSES)
    params = init_block_params(jax.random.PRNGKey(0), topology=topo,
                               tau_min=1e-3, tau_max=20e-3)
    opt = optax.adam(3e-3)
    opt_state = opt.init(params)

    train_step = make_train_step(topo, opt)
    eval_batch = make_eval_step(topo)

    cond = sweep_1a_leak()[0]   # delta_leak=0 baseline
    t0 = time.time()
    rows = run_one_condition(
        cond, train_X, train_y, val_X, val_y,
        train_step, eval_batch, params, opt, opt_state,
        epochs=args.epochs, batch_size=32, val_size=args.val_size,
        perm_key=jax.random.PRNGKey(42),
        perturbation_key=jax.random.PRNGKey(7),
        log_prefix="[baseline]",
    )
    print(f"\nbaseline final val_acc = {rows[-1]['val_acc']:.4f}",
          f"  total wall = {time.time()-t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
