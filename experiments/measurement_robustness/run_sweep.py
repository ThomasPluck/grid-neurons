"""Run a single sweep (or `all`) of perturbation conditions on row-wise MNIST.

The training loop is the same as scripts/mnist_rowwise.py: 28x10 grid,
identity encoder, summed-readout softmax-CE, Adam eta=3e-3, batch=32,
tau in [1, 20] ms, input gain 10. The only deviation is that the SnAP-1
local rule is augmented with a per-condition Perturbation specifying
which measurement-noise / -bias to inject.

For each condition we re-init the parameters and the optimiser to the
same starting state (single seed) so that all conditions in a sweep
share an identical baseline; the only change between runs is the
perturbation.

Output: one CSV per sweep at ``results/<sweep>.csv`` plus a tiny JSON
sidecar with the sweep configuration. CSV columns:

    experiment, perturbation_param, value, seed, epoch,
    train_loss, train_acc, val_acc, walltime
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, THIS_DIR)

import jax
import jax.numpy as jnp
import numpy as np
import optax

from grid_neurons.benchmarks.mnist import load, N_CHANNELS, NUM_CLASSES
from grid_neurons.block import (
    Perturbation, init_block_params, zero_block_state, block_forward,
    no_perturbation,
)
from grid_neurons.topology import build_grid
from grid_neurons.training import local_grads, readout_from_rates

from sweeps import ALL_SWEEPS, Condition, SWEEP_ORDER


DT = 1e-3
N, M = N_CHANNELS, NUM_CLASSES   # 28 x 10


def make_train_step(topo, opt):
    @jax.jit
    def per(p, perturbation, key, seq, tgt):
        s0 = zero_block_state(topo)
        loss, g, readout = local_grads(
            p, s0, seq, tgt, topo, DT, "sum",
            perturbation=perturbation, key=key,
        )
        acc = (jnp.argmax(readout) == tgt).astype(jnp.float32)
        return loss, g, acc

    @jax.jit
    def train_step(p, opt_state, perturbation, batch_key, seqs, tgts):
        bs = seqs.shape[0]
        keys = jax.random.split(batch_key, bs)
        losses, grads, accs = jax.vmap(
            per, in_axes=(None, None, 0, 0, 0)
        )(p, perturbation, keys, seqs, tgts)
        mg = jax.tree_util.tree_map(lambda x: x.mean(0), grads)
        u, new_opt_state = opt.update(mg, opt_state, p)
        return optax.apply_updates(p, u), new_opt_state, losses.mean(), accs.mean()

    return train_step


def make_eval_step(topo):
    @jax.jit
    def eval_batch(p, seqs, tgts):
        # Forward dynamics are unperturbed -- we evaluate the trained network
        # without measurement noise (the noise was only in the *learning* path).
        def one(seq, tgt):
            s0 = zero_block_state(topo)
            def step(st, x):
                ns, r, _, _ = block_forward(p, st, x, topo, DT)
                return ns, r
            _, rates = jax.lax.scan(step, s0, seq)
            return (jnp.argmax(readout_from_rates(rates)) == tgt).astype(jnp.float32)
        return jax.vmap(one)(seqs, tgts).mean()
    return eval_batch


def run_one_condition(
    cond: Condition, train_X, train_y, val_X, val_y,
    train_step, eval_batch, init_params, opt, opt_state_init,
    *, epochs: int, batch_size: int, val_size: int, perm_key,
    perturbation_key, log_prefix: str,
) -> List[dict]:
    """Run one condition for `epochs` epochs and return a list of per-epoch
    metric dicts.

    init_params and opt_state_init are reused across conditions to fix the
    starting point. The shuffle permutation key is the same across conditions
    so each condition sees data in the same order.
    """
    params = init_params
    opt_state = opt_state_init

    rows = []
    t0 = time.time()
    pk = perm_key
    nk = perturbation_key
    for ep in range(epochs):
        pk, kperm = jax.random.split(pk)
        perm = jax.random.permutation(kperm, train_X.shape[0])
        Xs = train_X[perm]; ys = train_y[perm]

        losses_ep, accs_ep = [], []
        n_batches = (Xs.shape[0] // batch_size) * batch_size
        for i in range(0, n_batches, batch_size):
            nk, ksub = jax.random.split(nk)
            params, opt_state, L, A = train_step(
                params, opt_state, cond.perturbation, ksub,
                Xs[i:i + batch_size], ys[i:i + batch_size],
            )
            losses_ep.append(float(L))
            accs_ep.append(float(A))

        v_idx = jax.random.permutation(jax.random.PRNGKey(ep + 999),
                                        val_X.shape[0])[:val_size]
        vs = val_X[v_idx]; vt = val_y[v_idx]
        chunk = max(min(200, val_size), 1)
        val_acc = float(np.mean([
            float(eval_batch(params, vs[j:j + chunk], vt[j:j + chunk]))
            for j in range(0, val_size, chunk)
        ]))
        ml = float(np.mean(losses_ep))
        mta = float(np.mean(accs_ep))
        wall = time.time() - t0
        rows.append(dict(
            experiment=cond.sweep,
            perturbation_param=cond.param,
            value=float(cond.value),
            seed=0,
            epoch=ep,
            train_loss=ml,
            train_acc=mta,
            val_acc=val_acc,
            walltime=wall,
        ))
        print(f"  {log_prefix} ep {ep} loss={ml:.3f} train_acc={mta:.3f} "
              f"val={val_acc:.3f} t={wall:.1f}s", flush=True)
    return rows


def run_sweep(
    sweep_name: str, *,
    epochs: int, train_size: int, val_size: int, batch_size: int,
    init_seed: int, perm_seed: int, perturbation_seed: int,
    out_dir: str,
):
    print(f"\n=== Sweep: {sweep_name} ===", flush=True)
    print(f"epochs={epochs} train_size={train_size} val_size={val_size} "
          f"batch_size={batch_size} init_seed={init_seed}", flush=True)

    sweep = ALL_SWEEPS[sweep_name]()

    train_X, train_y, val_X, val_y = load(input_gain=10.0,
                                          train_size=train_size)
    print(f"loaded MNIST: train={train_X.shape}, val={val_X.shape}", flush=True)

    topo = build_grid(N, M)
    init_params = init_block_params(
        jax.random.PRNGKey(init_seed), topology=topo,
        tau_min=1e-3, tau_max=20e-3,
    )
    opt = optax.adam(3e-3)
    opt_state_init = opt.init(init_params)

    train_step = make_train_step(topo, opt)
    eval_batch = make_eval_step(topo)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{sweep_name}.csv")
    side_path = os.path.join(out_dir, f"{sweep_name}.json")
    cfg = dict(
        sweep=sweep_name, epochs=epochs, train_size=train_size,
        val_size=val_size, batch_size=batch_size, init_seed=init_seed,
        perm_seed=perm_seed, perturbation_seed=perturbation_seed,
        n_conditions=len(sweep),
    )
    with open(side_path, "w") as f:
        json.dump(cfg, f, indent=2)

    fields = ["experiment", "perturbation_param", "value", "seed", "epoch",
              "train_loss", "train_acc", "val_acc", "walltime"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

    perm_key = jax.random.PRNGKey(perm_seed)
    pert_key = jax.random.PRNGKey(perturbation_seed)
    sweep_t0 = time.time()
    for ci, cond in enumerate(sweep):
        log_prefix = f"[{sweep_name}][{ci+1}/{len(sweep)} {cond.param}={cond.value:+.3f}]"
        print(f"\n{log_prefix}", flush=True)
        rows = run_one_condition(
            cond, train_X, train_y, val_X, val_y,
            train_step, eval_batch, init_params, opt, opt_state_init,
            epochs=epochs, batch_size=batch_size, val_size=val_size,
            perm_key=perm_key, perturbation_key=pert_key,
            log_prefix=log_prefix,
        )
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            for r in rows:
                w.writerow(r)
        print(f"  -> appended {len(rows)} rows; sweep elapsed "
              f"{time.time() - sweep_t0:.0f}s", flush=True)
    print(f"\n=== Sweep {sweep_name} done in {time.time() - sweep_t0:.0f}s ===",
          flush=True)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="all",
                    choices=["all"] + list(ALL_SWEEPS.keys()))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--train-size", type=int, default=6000)
    ap.add_argument("--val-size", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--perm-seed", type=int, default=42)
    ap.add_argument("--perturbation-seed", type=int, default=7)
    ap.add_argument("--out-dir", default=os.path.join(THIS_DIR, "results"))
    args = ap.parse_args()

    sweeps_to_run = SWEEP_ORDER if args.sweep == "all" else [args.sweep]

    for sname in sweeps_to_run:
        run_sweep(
            sname, epochs=args.epochs, train_size=args.train_size,
            val_size=args.val_size, batch_size=args.batch_size,
            init_seed=args.init_seed, perm_seed=args.perm_seed,
            perturbation_seed=args.perturbation_seed,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
