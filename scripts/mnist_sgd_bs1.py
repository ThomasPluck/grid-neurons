"""Hardware-realistic MNIST training on the grid:
  - local SnAP-1 rule (same cached eligibility traces, gradient summed
    over timesteps within a sequence)
  - batch size 1 (one training example = one parameter update)
  - per-group plain SGD (no Adam, no per-parameter state)
      lr_w_left = lr_w_top = eta0
      lr_bias    = eta0 / 20
      lr_log_tau = 4 * eta0

This is the configuration a chip would actually implement: each cell holds
only its local state + eligibility traces, the optimiser carries zero
per-parameter state, and gradients are applied as each training example
finishes. No variance-reduction via batching.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-size", type=int, default=60000)
    ap.add_argument("--eta0", type=float, default=1e-2)
    ap.add_argument("--gain", type=float, default=10.0)
    ap.add_argument("--tau-min", type=float, default=1e-3)
    ap.add_argument("--tau-max", type=float, default=20e-3)
    args = ap.parse_args()

    import jax.numpy as jnp
    import numpy as np

    from grid_neurons.benchmarks.mnist import load, N_CHANNELS, NUM_CLASSES
    from grid_neurons.topology import build_grid
    from grid_neurons.block import (
        init_block_params, zero_block_state, block_forward,
        BlockParams,
    )
    from grid_neurons.cell import CellParams
    from grid_neurons.training import local_grads, readout_from_rates

    DT = 1e-3
    topo = build_grid(N_CHANNELS, NUM_CLASSES)
    print(f"Grid {N_CHANNELS}x{NUM_CLASSES}, batch=1, per-group SGD, eta0={args.eta0}")

    train_X, train_y, val_X, val_y = load(input_gain=args.gain,
                                          train_size=args.train_size)
    params = init_block_params(jax.random.PRNGKey(0), topology=topo,
                               tau_min=args.tau_min, tau_max=args.tau_max)

    def per_group_sgd_update(p, g, eta0):
        """Plain SGD with per-group LRs.  Zero optimiser state."""
        lr_w = eta0
        lr_b = eta0 / 20.0
        lr_tau = 4.0 * eta0
        new_cells = CellParams(
            log_tau=p.cells.log_tau - lr_tau * g.cells.log_tau,
            w_left=p.cells.w_left - lr_w * g.cells.w_left,
            w_top=p.cells.w_top  - lr_w * g.cells.w_top,
            bias=p.cells.bias    - lr_b * g.cells.bias,
        )
        return BlockParams(cells=new_cells,
                           input_routing=p.input_routing)

    @jax.jit
    def step_one_example(p, seq, tgt):
        s0 = zero_block_state(topo)
        loss, g, readout = local_grads(p, s0, seq, tgt, topo, DT, "sum")
        acc = (jnp.argmax(readout) == tgt).astype(jnp.float32)
        new_p = per_group_sgd_update(p, g, args.eta0)
        return new_p, loss, acc

    @jax.jit
    def eval_one(p, seq, tgt):
        s0 = zero_block_state(topo)
        def step(st, x):
            ns, r, _, _ = block_forward(p, st, x, topo, DT)
            return ns, r
        _, rates = jax.lax.scan(step, s0, seq)
        return (jnp.argmax(readout_from_rates(rates)) == tgt).astype(jnp.float32)

    dk = jax.random.PRNGKey(42)
    t0 = time.time()
    hist = []
    for ep in range(args.epochs):
        dk, kperm = jax.random.split(dk)
        perm = jax.random.permutation(kperm, train_X.shape[0])
        X_sh = train_X[perm]; y_sh = train_y[perm]
        losses_ep, accs_ep = [], []
        for i in range(X_sh.shape[0]):
            params, L, acc = step_one_example(params, X_sh[i], int(y_sh[i]))
            losses_ep.append(float(L))
            accs_ep.append(float(acc))
            if (i + 1) % 2000 == 0:
                recent = accs_ep[-2000:]
                print(f"  ep {ep:02d}  sample {i+1}/{X_sh.shape[0]}  "
                      f"recent_train_acc={np.mean(recent):.3f}  "
                      f"t={time.time()-t0:.0f}s")
        # Val on 2000 random val samples
        v_idx = jax.random.permutation(jax.random.PRNGKey(ep + 999),
                                       val_X.shape[0])[:2000]
        vs = val_X[v_idx]; vt = val_y[v_idx]
        val_accs = [float(eval_one(params, vs[j], int(vt[j]))) for j in range(2000)]
        v = float(np.mean(val_accs))
        ml = float(np.mean(losses_ep)); mta = float(np.mean(accs_ep))
        hist.append((ep, ml, mta, v, time.time() - t0))
        print(f"[sgd-bs1] ep {ep:02d}  loss={ml:.3f}  train_acc={mta:.3f}  "
              f"val={v:.3f}  t={time.time()-t0:.0f}s")

    os.makedirs("results", exist_ok=True)
    with open("results/mnist_sgd_bs1.txt", "w") as f:
        f.write(f"# SGD per-group, batch=1, grid=28x10, eta0={args.eta0}\n")
        f.write("# ep  train_loss  train_acc  val_acc  time\n")
        for h in hist:
            f.write(f"{h[0]} {h[1]:.4f} {h[2]:.4f} {h[3]:.4f} {h[4]:.1f}\n")


if __name__ == "__main__":
    main()
