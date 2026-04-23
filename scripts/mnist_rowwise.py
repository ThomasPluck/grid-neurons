"""Row-wise MNIST on the grid: N=28 left-shoreline inputs, M=10 bottom
output cells. No learned encoder or decoder. Summed-readout softmax-CE.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grad", choices=["bptt", "local"], default="local")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--train-size", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--N", type=int, default=28)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--gain", type=float, default=10.0)
    ap.add_argument("--tau-min", type=float, default=1e-3)
    ap.add_argument("--tau-max", type=float, default=20e-3)
    args = ap.parse_args()

    import jax.numpy as jnp
    import numpy as np
    import optax

    from grid_neurons.benchmarks.mnist import load, N_CHANNELS, NUM_CLASSES
    from grid_neurons.topology import build_grid
    from grid_neurons.block import init_block_params, zero_block_state, block_forward
    from grid_neurons.training import bptt_grads, local_grads, readout_from_rates

    DT = 1e-3
    assert args.N == N_CHANNELS, f"grid N ({args.N}) must equal input dim ({N_CHANNELS})"
    assert args.M == NUM_CLASSES, f"grid M ({args.M}) must equal num classes ({NUM_CLASSES})"
    topo = build_grid(args.N, args.M)
    print(f"Grid {args.N}x{args.M}  ({args.N*args.M} cells)")

    train_X, train_y, val_X, val_y = load(input_gain=args.gain,
                                          train_size=args.train_size)
    params = init_block_params(jax.random.PRNGKey(0), topology=topo,
                               tau_min=args.tau_min, tau_max=args.tau_max)
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    grad_fn = bptt_grads if args.grad == "bptt" else local_grads

    def per(p, seq, tgt):
        s0 = zero_block_state(topo)
        loss, g, readout = grad_fn(p, s0, seq, tgt, topo, DT, "sum")
        acc = (jnp.argmax(readout) == tgt).astype(jnp.float32)
        return loss, g, acc

    @jax.jit
    def train_step(p, os, ss, ts):
        losses, grads, accs = jax.vmap(per, in_axes=(None, 0, 0))(p, ss, ts)
        mg = jax.tree_util.tree_map(lambda x: x.mean(0), grads)
        u, new_os = opt.update(mg, os, p)
        return optax.apply_updates(p, u), new_os, losses.mean(), accs.mean()

    @jax.jit
    def eval_batch(p, seqs, tgts):
        def one(seq, tgt):
            s0 = zero_block_state(topo)
            def step(st, x):
                ns, r, _, _ = block_forward(p, st, x, topo, DT)
                return ns, r
            _, rates = jax.lax.scan(step, s0, seq)
            return (jnp.argmax(readout_from_rates(rates)) == tgt).astype(jnp.float32)
        return jax.vmap(one)(seqs, tgts).mean()

    bs = 32
    dk = jax.random.PRNGKey(42)
    history = []
    t0 = time.time()
    for ep in range(args.epochs):
        losses_ep, accs_ep = [], []
        dk, kperm = jax.random.split(dk)
        perm = jax.random.permutation(kperm, train_X.shape[0])
        X_sh = train_X[perm]; y_sh = train_y[perm]
        for i in range(0, X_sh.shape[0] - bs + 1, bs):
            params, opt_state, L, acc = train_step(params, opt_state,
                                                   X_sh[i:i+bs], y_sh[i:i+bs])
            losses_ep.append(float(L)); accs_ep.append(float(acc))
        v_idx = jax.random.permutation(jax.random.PRNGKey(ep + 999),
                                       val_X.shape[0])[:2000]
        vs = val_X[v_idx]; vt = val_y[v_idx]
        chunks = [(vs[j:j+200], vt[j:j+200]) for j in range(0, 2000, 200)]
        val_acc = float(np.mean([float(eval_batch(params, s, t)) for s, t in chunks]))
        ml = float(np.mean(losses_ep)); mta = float(np.mean(accs_ep))
        history.append((ep, ml, mta, val_acc, time.time() - t0))
        print(f"[{args.grad}] ep {ep:02d}  loss={ml:.3f}  train_acc={mta:.3f}  "
              f"val={val_acc:.3f}  t={time.time()-t0:.0f}s")

    os.makedirs("results", exist_ok=True)
    with open(f"results/mnist_{args.grad}_grid{args.N}x{args.M}.txt", "w") as f:
        f.write(f"# {args.grad} grid={args.N}x{args.M}\n")
        f.write("# ep  train_loss  train_acc  val_acc  time\n")
        for h in history:
            f.write(f"{h[0]} {h[1]:.4f} {h[2]:.4f} {h[3]:.4f} {h[4]:.1f}\n")


if __name__ == "__main__":
    main()
