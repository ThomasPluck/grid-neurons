"""Single-step gradient-agreement check: local SnAP-1 vs BPTT at init,
on a small 4x4 grid with a short synthetic sequence. Uses float-64 for
a precision-floor read.
"""
import os
import sys
# Enable running script-style without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax


def main():
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from grid_neurons.topology import build_grid
    from grid_neurons.block import init_block_params, zero_block_state
    from grid_neurons.training import bptt_grads, local_grads

    DT = 1e-3
    N, M = 4, 4                 # 4x4 grid (N input rows x M output cols)
    T = int(os.environ.get("T", "1"))  # use env to sweep; default 1 for clean check
    topo = build_grid(N, M)
    key = jax.random.PRNGKey(0)
    params = init_block_params(key, topology=topo)
    # cast to f64
    params = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if isinstance(x, jnp.ndarray)
                  and jnp.issubdtype(x.dtype, jnp.floating) else x, params,
    )

    # Build a random sequence and a random target.
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    seq = jax.random.normal(k1, (T, N), dtype=jnp.float64) * 0.5
    tgt = int(jax.random.randint(k2, (), 0, M))
    s0 = zero_block_state(topo)

    for loss_mode in ("per_t", "sum"):
        _, g_b, _ = bptt_grads(params, s0, seq, tgt, topo, DT, loss_mode)
        _, g_l, _ = local_grads(params, s0, seq, tgt, topo, DT, loss_mode)

        def grp(label, a, b):
            mx = float(jnp.max(jnp.abs(a - b)))
            ma = float(jnp.max(jnp.abs(b)))
            rel = mx / max(ma, 1e-30)
            print(f"  {label:16s}  max|d|={mx:.3e}  max|bptt|={ma:.3e}  rel={rel:.3e}")

        print(f"\n=== loss_mode={loss_mode}, T={T}, grid={N}x{M} ===")
        grp("w_left",    g_l.cells.w_left,  g_b.cells.w_left)
        grp("w_top",     g_l.cells.w_top,   g_b.cells.w_top)
        grp("bias",      g_l.cells.bias,    g_b.cells.bias)
        grp("log_tau",   g_l.cells.log_tau, g_b.cells.log_tau)


if __name__ == "__main__":
    main()
