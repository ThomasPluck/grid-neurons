# grid-neurons

Grid-lattice variant of the
[tree-neurons](https://github.com/ThomasPluck/tree-neurons) architecture:
same Gm-C + tanh + bias primitive, but arranged as a 2-in/2-out
rectangular grid with native shoreline I/O rather than a tree.

- Each cell has two weighted inputs (from-left and from-above) and its
  output is routed to two neighbours (to-right and to-below).
- Left shoreline (`j=0`) is the external input; bottom shoreline
  (`i=N-1`) is the block output. Top and right shorelines are nulled.
- Four learnable scalars per cell: `(log_tau, w_left, w_top, bias)`.
- Encoder = identity by shoreline geometry (or an optional fixed
  sparse-random projection); decoder = summed-rate softmax-CE. No
  learned encoder/decoder parameters.

Local rule is SnAP-1 with **two** cross-traces per cell (one per
immediate descendant), plus the three self-traces. Single-step
gradients agree with BPTT at float-64 eps; long-sequence agreement
diverges in T because each cell has two descendant paths whose
multi-hop interactions SnAP-1 truncates.

```
grid_neurons/
  cell.py        # single-cell primitive + self-traces
  topology.py    # N x M grid, raster forward/backward order
  block.py      # forward (two-pass) + backward with SnAP-1 cross-traces
  training.py    # local_grads, bptt_grads, readout helpers
  benchmarks/    # synthetic + MNIST loaders (copied from tree-neurons)
scripts/
  check_gradient_agreement.py   # T in {1, 5, 20, 100} vs BPTT at f64
  mnist_rowwise.py              # grid trained end-to-end on row-wise MNIST
paper/
  main.tex                      # paper skeleton specialising the tree paper
```

Numerical-agreement snapshot (4x4 grid, f64, random seed):

```
T   = 1    w_L rel 1.6e-16   log_tau rel 4.3e-16   (machine precision)
T   = 5    w_L rel 6.1e-03                       1.5e-02
T   = 20                     5.8e-02                       5.3e-02
T   = 100                    1.2e-01                       2.1e-01
```

Smoke test on row-wise MNIST (grid 28x10, local SnAP-1, Adam 3e-3,
2000-train / 2 epochs): val acc 0.126 -> 0.133 (slow: SnAP-1 is a
significantly harder approximation on this topology than on the tree).
