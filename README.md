# grid-neurons

A rectangular grid of first-order analog filters, trained end-to-end on
row-wise sequential MNIST by a local RTRL rule that requires no
backprop-through-time and no per-parameter optimiser state.

Companion to [tree-neurons](https://github.com/ThomasPluck/tree-neurons):
same Gm-C + tanh + bias cell primitive, same SnAP-k-family local rule,
different topology.

## The architecture

An N × M grid of identical cells. Each cell has two weighted inputs
(from its left and above neighbours) and one output, which is wired to
both its right and below neighbours. External input enters through the
left column; the bottom row is the block readout; top and right
shorelines are nulled.

```
j = 0    1    2  ...  M-1
i = 0   [*]  [*]  [*]  [*]    <- top row gets 0 from "above"
    1   [*]  [*]  [*]  [*]
  ...
  N-1   [*]  [*]  [*]  [*]    <- bottom row: M-dim readout

  col 0: N external input channels, one per row
  col M-1: right-going edges dropped
```

For row-wise MNIST the grid is **28 × 10**: 28 left-shoreline rows
(one per input-pixel column), 10 bottom-shoreline columns (one per
class). This makes encoder and decoder purely geometric — no learned
weights anywhere outside the grid itself.

Four learnable scalars per cell: `(log_tau, w_left, w_top, bias)`.
Total trainable parameters on the MNIST grid: **1,120**.

## Results

Full-scale row-wise MNIST, 60k training images, 10 epochs, local SnAP-1:

| setup | optimiser | batch | final val acc | wall time |
|---|---|---|---|---|
| SnAP-1, Adam 3e-3 | Adam | 32 | **0.522** | ~28 min |
| SnAP-1, per-group SGD (η₀=1e-2) | plain SGD, no adaptive state | **1** | 0.320 *(3 epochs only, still climbing)* | ~91 min |

Both random-chance is 0.10.

The SGD batch-1 run is the hardware-realistic configuration: zero
per-parameter optimiser state, one gradient step per training example,
same local rule.

## Reproducing

```bash
pip install -r requirements.txt

# Adam batch=32, ~28 min on CPU, yields the 0.522 number:
python -u scripts/mnist_rowwise.py --grad local --epochs 10 --train-size 60000

# Hardware-realistic SGD batch=1, ~30 min / epoch on CPU:
python -u scripts/mnist_sgd_bs1.py --epochs 3 --eta0 1e-2

# Single-step BPTT vs local gradient check (sanity):
python -u scripts/check_gradient_agreement.py
```

MNIST is fetched once via `sklearn.datasets.fetch_openml` and cached at
`~/.cache/dendritic_mnist.npz`.

## Package layout

```
grid_neurons/
  cell.py        # single-cell primitive: filter + tanh + bias + self-traces
  topology.py    # N x M grid, raster forward/backward orders, neighbour indexing
  block.py       # forward (two-pass: scan + SnAP-1 cross-trace update) and
                 # reverse-raster backward with past-only subtraction
  training.py    # local_grads (SnAP-1), bptt_grads (jax.grad reference),
                 # summed/per-t CE losses, readout helpers
  benchmarks/    # MNIST + synthetic-task loaders (shared with tree-neurons)

scripts/
  mnist_rowwise.py            # main entry: local or BPTT, Adam, batched
  mnist_sgd_bs1.py            # hardware-realistic: per-group SGD, batch=1
  check_gradient_agreement.py # single-step local vs BPTT at float-64

paper/
  main.tex / main.pdf         # 6-page paper, sister document to the tree paper
```

## Design notes

- **No learned encoder.** When `N_ext == N` (e.g. MNIST 28 input channels
  into a 28-row grid), the encoder is `jnp.eye(28)` — literal identity
  wiring. An optional sparse-random projection is available for
  `N_ext ≠ N`, also non-learned. `input_routing` gradient is
  explicitly zeroed in `block_backward`.
- **No learned decoder.** The readout is the bottom-row cells' outputs
  summed over time, fed directly to a standard softmax cross-entropy
  loss. No readout weights.
- **Local rule.** Per-cell state during training: 3 self-traces + 8
  descendant cross-traces = 11 scalars per cell on top of the 4
  parameters.
- **Optimiser.** The Adam result above is a reference; the
  hardware-realistic configuration uses plain SGD with per-group
  learning rates `(η_w, η_b = η_w/20, η_τ = 4η_w)` and batch size 1.
  Zero per-parameter optimiser state.

## Status

Code runs; the 0.522 number reproduces. Paper is draft-quality and
will benefit from a careful human pass before wider circulation.
