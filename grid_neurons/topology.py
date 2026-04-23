"""Rectangular N x M grid topology.

Cells are indexed (i, j) with i in [0, N), j in [0, M).
Connectivity: each cell (i, j) receives two inputs:
  - from-left  : y at (i, j-1), or external_input[i] if j == 0
  - from-above : y at (i-1, j), or 0 if i == 0 (top shoreline null)
and drives two outputs:
  - to-right   : y_{i, j}  -> input of (i, j+1); discarded if j == M-1
  - to-below   : y_{i, j}  -> input of (i+1, j); becomes block output if i == N-1

Shorelines:
  - LEFT  (j = 0):   external input, one channel per row  -> block has N input channels
  - BOTTOM (i = N-1): block output, one channel per column -> block has M output channels
  - TOP    (i = 0):   zero (null)
  - RIGHT  (j = M-1): discarded

Forward scan order is raster (i ascending, j ascending within each row). This
is a valid topological order because every cell's inputs come from strictly
earlier cells in this sweep. Backward order is the reverse.
"""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class GridTopology(NamedTuple):
    N: int                         # grid rows (= input dim, left shoreline length)
    M: int                         # grid cols (= output dim, bottom shoreline length)
    forward_order: jnp.ndarray     # (N*M, 2) flat (i,j) sweep, raster
    backward_order: jnp.ndarray    # reverse of above
    # For SnAP-1 convenience: for each cell, global index of its descendant-below
    # and descendant-right (-1 if that edge is a nulled shoreline, i.e. the cell
    # is on the bottom row or right column).
    desc_below: jnp.ndarray        # (N*M,)
    desc_right: jnp.ndarray        # (N*M,)
    # For each cell, global index of its predecessors (from-above and from-left);
    # -1 when the corresponding shoreline is input (left) or null (top).
    pred_top:  jnp.ndarray         # (N*M,)
    pred_left: jnp.ndarray         # (N*M,)
    # Leaf / output helpers:
    left_row_cells:  jnp.ndarray   # (N,) cells on left shoreline (j=0)
    bottom_col_cells: jnp.ndarray  # (M,) cells on bottom shoreline (i=N-1)


def build_grid(N: int, M: int) -> GridTopology:
    idx = lambda i, j: i * M + j

    order = np.array([[i, j] for i in range(N) for j in range(M)], dtype=np.int32)

    n_cells = N * M
    desc_below = np.full(n_cells, -1, dtype=np.int32)
    desc_right = np.full(n_cells, -1, dtype=np.int32)
    pred_top = np.full(n_cells, -1, dtype=np.int32)
    pred_left = np.full(n_cells, -1, dtype=np.int32)
    for i in range(N):
        for j in range(M):
            k = idx(i, j)
            if i + 1 < N:
                desc_below[k] = idx(i + 1, j)
            if j + 1 < M:
                desc_right[k] = idx(i, j + 1)
            if i - 1 >= 0:
                pred_top[k] = idx(i - 1, j)
            if j - 1 >= 0:
                pred_left[k] = idx(i, j - 1)

    left_row = np.array([idx(i, 0) for i in range(N)], dtype=np.int32)
    bottom_col = np.array([idx(N - 1, j) for j in range(M)], dtype=np.int32)

    return GridTopology(
        N=N, M=M,
        forward_order=jnp.asarray(order),
        backward_order=jnp.asarray(order[::-1].copy()),
        desc_below=jnp.asarray(desc_below),
        desc_right=jnp.asarray(desc_right),
        pred_top=jnp.asarray(pred_top),
        pred_left=jnp.asarray(pred_left),
        left_row_cells=jnp.asarray(left_row),
        bottom_col_cells=jnp.asarray(bottom_col),
    )


def num_cells(topo: GridTopology) -> int:
    return int(topo.N * topo.M)
