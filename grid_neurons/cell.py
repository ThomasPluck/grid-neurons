"""Grid-cell primitive: same first-order Gm-C + tanh + bias as the tree,
but with TWO inputs (from-left and from-above) and therefore two weights.

Forward (zero-order-hold discretisation, exact for constant drive over dt):

    drive_ij(t) = w_L_ij * y_left(t) + w_T_ij * y_top(t)
    a_ij        = exp(-dt/tau_ij),  c_ij = 1 - a_ij
    s_ij(t+1)   = a_ij * s_ij(t) + c_ij * drive_ij(t)
    y_ij(t+1)   = tanh(s_ij(t+1) + b_ij)

Each cell carries four learnable scalars:
    theta_ij = { log_tau_ij, w_L_ij, w_T_ij, b_ij }

Per-step self-traces (`e_*` = ds/dparam, updated in place each timestep):

    e_L (t+1) = a * e_L (t) + c * y_left
    e_T (t+1) = a * e_T (t) + c * y_top
    eta (t+1) = a * eta(t) + (a*dt/tau^2) * (s_old - drive)   # for d s / d log_tau

Bias has no filter-state trace (d y/d b = 1 - y^2 directly).

Backward message-passing Jacobians:
    d s_ij / d y_left  = c_ij * w_L_ij
    d s_ij / d y_top   = c_ij * w_T_ij
So a cell's backward-message-out to each of its TWO descendants (below, right)
is m_ij * (1 - y_ij^2) * c_ij scaled by the descendant's in-weight.
"""
from typing import NamedTuple

import jax.numpy as jnp


class CellParams(NamedTuple):
    log_tau: jnp.ndarray    # per-cell time-constant (log)
    w_left: jnp.ndarray     # weight on left input
    w_top: jnp.ndarray      # weight on top input
    bias: jnp.ndarray       # bias inside tanh


class CellState(NamedTuple):
    s: jnp.ndarray           # filter state
    e_left: jnp.ndarray      # self-trace: ds/dw_left
    e_top: jnp.ndarray       # self-trace: ds/dw_top
    eta: jnp.ndarray         # self-trace: ds/dlog_tau
    # SnAP-1 cross-traces for the two descendants (below, right).
    # Each is (ds_descendant / dw_source), kept per learnable param type.
    # Shape (..., 2)  -- [0] = below, [1] = right.
    e_desc_w_left: jnp.ndarray
    e_desc_w_top:  jnp.ndarray
    e_desc_b:      jnp.ndarray
    e_desc_tau:    jnp.ndarray


def cell_step(
    params: CellParams,
    state: CellState,
    y_left: jnp.ndarray,
    y_top: jnp.ndarray,
    dt: float,
) -> tuple[CellState, jnp.ndarray]:
    """One timestep for a single cell.

    Returns (new_state, output_y). The cross-traces `e_desc_*` are not
    updated here -- they are updated in a second pass once all this step's
    post-step values are known (see block.py for the two-pass forward)."""
    tau = jnp.exp(params.log_tau)
    a = jnp.exp(-dt / tau)
    c = 1.0 - a
    drive = params.w_left * y_left + params.w_top * y_top

    s_new = a * state.s + c * drive
    y_new = jnp.tanh(s_new + params.bias)

    e_L_new = a * state.e_left + c * y_left
    e_T_new = a * state.e_top + c * y_top
    eta_new = a * state.eta + (a * dt / tau ** 2) * (state.s - drive)

    new_state = CellState(
        s=s_new,
        e_left=e_L_new,
        e_top=e_T_new,
        eta=eta_new,
        e_desc_w_left=state.e_desc_w_left,
        e_desc_w_top=state.e_desc_w_top,
        e_desc_b=state.e_desc_b,
        e_desc_tau=state.e_desc_tau,
    )
    return new_state, y_new


def zero_cell_state(shape=()) -> CellState:
    z = jnp.zeros(shape)
    z2 = jnp.zeros(shape + (2,))
    return CellState(
        s=z, e_left=z, e_top=z, eta=z,
        e_desc_w_left=z2, e_desc_w_top=z2, e_desc_b=z2, e_desc_tau=z2,
    )
