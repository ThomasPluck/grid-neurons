"""Block-level forward/backward over an N x M grid.

A block is ONE grid. External input flows in through the left shoreline
(N-dim), output comes out of the bottom shoreline (M-dim). No learned
encoder or decoder: the left shoreline IS the input, the bottom
shoreline IS the output (readout).

Forward: one scan over forward_order (raster) running cell_step per cell,
using the just-computed y's of left/top neighbours. After the scan, a
second pass over cells updates the SnAP-1 descendant cross-traces using
post-step values. This mirrors the two-pass scheme from the tree paper.

Backward: one scan over backward_order depositing dL/dy at bottom-row
cells, propagating messages up (to the top neighbour) and left (to the
left neighbour) via the spatial Jacobians, and accumulating per-cell
gradients using the self-traces + SnAP-1 descendant traces.

input_routing: we keep this optional and zero by default. The left
shoreline is a one-per-row direct external input, so for standard runs
the block doesn't need a routing matrix. We expose an optional
input_routing of shape (N, N_ext) so sparse random encoders can be
applied; the encoder is fixed at init (not learned).

Measurement-robustness perturbations: the optional ``Perturbation`` knobs
plug into the *learning path only* (forward second pass + backward).
Forward dynamics (s, y, drive) are unchanged; only the SnAP-1 cross-trace
update and the SnAP-1 backward bracket pick up perturbations. See
``Perturbation`` for the meanings of each scalar.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .cell import CellParams, CellState, cell_step
from .topology import GridTopology, num_cells


class Perturbation(NamedTuple):
    """Knobs for the measurement-noise / -bias robustness study.

    All zeros / ones below = no perturbation (use ``no_perturbation()``).

      delta_leak     : 1a. Cross-trace retention coefficient becomes
                       ``a_d * (1 - delta_leak)`` (cap leakier than nominal).
      msg_gain_pos   : 1b. Backward message ``m`` is multiplied by this
                       when m > 0.
      msg_gain_neg   : 1b. Backward message ``m`` is multiplied by this
                       when m < 0.
                       (Symmetric gain: msg_gain_pos = msg_gain_neg = 1-delta.)
      delta_sub      : 1c. The eps-cross term in the SnAP-1 backward bracket
                       becomes ``(1 + delta_sub) * eps^(1,d)`` while the
                       current-step ``c_d * kappa_d * f`` is kept at unit gain.
                       Models incomplete cancellation of the past-only
                       subtraction.
      sigma_eps_rel  : 2a. Zero-mean Gaussian noise added to each cross-trace
                       after each update step, std = sigma_eps_rel * RMS(eps)
                       computed per-parameter-type, per-step.
      sigma_msg_rel  : 2b. Zero-mean Gaussian noise added to the backward
                       message read at each cell, std = sigma_msg_rel *
                       RMS(dL/d_rate) computed once at the start of the
                       backward pass.
    """
    delta_leak: jnp.ndarray
    msg_gain_pos: jnp.ndarray
    msg_gain_neg: jnp.ndarray
    delta_sub: jnp.ndarray
    sigma_eps_rel: jnp.ndarray
    sigma_msg_rel: jnp.ndarray


def no_perturbation() -> Perturbation:
    """Identity / no-op perturbation: leak=0, gains=1, sub=0, sigmas=0."""
    z = jnp.asarray(0.0, dtype=jnp.float32)
    o = jnp.asarray(1.0, dtype=jnp.float32)
    return Perturbation(
        delta_leak=z, msg_gain_pos=o, msg_gain_neg=o,
        delta_sub=z, sigma_eps_rel=z, sigma_msg_rel=z,
    )


class BlockParams(NamedTuple):
    cells: CellParams       # each field has shape (N*M,)
    # Optional fixed encoder mapping external inputs to left-shoreline drives.
    # Shape (N, N_ext); drives to left-shoreline cell i = sum_n W[i,n] * inp[n].
    # If None, we use identity (N_ext must equal N).
    input_routing: jnp.ndarray


class BlockState(NamedTuple):
    cells: CellState        # each field has shape (N*M, ...)


def init_block_params(key, topology: GridTopology, N_ext: int | None = None,
                      tau_min: float = 5e-3, tau_max: float = 200e-3,
                      bias_init_std: float = 0.05,
                      sparsity: float = 0.1) -> BlockParams:
    n = num_cells(topology)
    k_tau, k_wL, k_wT, k_b, k_enc, k_mask = jax.random.split(key, 6)
    log_tau = jax.random.uniform(
        k_tau, (n,), minval=jnp.log(tau_min), maxval=jnp.log(tau_max))
    # Fan-in of 2 (from left + from top); for top/left shoreline cells the
    # effective fan-in is 1 but we keep the init simple.
    w_left = jax.random.normal(k_wL, (n,)) / jnp.sqrt(2.0)
    w_top = jax.random.normal(k_wT, (n,)) / jnp.sqrt(2.0)
    bias = jax.random.normal(k_b, (n,)) * bias_init_std

    N = topology.N
    if N_ext is None:
        N_ext = N
    if N_ext == N:
        routing = jnp.eye(N, dtype=jnp.float32)
    else:
        dense = jax.random.normal(k_enc, (N, N_ext)) / jnp.sqrt(
            max(N_ext * sparsity, 1.0))
        mask = jax.random.bernoulli(k_mask, p=sparsity, shape=(N, N_ext))
        routing = dense * mask

    return BlockParams(
        cells=CellParams(log_tau=log_tau, w_left=w_left, w_top=w_top,
                         bias=bias),
        input_routing=routing,
    )


def zero_block_state(topology: GridTopology) -> BlockState:
    n = num_cells(topology)
    from .cell import zero_cell_state
    return BlockState(cells=zero_cell_state((n,)))


def route_inputs(W: jnp.ndarray, inputs_ext: jnp.ndarray) -> jnp.ndarray:
    """Apply the fixed encoder. W: (N, N_ext), inputs_ext: (N_ext,) -> (N,)."""
    return W @ inputs_ext


def _gather(arr: jnp.ndarray, idx: jnp.ndarray, zero=0.0):
    """arr[idx] with idx==-1 giving `zero`."""
    safe = jnp.maximum(idx, 0)
    val = arr[safe]
    return jnp.where(idx >= 0, val, jnp.asarray(zero, dtype=arr.dtype))


def block_forward(
    params: BlockParams,
    state: BlockState,
    inputs_ext: jnp.ndarray,       # (N_ext,)
    topology: GridTopology,
    dt: float,
    perturbation: Perturbation | None = None,
    key: jnp.ndarray | None = None,
) -> tuple[BlockState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (new_state, rates_out, y_all, drives_all).

    rates_out:   (M,) values on bottom shoreline at this step.
    y_all:       (N*M,) post-step y values for every cell.
    drives_all:  (N*M,) input-sum drives seen by each cell this step
                 (kept for backward). drive_ij = w_L*y_left + w_T*y_top.
    """
    if perturbation is None:
        perturbation = no_perturbation()
    if key is None:
        key = jax.random.PRNGKey(0)

    n = num_cells(topology)
    N, M = topology.N, topology.M
    left_drives = route_inputs(params.input_routing, inputs_ext)  # (N,)

    cp = params.cells
    cs = state.cells

    def step(carry, idx):
        y_buf, cs_s, cs_e_L, cs_e_T, cs_eta, drives_buf = carry
        # idx: (2,) (i, j) of the cell to update
        i, j = idx[0], idx[1]
        k = i * M + j

        # y_left = left-neighbour's y; if j==0 use left_drives[i].
        y_left = jnp.where(j > 0, y_buf[k - 1], left_drives[i])
        # y_top  = top-neighbour's y; if i==0 use 0.
        y_top = jnp.where(i > 0, y_buf[k - M], jnp.asarray(0.0, dtype=dt_type))

        cell_p = CellParams(
            log_tau=cp.log_tau[k], w_left=cp.w_left[k],
            w_top=cp.w_top[k], bias=cp.bias[k],
        )
        cell_s = CellState(
            s=cs_s[k], e_left=cs_e_L[k], e_top=cs_e_T[k], eta=cs_eta[k],
            e_desc_w_left=cs.e_desc_w_left[k],
            e_desc_w_top=cs.e_desc_w_top[k],
            e_desc_b=cs.e_desc_b[k],
            e_desc_tau=cs.e_desc_tau[k],
        )
        new_cs, y_new = cell_step(cell_p, cell_s, y_left, y_top, dt)

        drive = cell_p.w_left * y_left + cell_p.w_top * y_top

        y_buf = y_buf.at[k].set(y_new)
        cs_s = cs_s.at[k].set(new_cs.s)
        cs_e_L = cs_e_L.at[k].set(new_cs.e_left)
        cs_e_T = cs_e_T.at[k].set(new_cs.e_top)
        cs_eta = cs_eta.at[k].set(new_cs.eta)
        drives_buf = drives_buf.at[k].set(drive)
        return (y_buf, cs_s, cs_e_L, cs_e_T, cs_eta, drives_buf), None

    dt_type = cp.w_left.dtype
    y_init = jnp.zeros(n, dtype=dt_type)
    drives_init = jnp.zeros(n, dtype=dt_type)
    init_carry = (y_init, cs.s, cs.e_left, cs.e_top, cs.eta, drives_init)

    (y_all, s_all, eL_all, eT_all, eta_all, drives_all), _ = jax.lax.scan(
        step, init_carry, topology.forward_order,
    )

    # ---- Second pass: update SnAP-1 descendant cross-traces ----
    # For cell k with (post-step) output y_k, drive coefficients
    # feeding into each descendant (d in {below, right}):
    #   s_d(t+1) = a_d s_d(t) + c_d * (w_T_d * y_top_d + w_L_d * y_left_d)
    # If d = below(k), then y_top_d = y_k; the other feeder for d is a
    # sibling, unrelated to k's parameters.
    # Cross-trace update for cell k, descendant d=below:
    #   eps^(1)_w_L[k, below](t+1) = a_d eps^(1)_w_L[k, below](t)
    #                              + c_d * w_T_d * (1 - y_k^2) * e_left_k(t+1)
    # (analogously for the other param types and for d=right with w_L_d factor).
    tau_all = jnp.exp(cp.log_tau)
    a_all = jnp.exp(-dt / tau_all)
    c_all = 1.0 - a_all
    y_s_all = jnp.tanh(s_all + cp.bias)
    omy2 = 1.0 - y_s_all ** 2

    e_desc_w_L = cs.e_desc_w_left     # (n, 2)
    e_desc_w_T = cs.e_desc_w_top
    e_desc_b   = cs.e_desc_b
    e_desc_tau = cs.e_desc_tau

    # Below-descendant update (desc index = topology.desc_below[k]).
    below_idx = topology.desc_below    # (n,)
    right_idx = topology.desc_right

    has_below = below_idx >= 0
    has_right = right_idx >= 0
    safe_b = jnp.maximum(below_idx, 0)
    safe_r = jnp.maximum(right_idx, 0)

    # Per-descendant coefficients for "below" feed from k: y_k becomes y_top of below.
    a_below = a_all[safe_b]; c_below = c_all[safe_b]
    w_T_below = cp.w_top[safe_b]
    # For "right" feed: y_k becomes y_left of right.
    a_right = a_all[safe_r]; c_right = c_all[safe_r]
    w_L_right = cp.w_left[safe_r]

    # dy_k / dparam_k (post-step). For bias: just (1 - y_k^2).
    dy_dL  = omy2 * eL_all
    dy_dT  = omy2 * eT_all
    dy_db  = omy2
    dy_dlt = omy2 * eta_all * tau_all

    # 1a (trace leak bias): a_d -> a_d * (1 - delta_leak) for cross-traces only.
    leak_factor = 1.0 - perturbation.delta_leak
    a_below_pert = a_below * leak_factor
    a_right_pert = a_right * leak_factor

    # Update cross-traces, masked by has_below/has_right.
    def update(old_below, old_right, dy):
        upd_b = jnp.where(has_below,
                          a_below_pert * old_below + c_below * w_T_below * dy,
                          old_below)
        upd_r = jnp.where(has_right,
                          a_right_pert * old_right + c_right * w_L_right * dy,
                          old_right)
        return jnp.stack([upd_b, upd_r], axis=-1)

    new_e_desc_w_L = update(e_desc_w_L[:, 0], e_desc_w_L[:, 1], dy_dL)
    new_e_desc_w_T = update(e_desc_w_T[:, 0], e_desc_w_T[:, 1], dy_dT)
    new_e_desc_b   = update(e_desc_b[:, 0],   e_desc_b[:, 1],   dy_db)
    new_e_desc_tau = update(e_desc_tau[:, 0], e_desc_tau[:, 1], dy_dlt)

    # 2a (cross-trace noise): add zero-mean Gaussian noise per parameter-type,
    # std = sigma_eps_rel * RMS(eps) computed across this batch's array.
    # When sigma_eps_rel == 0 this is a no-op (multiply by zero).
    def _add_noise(eps_arr, k):
        rms = jnp.sqrt(jnp.mean(eps_arr ** 2) + 1e-12)
        scale = perturbation.sigma_eps_rel * rms
        noise = jax.random.normal(k, eps_arr.shape, dtype=eps_arr.dtype) * scale
        return eps_arr + noise

    k_eL, k_eT, k_eb, k_etau = jax.random.split(key, 4)
    new_e_desc_w_L = _add_noise(new_e_desc_w_L, k_eL)
    new_e_desc_w_T = _add_noise(new_e_desc_w_T, k_eT)
    new_e_desc_b   = _add_noise(new_e_desc_b,   k_eb)
    new_e_desc_tau = _add_noise(new_e_desc_tau, k_etau)

    new_cells = CellState(
        s=s_all, e_left=eL_all, e_top=eT_all, eta=eta_all,
        e_desc_w_left=new_e_desc_w_L, e_desc_w_top=new_e_desc_w_T,
        e_desc_b=new_e_desc_b, e_desc_tau=new_e_desc_tau,
    )
    rates_out = y_s_all[topology.bottom_col_cells]   # (M,)
    return BlockState(cells=new_cells), rates_out, y_all, drives_all


def block_backward(
    params: BlockParams,
    state_post: BlockState,
    y_all: jnp.ndarray,             # (N*M,) post-step outputs this step
    drives_all: jnp.ndarray,        # (N*M,) drive values this step (unused for now)
    dL_d_rate: jnp.ndarray,         # (M,) dL/dy at bottom shoreline this step
    topology: GridTopology,
    dt: float,
    perturbation: Perturbation | None = None,
    key: jnp.ndarray | None = None,
) -> BlockParams:
    """Single-timestep local backward. Returns per-parameter gradients for
    branch parameters; input_routing gradient returned as zero (encoder
    frozen by design). SnAP-1 past-only subtraction applied when using
    descendant cross-traces (same structure as the tree paper)."""
    if perturbation is None:
        perturbation = no_perturbation()
    if key is None:
        key = jax.random.PRNGKey(0)

    N, M = topology.N, topology.M
    n = num_cells(topology)

    cp = params.cells
    cs = state_post.cells

    tau_all = jnp.exp(cp.log_tau)
    a_all = jnp.exp(-dt / tau_all)
    c_all = 1.0 - a_all
    y_s_all = jnp.tanh(cs.s + cp.bias)
    omy2 = 1.0 - y_s_all ** 2

    # Deposit dL/d_rate at bottom-row cells.
    dt_type = cp.w_left.dtype
    msg_init = jnp.zeros(n, dtype=dt_type).at[topology.bottom_col_cells].set(dL_d_rate)

    # 2b: precompute msg-noise std relative to dL/d_rate RMS at this step.
    # The msg array starts populated only at the bottom shoreline; using its
    # RMS at the moment of read varies wildly during the backward sweep, so we
    # fix the noise scale once based on the readout-message magnitude.
    msg_rms = jnp.sqrt(jnp.mean(dL_d_rate ** 2) + 1e-12)
    noise_std = perturbation.sigma_msg_rel * msg_rms

    dw_L = jnp.zeros(n, dtype=dt_type)
    dw_T = jnp.zeros(n, dtype=dt_type)
    dbias = jnp.zeros(n, dtype=dt_type)
    dlog_tau = jnp.zeros(n, dtype=dt_type)

    # 1b helpers: scale m by gain_pos when m>0, gain_neg when m<0.
    gp = perturbation.msg_gain_pos
    gn = perturbation.msg_gain_neg

    def _msg_perturb(m, k_noise):
        # 2b additive noise, then 1b sign-asymmetric gain.
        m_noisy = m + jax.random.normal(k_noise, m.shape, dtype=m.dtype) * noise_std
        return jnp.where(m_noisy > 0, m_noisy * gp, m_noisy * gn)

    # 1c: subtraction bias multiplies the eps^(1,d) term in the bracket.
    sub_factor = 1.0 + perturbation.delta_sub

    def step(carry, scan_in):
        msg, dwL, dwT, dB, dLT = carry
        idx, k_msg_b, k_msg_d_below, k_msg_d_right = scan_in
        i, j = idx[0], idx[1]
        k = i * M + j
        m_b_raw = msg[k]
        m_b = _msg_perturb(m_b_raw, k_msg_b)
        omy2_b = omy2[k]

        # Self-path (spatial contribution through y_k -> m_k).
        dwL_self = m_b * omy2_b * cs.e_left[k]
        dwT_self = m_b * omy2_b * cs.e_top[k]
        dB_self  = m_b * omy2_b
        dLT_self = m_b * omy2_b * cs.eta[k] * tau_all[k]

        # SnAP-1 descendant contributions. For each descendant d in
        # {below, right}, we need:
        #    m_d * (1 - y_d^2) * past_eps^(1)[k,d]
        # where past_eps = eps(t+1) - current-step contribution.
        def desc_term(desc_idx, slot, k_msg_d):
            has = desc_idx >= 0
            safe = jnp.maximum(desc_idx, 0)
            m_d_raw = msg[safe]
            m_d = _msg_perturb(m_d_raw, k_msg_d)
            omy2_d = omy2[safe]
            a_d = a_all[safe]; c_d = c_all[safe]
            # Coefficient feeding y_k into descendant d.
            # slot=0 below: d = below(k), y_k is the top-input of d, so factor = w_top_d
            # slot=1 right: d = right(k), y_k is the left-input of d, factor = w_left_d
            w_feed = jnp.where(slot == 0, cp.w_top[safe], cp.w_left[safe])
            # current-step self-feed contribution to eps_d:
            #   eps(t+1) = a eps(t) + c * w_feed * dy_dparam(t+1)
            # so past = eps(t+1) - c * w_feed * dy_dparam(t+1).
            # 1c subtraction bias: scale eps^(1,d) by (1+delta_sub) relative
            # to the current-step c_d * kappa_d * f term.
            dy_dL = omy2_b * cs.e_left[k]
            dy_dT = omy2_b * cs.e_top[k]
            dy_db = omy2_b
            dy_dlt = omy2_b * cs.eta[k] * tau_all[k]

            past_L  = sub_factor * cs.e_desc_w_left[k, slot] - c_d * w_feed * dy_dL
            past_T  = sub_factor * cs.e_desc_w_top [k, slot] - c_d * w_feed * dy_dT
            past_b  = sub_factor * cs.e_desc_b     [k, slot] - c_d * w_feed * dy_db
            past_lt = sub_factor * cs.e_desc_tau   [k, slot] - c_d * w_feed * dy_dlt

            gate = jnp.where(has, m_d * omy2_d, 0.0)
            return (gate * past_L, gate * past_T, gate * past_b, gate * past_lt)

        bL0, bT0, bB0, bLT0 = desc_term(topology.desc_below[k], 0, k_msg_d_below)
        bL1, bT1, bB1, bLT1 = desc_term(topology.desc_right[k], 1, k_msg_d_right)

        dwL = dwL.at[k].set(dwL_self + bL0 + bL1)
        dwT = dwT.at[k].set(dwT_self + bT0 + bT1)
        dB  = dB .at[k].set(dB_self  + bB0 + bB1)
        dLT = dLT.at[k].set(dLT_self + bLT0 + bLT1)

        # Propagate message to the two predecessors (top and left of k).
        # Use the perturbed m_b: realistic analog readout failure compounds
        # as messages traverse the grid.
        out_msg = m_b * cp.w_left[k] * omy2_b * c_all[k]  # flows to left predecessor
        out_msg_top = m_b * cp.w_top[k] * omy2_b * c_all[k]  # flows to top predecessor

        left_k = topology.pred_left[k]
        top_k = topology.pred_top[k]
        msg = jax.lax.cond(
            left_k >= 0,
            lambda m: m.at[jnp.maximum(left_k, 0)].add(out_msg),
            lambda m: m, msg,
        )
        msg = jax.lax.cond(
            top_k >= 0,
            lambda m: m.at[jnp.maximum(top_k, 0)].add(out_msg_top),
            lambda m: m, msg,
        )
        return (msg, dwL, dwT, dB, dLT), None

    # Three independent streams of msg-noise per cell (for m_b, m_d_below, m_d_right).
    # Single big split is faster than the vmap'd nested split.
    all_keys = jax.random.split(key, n * 3)
    keys_per_cell = all_keys.reshape((n, 3) + all_keys.shape[1:])
    scan_inputs = (
        topology.backward_order,
        keys_per_cell[:, 0],
        keys_per_cell[:, 1],
        keys_per_cell[:, 2],
    )

    (_, dw_L, dw_T, dbias, dlog_tau), _ = jax.lax.scan(
        step, (msg_init, dw_L, dw_T, dbias, dlog_tau), scan_inputs,
    )

    return BlockParams(
        cells=CellParams(
            log_tau=dlog_tau, w_left=dw_L, w_top=dw_T, bias=dbias),
        input_routing=jnp.zeros_like(params.input_routing),
    )
