"""Training entry points: local SnAP-1 rule vs BPTT reference.

Same shape of API as the tree-neurons paper:

    local_grads(params, s0, seq, tgt, topo, dt, loss_mode)
    bptt_grads (params, s0, seq, tgt, topo, dt, loss_mode)

loss_mode:
  "per_t": L = (1/T) sum_t CE(r_t, target)
  "sum":   L = CE(sum_t r_t, target)      (default for MNIST where per-t
                                           softmax is squashed by tanh range)
"""
import jax
import jax.numpy as jnp

from .block import BlockParams, BlockState, block_backward, block_forward, zero_block_state
from .topology import GridTopology


def readout_from_rates(rates_over_time: jnp.ndarray) -> jnp.ndarray:
    return rates_over_time.sum(axis=0)


def per_timestep_ce(rates: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    def one_t(r_t):
        return -jax.nn.log_softmax(r_t)[target]
    return jax.vmap(one_t)(rates).mean()


def summed_ce(rates: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return -jax.nn.log_softmax(rates.sum(axis=0))[target]


def run_forward(params: BlockParams, init_state: BlockState,
                sequence: jnp.ndarray, topology: GridTopology, dt: float):
    def step(state, x):
        ns, rt, y_all, drives = block_forward(params, state, x, topology, dt)
        return ns, (state, rt, y_all, drives)
    final_state, (states_pre, rates, y_all_t, drives_t) = jax.lax.scan(
        step, init_state, sequence,
    )
    return final_state, rates, y_all_t, drives_t, states_pre


def local_grads(params, init_state, sequence, target, topology, dt,
                loss_mode="per_t"):
    final_state, rates, y_all_t, drives_t, states_pre = run_forward(
        params, init_state, sequence, topology, dt,
    )
    T = sequence.shape[0]
    if loss_mode == "sum":
        loss = summed_ce(rates, target)
        dL_d_r = jax.grad(lambda r: -jax.nn.log_softmax(r)[target])(rates.sum(0))
        dL_d_rate_all = jnp.broadcast_to(dL_d_r[None, :], rates.shape)
    elif loss_mode == "per_t":
        loss = per_timestep_ce(rates, target)
        def dL_dr_at(r_t):
            return jax.grad(lambda rr: -jax.nn.log_softmax(rr)[target])(r_t) / T
        dL_d_rate_all = jax.vmap(dL_dr_at)(rates)
    else:
        raise ValueError(f"unknown loss_mode={loss_mode}")

    # Post-step state at each timestep: states_pre shifted up + final.
    states_post_stack = jax.tree_util.tree_map(
        lambda sp, final: jnp.concatenate([sp[1:], final[None]], axis=0),
        states_pre, final_state,
    )

    def bw_step(acc, t):
        state_post_t = jax.tree_util.tree_map(lambda x: x[t], states_post_stack)
        g_t = block_backward(
            params=params, state_post=state_post_t,
            y_all=y_all_t[t], drives_all=drives_t[t],
            dL_d_rate=dL_d_rate_all[t], topology=topology, dt=dt,
        )
        return jax.tree_util.tree_map(lambda a, b: a + b, acc, g_t), None

    zero_grad = jax.tree_util.tree_map(jnp.zeros_like, params)
    total_grad, _ = jax.lax.scan(bw_step, zero_grad, jnp.arange(T))
    return loss, total_grad, readout_from_rates(rates)


def _bptt_loss(params, init_state, sequence, target, topology, dt, loss_mode):
    def step(state, x):
        ns, rt, _, _ = block_forward(params, state, x, topology, dt)
        return ns, rt
    _, rates = jax.lax.scan(step, init_state, sequence)
    if loss_mode == "sum":
        loss = summed_ce(rates, target)
    elif loss_mode == "per_t":
        loss = per_timestep_ce(rates, target)
    else:
        raise ValueError(f"unknown loss_mode={loss_mode}")
    return loss, readout_from_rates(rates)


def bptt_grads(params, init_state, sequence, target, topology, dt,
               loss_mode="per_t"):
    def loss_only(p):
        l, _ = _bptt_loss(p, init_state, sequence, target, topology, dt, loss_mode)
        return l
    grads = jax.grad(loss_only)(params)
    loss, readout = _bptt_loss(params, init_state, sequence, target, topology,
                               dt, loss_mode)
    return loss, grads, readout
