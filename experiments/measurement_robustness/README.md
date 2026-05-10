# Measurement-noise robustness sweep

Reviewer-driven study (Sam Dillavou): how does the SnAP-1 grid rule
degrade under bias and precision errors in the *measured* quantities of
its update — distinct from parameter mismatch or quantisation.

## Layout

```
experiments/measurement_robustness/
  sweeps.py        # the eight sweep configs (bias 1a/1b/1c + precision 2a/2b/2c)
  run_sweep.py     # training driver: identical to scripts/mnist_rowwise.py
                   # except the SnAP-1 rule receives a per-condition Perturbation
  baseline_check.py# single unperturbed run, used to gauge timing
  analyze.py       # CSV -> summary table -> per-sweep + combined plots -> markdown report
  results/         # per-sweep CSVs (one row per epoch per condition)
  plots/           # generated plots
  measurement_robustness.md  # final markdown report
```

## Reproducing

The full sweep at the spec'd 5-epochs / single-seed config takes
~5 hours on CPU at ``--train-size 6000``. Section 4 of the paper is
``--train-size 60000`` (10x larger); robustness *trends* are visible at
the smaller scale and the comparison across conditions is the point.

```bash
python -u experiments/measurement_robustness/run_sweep.py \
    --sweep all --epochs 5 --train-size 6000

python -u experiments/measurement_robustness/analyze.py
```

To run a single sweep:

```bash
python -u experiments/measurement_robustness/run_sweep.py \
    --sweep 1c_subtraction --epochs 5 --train-size 6000
```

## What's measured

All perturbations live in ``grid_neurons/block.py`` (``Perturbation``
NamedTuple). Forward dynamics are untouched — the rule's *measurements*
are perturbed, the network's *computation* is not.

| Sweep | Knob in `Perturbation` | What it models |
|---|---|---|
| 1a leak               | `delta_leak`               | Cap leakier than its filter constant |
| 1b gain (sym + asym)  | `msg_gain_pos`, `msg_gain_neg` | Backward-message readout gain offset, optionally sign-asymmetric |
| 1c subtraction        | `delta_sub`                | Mismatched gain on the cross-trace term vs the current-step subtraction in Eq. 8 |
| 2a $\varepsilon$ noise  | `sigma_eps_rel`            | Additive Gaussian on the cross-traces, std = $\sigma$ × RMS |
| 2b msg noise          | `sigma_msg_rel`            | Additive Gaussian on the backward messages |
| 2c combined           | both                       | The realistic both-readouts-noisy case |
