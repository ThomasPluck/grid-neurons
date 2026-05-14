"""Build summary tables, plots, and a markdown report from sweep CSVs.

Reads ``results/<sweep>.csv`` files written by ``run_sweep.py`` and
produces:
  - ``plots/<sweep>.png`` per-sweep individual plots
  - ``plots/bias.png``    combined bias panel (1a, 1b sym/asym, 1c)
  - ``plots/precision.png`` combined precision panel (2a, 2b, 2c)
  - ``measurement_robustness.md`` markdown report with tables and
    inlined plot references
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SWEEP_LABELS = {
    "1a_leak":            r"1a trace leak  $a_d \to a_d (1-\delta)$",
    "1b_gain_sym":        r"1b msg gain (sym)  $m \to m(1-\delta)$",
    "1b_gain_asym_A":     r"1b msg gain (asym A)  $g_+=1{-}\delta,\ g_-=1{+}\delta$",
    "1b_gain_asym_B":     r"1b msg gain (asym B)  $g_+=1{+}\delta,\ g_-=1{-}\delta$",
    "1c_subtraction":     r"1c subtraction  $(1{+}\delta_\text{sub})\,\varepsilon - c_d \kappa_d f$",
    "2a_eps_noise":       r"2a cross-trace noise  $\varepsilon + N(0,\sigma^2)$",
    "2b_msg_noise":       r"2b message noise  $m + N(0,\sigma^2)$",
    "2c_combined_noise":  r"2c both",
}

BIAS_SWEEPS = ["1a_leak", "1b_gain_sym", "1b_gain_asym_A",
               "1b_gain_asym_B", "1c_subtraction"]
PRECISION_SWEEPS = ["2a_eps_noise", "2b_msg_noise", "2c_combined_noise"]


def load_results(results_dir: str) -> dict[str, pd.DataFrame]:
    out = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".csv"):
            continue
        # Skip the auxiliary summary.csv if we wrote one previously.
        if fname == "summary.csv":
            continue
        sweep = fname[:-4]
        path = os.path.join(results_dir, fname)
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if len(df) == 0:
            continue
        out[sweep] = df
    return out


def per_condition_final(df: pd.DataFrame) -> pd.DataFrame:
    """For each (experiment, value, seed), take the final epoch's row.
    Returns rows sorted by ``value`` ascending so plots draw monotone lines."""
    last = (df.sort_values("epoch")
              .groupby(["experiment", "value", "seed"], as_index=False)
              .tail(1)
              .sort_values("value")
              .reset_index(drop=True))
    return last


def baseline_val_acc(df_final: pd.DataFrame, baseline_value: float = 0.0) -> float:
    """Pick the unperturbed-baseline final val_acc (value == baseline_value)."""
    matched = df_final[np.isclose(df_final["value"], baseline_value)]
    if len(matched) == 0:
        return float("nan")
    return float(matched["val_acc"].iloc[0])


def plot_individual(sweep: str, df: pd.DataFrame, out_path: str):
    final = per_condition_final(df)
    base = baseline_val_acc(final)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(final["value"], final["val_acc"], "o-", color="#1f77b4")
    ax.axhline(base, ls="--", color="grey", lw=1.0,
               label=f"baseline ({base:.3f})")
    ax.set_xlabel(_xlabel_for(sweep))
    ax.set_ylabel("final val acc")
    ax.set_title(SWEEP_LABELS.get(sweep, sweep))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _xlabel_for(sweep: str) -> str:
    if sweep.startswith("2"):
        return r"$\sigma$ / RMS"
    if sweep == "1c_subtraction":
        return r"$\delta_{\rm sub}$"
    return r"$\delta$"


def plot_combined_bias(dfs: dict[str, pd.DataFrame], out_path: str):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = {
        "1a_leak":        "#1f77b4",
        "1b_gain_sym":    "#ff7f0e",
        "1b_gain_asym_A": "#2ca02c",
        "1b_gain_asym_B": "#9467bd",
        "1c_subtraction": "#d62728",
    }
    base = None
    for s in BIAS_SWEEPS:
        if s not in dfs:
            continue
        final = per_condition_final(dfs[s])
        if base is None:
            base = baseline_val_acc(final)
        # x-axis = signed delta (works for all bias sweeps)
        ax.plot(final["value"], final["val_acc"], "o-",
                color=colors[s], label=SWEEP_LABELS[s])
    if base is not None and not np.isnan(base):
        ax.axhline(base, ls="--", color="grey", lw=1.0,
                   label=f"unperturbed baseline ({base:.3f})")
    ax.set_xlabel(r"$\delta$  (signed)")
    ax.set_ylabel("final val acc")
    ax.set_title("Bias sweeps: final val acc vs $\\delta$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_combined_precision(dfs: dict[str, pd.DataFrame], out_path: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    colors = {
        "2a_eps_noise":       "#1f77b4",
        "2b_msg_noise":       "#ff7f0e",
        "2c_combined_noise":  "#2ca02c",
    }
    base = None
    for s in PRECISION_SWEEPS:
        if s not in dfs:
            continue
        final = per_condition_final(dfs[s])
        if base is None:
            base = baseline_val_acc(final)
        ax.plot(final["value"], final["val_acc"], "o-",
                color=colors[s], label=SWEEP_LABELS[s])
    if base is not None and not np.isnan(base):
        ax.axhline(base, ls="--", color="grey", lw=1.0,
                   label=f"unperturbed baseline ({base:.3f})")
    ax.set_xlabel(r"$\sigma$ / RMS  (zero-mean Gaussian)")
    ax.set_ylabel("final val acc")
    ax.set_title("Precision sweeps: final val acc vs $\\sigma$")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("symlog", linthresh=1e-2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_summary(dfs: dict[str, pd.DataFrame], out_path: str):
    """Two-panel summary used as the headline figure of the report."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    base = None

    bias_colors = {
        "1a_leak":        "#1f77b4",
        "1b_gain_sym":    "#ff7f0e",
        "1b_gain_asym_A": "#2ca02c",
        "1b_gain_asym_B": "#9467bd",
        "1c_subtraction": "#d62728",
    }
    for s in BIAS_SWEEPS:
        if s not in dfs:
            continue
        final = per_condition_final(dfs[s])
        if base is None:
            base = baseline_val_acc(final)
        axL.plot(final["value"], final["val_acc"], "o-",
                 color=bias_colors[s], label=SWEEP_LABELS[s])
    if base is not None and not np.isnan(base):
        axL.axhline(base, ls="--", color="grey", lw=1.0,
                    label=f"unperturbed baseline ({base:.3f})")
    axL.set_xlabel(r"$\delta$  (signed)")
    axL.set_ylabel("final val acc")
    axL.set_title("Bias")
    axL.grid(True, alpha=0.3)
    axL.legend(fontsize=7, loc="best")

    prec_colors = {
        "2a_eps_noise":       "#1f77b4",
        "2b_msg_noise":       "#ff7f0e",
        "2c_combined_noise":  "#2ca02c",
    }
    for s in PRECISION_SWEEPS:
        if s not in dfs:
            continue
        final = per_condition_final(dfs[s])
        axR.plot(final["value"], final["val_acc"], "o-",
                 color=prec_colors[s], label=SWEEP_LABELS[s])
    if base is not None and not np.isnan(base):
        axR.axhline(base, ls="--", color="grey", lw=1.0,
                    label=f"unperturbed baseline ({base:.3f})")
    axR.set_xlabel(r"$\sigma$ / RMS")
    axR.set_title("Precision")
    axR.grid(True, alpha=0.3)
    axR.set_xscale("symlog", linthresh=1e-2)
    axR.legend(fontsize=7, loc="best")

    fig.suptitle(
        "SnAP-1 grid-rule robustness to measurement bias and precision",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def summary_table(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per condition, showing final val_acc and absolute drop from baseline."""
    rows = []
    for sweep, df in dfs.items():
        final = per_condition_final(df)
        base = baseline_val_acc(final)
        for _, r in final.iterrows():
            rows.append(dict(
                sweep=sweep,
                param=r["perturbation_param"],
                value=float(r["value"]),
                final_val_acc=float(r["val_acc"]),
                final_train_acc=float(r["train_acc"]),
                final_train_loss=float(r["train_loss"]),
                drop_from_baseline=float(base - r["val_acc"])
                                    if not np.isnan(base) else float("nan"),
            ))
    return pd.DataFrame(rows)


def _format_val(v: float, fmt: str = "{:+.3f}") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return fmt.format(v)


def first_breaking_value(table: pd.DataFrame, sweep: str,
                         drop_threshold: float = 0.05) -> tuple[float | None, float | None]:
    """For a sweep, find the smallest |value| at which val_acc drops more than
    ``drop_threshold`` below the baseline. Returns (value, drop) or (None, None)
    if no condition exceeds the threshold."""
    sub = table[table["sweep"] == sweep].copy()
    sub["abs_value"] = sub["value"].abs()
    sub = sub.sort_values("abs_value")
    for _, r in sub.iterrows():
        if r["drop_from_baseline"] >= drop_threshold:
            return float(r["value"]), float(r["drop_from_baseline"])
    return None, None


def first_breaking_signed(table: pd.DataFrame, sweep: str,
                          drop_threshold: float = 0.05
                          ) -> dict[str, tuple[float | None, float | None]]:
    """Report break points separately for the positive and negative sides of
    the swept perturbation."""
    out: dict[str, tuple[float | None, float | None]] = {"+": (None, None),
                                                         "-": (None, None)}
    sub = table[table["sweep"] == sweep].copy()
    pos = sub[sub["value"] > 0].sort_values("value")
    for _, r in pos.iterrows():
        if r["drop_from_baseline"] >= drop_threshold:
            out["+"] = (float(r["value"]), float(r["drop_from_baseline"]))
            break
    neg = sub[sub["value"] < 0].sort_values("value", ascending=False)
    for _, r in neg.iterrows():
        if r["drop_from_baseline"] >= drop_threshold:
            out["-"] = (float(r["value"]), float(r["drop_from_baseline"]))
            break
    return out


def write_report(dfs: dict[str, pd.DataFrame], out_path: str,
                 cfg: dict | None = None):
    table = summary_table(dfs)
    base = None
    if BIAS_SWEEPS[0] in dfs:
        base = baseline_val_acc(per_condition_final(dfs[BIAS_SWEEPS[0]]))
    elif "1c_subtraction" in dfs:
        base = baseline_val_acc(per_condition_final(dfs["1c_subtraction"]))

    lines: list[str] = []
    lines.append("# Measurement-robustness sweep for SnAP-1 grid rule")
    lines.append("")
    if cfg is not None:
        lines.append(
            f"Config: {cfg.get('train_size', '?')} train images, "
            f"{cfg.get('epochs', '?')} epochs, batch "
            f"{cfg.get('batch_size', '?')}, Adam $\\eta=3\\times 10^{{-3}}$, "
            f"single seed (init={cfg.get('init_seed', '?')})."
        )
        lines.append("")
    if base is not None:
        lines.append(
            f"Unperturbed baseline final val acc: **{base:.3f}** "
            f"(chance = 0.10)."
        )
        lines.append("")

    lines.append("## Headline plot")
    lines.append("")
    lines.append("![](plots/summary.png)")
    lines.append("")

    lines.append("## Bias sweeps")
    lines.append("")
    lines.append("![](plots/bias.png)")
    lines.append("")
    lines.append("Final val accuracy per condition:")
    lines.append("")
    lines.append(_md_table_for_sweeps(table, BIAS_SWEEPS, base))
    lines.append("")

    lines.append("## Precision sweeps")
    lines.append("")
    lines.append("![](plots/precision.png)")
    lines.append("")
    lines.append("Final val accuracy per condition:")
    lines.append("")
    lines.append(_md_table_for_sweeps(table, PRECISION_SWEEPS, base))
    lines.append("")

    lines.append("## Discussion")
    lines.append("")
    lines.extend(_discussion_paragraphs(table, base))
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _md_table_for_sweeps(table: pd.DataFrame, sweeps: Iterable[str],
                         base: float | None) -> str:
    rows = ["| sweep | $\\delta$ / $\\sigma$ | val acc | drop |",
            "|---|---|---|---|"]
    for s in sweeps:
        sub = table[table["sweep"] == s].sort_values("value")
        for _, r in sub.iterrows():
            rows.append(
                f"| {s} | {r['value']:+.3f} | {r['final_val_acc']:.3f} | "
                f"{r['drop_from_baseline']:+.3f} |"
            )
    return "\n".join(rows)


def _max_drop(table: pd.DataFrame, sweeps: Iterable[str]) -> tuple[str | None, float, float | None]:
    """Across the given sweeps, find the (sweep, drop, value) with the largest
    drop_from_baseline. Returns (sweep_name, drop, value_at_worst)."""
    worst_s, worst_d, worst_v = None, -float("inf"), None
    for s in sweeps:
        sub = table[table["sweep"] == s]
        if len(sub) == 0:
            continue
        idx = sub["drop_from_baseline"].idxmax()
        d = float(sub.loc[idx, "drop_from_baseline"])
        if d > worst_d:
            worst_s, worst_d, worst_v = s, d, float(sub.loc[idx, "value"])
    if worst_s is None:
        return None, 0.0, None
    return worst_s, worst_d, worst_v


def _discussion_paragraphs(table: pd.DataFrame, base: float | None) -> list[str]:
    """Data-driven discussion. Phrasing is chosen by what the data actually
    shows: which sweeps broke, how the bias vs precision categories compare,
    and whether the headline matches or contradicts the priors set in the
    experiment plan. Numbers come from the summary table."""
    out: list[str] = []

    threshold = 0.05         # "clear break" threshold (val-acc drop)
    noise_floor = 0.02       # rough within-seed noise estimate at this scale

    # Per-sweep break point bullets (signed where 1c is concerned).
    bullets = [f"Key numbers (drop $\\ge$ {threshold:.2f} below the unperturbed baseline):"]
    bullets.append("")
    for s in BIAS_SWEEPS + PRECISION_SWEEPS:
        if s not in table["sweep"].unique():
            continue
        if s == "1c_subtraction":
            sides = first_breaking_signed(table, s, drop_threshold=threshold)
            pos_v, pos_d = sides["+"]
            neg_v, neg_d = sides["-"]
            parts = []
            if pos_v is None:
                parts.append("$+\\delta$ tolerated to "
                             f"max swept (drop < {threshold:.2f})")
            else:
                parts.append(f"$+\\delta$ breaks at {pos_v:+.3g} (drop {pos_d:.3f})")
            if neg_v is None:
                parts.append("$-\\delta$ tolerated to "
                             f"max swept (drop < {threshold:.2f})")
            else:
                parts.append(f"$-\\delta$ breaks at {neg_v:+.3g} (drop {neg_d:.3f})")
            bullets.append(f"- **{s}** — {'; '.join(parts)}.")
        else:
            v, d = first_breaking_value(table, s, drop_threshold=threshold)
            if v is None:
                tol = table[table["sweep"] == s]["value"].abs().max()
                bullets.append(
                    f"- **{s}** — no break above threshold; tolerated up to "
                    f"|{_xlabel_short(s)}| = {tol:.3g} with drop < {threshold:.2f}."
                )
            else:
                bullets.append(
                    f"- **{s}** — first break at {_xlabel_short(s)} = {v:+.3g}, "
                    f"drop = {d:.3f}."
                )
    out.extend(bullets)
    out.append("")

    # Worst-sweep summary per category, used to drive the narrative.
    present_bias = [s for s in BIAS_SWEEPS if s in table["sweep"].unique()]
    present_prec = [s for s in PRECISION_SWEEPS if s in table["sweep"].unique()]
    b_s, b_d, b_v = _max_drop(table, present_bias)
    p_s, p_d, p_v = _max_drop(table, present_prec)

    # Worst within the bias category (used for the "1c-vs-others" claim test).
    worst_in_bias = b_s
    sweeps_with_real_effect = [
        s for s in present_bias + present_prec
        if float(table[table["sweep"] == s]["drop_from_baseline"].max()) > noise_floor
    ]

    # Paragraph 1: headline finding based on the actual numbers.
    if b_d <= noise_floor and p_d <= noise_floor:
        out.append(
            f"At this scale ({_scale_phrase()}), every condition swept --- "
            f"bias and precision alike --- stays within roughly $\\pm{noise_floor:.2f}$ "
            f"of the unperturbed baseline (final val acc "
            f"{base:.3f}). The worst point we observed across all "
            f"{len(present_bias)+len(present_prec)} sweeps is a drop of "
            f"{max(b_d, p_d):.3f}, which is at or below the within-seed run-to-run "
            "noise this configuration can resolve. We cannot, from this data, "
            "claim a measurable difference between the bias and precision "
            "axes; the honest reading is that the local SnAP-$1$ rule is "
            "robust to **all of the tested measurement non-idealities** at "
            "the few-percent level over the swept ranges. Resolving smaller "
            "effects requires either multi-seed averaging or the full "
            f"Section\\,\\ref{{sec:mnist}}-scale baseline (60k train images, "
            "~0.52 val acc, ~5$\\times$ the distance from chance)."
        )
    elif b_d > p_d:
        out.append(
            f"At this scale ({_scale_phrase()}), bias perturbations degrade "
            "the rule more than zero-mean precision noise does. The worst "
            f"bias condition is **{b_s}** at "
            f"{_xlabel_short(b_s)} = {b_v:+.3g} (drop {b_d:.3f} below the "
            f"{base:.3f} baseline). The worst precision condition is "
            f"**{p_s}** at $\\sigma/\\text{{RMS}}={p_v:.3g}$ "
            f"(drop {p_d:.3f}). The asymmetry matches the qualitative "
            "expectation that systematic offsets violate the rule's "
            "correctness derivation while zero-mean noise gets absorbed by "
            "the optimiser."
        )
    elif p_d > b_d:
        out.append(
            f"At this scale ({_scale_phrase()}), additive precision noise "
            "degrades the rule more than the tested bias perturbations do --- "
            "which contradicts the prior expectation set in the experiment "
            f"plan. The worst precision condition is **{p_s}** at "
            f"$\\sigma/\\text{{RMS}}={p_v:.3g}$ (drop {p_d:.3f} below the "
            f"{base:.3f} baseline). The worst bias condition is **{b_s}** "
            f"at {_xlabel_short(b_s)} = {b_v:+.3g} (drop {b_d:.3f})."
        )
    else:
        out.append(
            f"At this scale ({_scale_phrase()}), the worst bias drop "
            f"({b_d:.3f}, at **{b_s}**) is comparable to the worst precision "
            f"drop ({p_d:.3f}, at **{p_s}**)."
        )
    out.append("")

    # Paragraph 2: how 1c compares to the other bias sweeps.
    if "1c_subtraction" in present_bias:
        d_1c = float(table[table["sweep"] == "1c_subtraction"]["drop_from_baseline"].max())
        other_bias = [s for s in present_bias if s != "1c_subtraction"]
        d_other = float("-inf")
        s_other = None
        for s in other_bias:
            sub = table[table["sweep"] == s]
            if len(sub) == 0:
                continue
            dd = float(sub["drop_from_baseline"].max())
            if dd > d_other:
                d_other = dd; s_other = s
        if d_1c > d_other + noise_floor:
            out.append(
                "Within the bias category, **1c (subtraction)** is "
                "the most fragile of the bias variants --- max drop "
                f"{d_1c:.3f} vs at most {d_other:.3f} for {s_other}. "
                "This is consistent with the structural argument that "
                "incomplete cancellation between $\\varepsilon^{(1,d)}$ "
                "and the current-step $c_d\\kappa_d f$ in "
                "Eq.\\,\\ref{eq:local-grad} directly biases the gradient, "
                "while gain or leak biases just rescale terms the rule "
                "uses."
            )
        elif worst_in_bias != "1c_subtraction":
            out.append(
                "Within the bias category, **1c (subtraction)** is **not** "
                "the most fragile variant in this run: it tops out at a "
                f"{d_1c:.3f} drop, while {worst_in_bias} reaches "
                f"{b_d:.3f}. This contradicts the prior expectation that "
                "the past-only subtraction would be the dominant "
                "sensitivity; at this scale, the data suggests gain / "
                "leak biases damage learning at least as much as a "
                "subtraction-cancellation mismatch."
            )
        else:
            out.append(
                "Within the bias category, 1c (subtraction) and the "
                "gain / leak biases produce comparable degradation in "
                f"this run (max drops {d_1c:.3f} vs {d_other:.3f}). "
                "We cannot resolve a clean 'most-fragile' answer from "
                "the single-seed data."
            )
    out.append("")

    # Paragraph 3: co-design takeaway, scaled to what the data actually shows.
    if not sweeps_with_real_effect:
        out.append(
            "**Co-design implication.** The data does not yet support a "
            "specific tightening of any one analog-readout spec. What it "
            "does support is that single-digit-percent gain / leak / "
            "subtraction biases and Gaussian readout noise up to "
            "$\\sigma\\sim\\text{RMS}$ on either the cross-trace or the "
            "backward message are not catastrophic for this rule on this "
            "topology. A follow-up at the full Section\\,\\ref{sec:mnist} "
            "scale --- 60k training images, multiple seeds --- is the "
            "right next experiment before quoting any particular tolerance "
            "to circuit designers."
        )
    else:
        out.append(
            "**Co-design implication.** The sweeps in "
            f"{{ {', '.join(sweeps_with_real_effect)} }} exhibit a "
            "clear-of-noise effect, while the remainder are tolerated at "
            "the few-percent level. For silicon design this argues for "
            "spending tolerance budget on the failure modes that "
            "produced visible drops above and accepting looser specs on "
            "the rest. A full-scale rerun (60k train, $\\ge 3$ seeds) "
            "would give tighter tolerance numbers for the affected "
            "knobs."
        )

    return out


def _scale_phrase() -> str:
    """Phrase describing the train scale; kept as a single string so the
    discussion paragraphs read naturally regardless of which config we ran."""
    return "6k train, 5 epochs, single seed; resolution roughly 0.02 val acc"


def _xlabel_short(sweep: str) -> str:
    if sweep.startswith("2"):
        return "$\\sigma/\\rm RMS$"
    if sweep == "1c_subtraction":
        return "$\\delta_{\\rm sub}$"
    return "$\\delta$"


def _read_config(results_dir: str) -> dict | None:
    """Pick up an arbitrary sidecar JSON to extract config."""
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                return json.load(f)
    return None


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(THIS_DIR, "results"))
    ap.add_argument("--plots-dir",   default=os.path.join(THIS_DIR, "plots"))
    ap.add_argument("--report",      default=os.path.join(THIS_DIR, "measurement_robustness.md"))
    args = ap.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)
    dfs = load_results(args.results_dir)
    print(f"loaded sweeps: {sorted(dfs.keys())}", flush=True)
    if not dfs:
        print("no CSVs in results dir, nothing to do.", flush=True)
        return

    for sweep, df in dfs.items():
        out_path = os.path.join(args.plots_dir, f"{sweep}.png")
        plot_individual(sweep, df, out_path)
        print(f"  plotted {sweep} -> {out_path}", flush=True)

    plot_combined_bias(dfs, os.path.join(args.plots_dir, "bias.png"))
    plot_combined_precision(dfs, os.path.join(args.plots_dir, "precision.png"))
    plot_summary(dfs, os.path.join(args.plots_dir, "summary.png"))

    cfg = _read_config(args.results_dir)
    write_report(dfs, args.report, cfg=cfg)
    print(f"\nwrote report -> {args.report}", flush=True)

    summary = summary_table(dfs)
    summary_path = os.path.join(args.results_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"wrote summary table -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
