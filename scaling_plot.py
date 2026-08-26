#!/usr/bin/env python
"""
scaling_plot.py
Log-log scaling-law plots: best_val_loss and per-parameter resolution vs.
(1) model size (n_params, from config.json) and (2) training data size
(train_events, from run_summary.json).

Usage:
    python scaling_plot.py \
        --model-sweep-dirs runs/exp4_scale_model_small_v1 runs/exp4_scale_model_medium_v1 \
                            runs/exp3_fix_v3_stability_fix_seed0 runs/exp4_scale_model_large_v1 \
        --data-sweep-dirs  runs/exp4_scale_data_small_v1 runs/exp4_scale_data_medium_v1 \
                            runs/exp3_fix_v3_stability_fix_seed0 runs/exp4_scale_data_large_v1 \
        --out-dir runs/exp4_scaling_plots
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARAM_NAMES = ["d0", "z0", "phi", "theta", "qop"]


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_point(run_dir: Path) -> dict:
    cfg = load_json(run_dir / "config.json")
    summary = load_json(run_dir / "run_summary.json")
    res = load_json(run_dir / "test_resolutions.json")

    row = {
        "run_dir": str(run_dir),
        "n_params": cfg.get("n_params"),
        "d_model": cfg.get("d_model"),
        "num_layers": cfg.get("num_layers"),
        "train_events": cfg.get("train_events"),
        "best_val_loss": summary.get("best_val_loss"),
        "epochs_run": summary.get("epochs_run"),
    }
    for p in PARAM_NAMES:
        row[f"resolution_{p}"] = res.get(p)
    return row


def fit_power_law_label(x, y):
    """Fit log(y) = a*log(x) + b; return a one-line label with the exponent."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None, None
    slope, intercept = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return slope, intercept


def load_acts_ref(acts_ref_dir):
    """ACTS resolution per param, from any one run's acts_comparison.json (same test set -> same value everywhere)."""
    if not acts_ref_dir:
        return {}
    comp = load_json(Path(acts_ref_dir) / "acts_comparison.json")
    return {p: comp[p]["acts"] for p in PARAM_NAMES if p in comp}


def _plot_one_metric(ax, sub, x_col, metric, x_label, color, acts_ref=None):
    ax.plot(sub[x_col], sub[metric], "o-", color=color, linewidth=2, markersize=7, label="ML")
    slope, _ = fit_power_law_label(sub[x_col], sub[metric])
    label = metric.replace("resolution_", "").replace("_", " ")
    subtitle = f"{label}" + (f"  (slope={slope:.2f})" if slope is not None else "")
    ax.set_title(subtitle, fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    # ACTS is not a model -- flat reference line, not a scaling point.
    if acts_ref is not None:
        ax.axhline(acts_ref, color="#d1242f", linestyle="--", linewidth=1.5, label="ACTS")
        ax.legend(fontsize=8)


def plot_sweep(df, x_col, x_label, title, out_path, sweep_name, out_dir, acts_ref=None):
    metrics = ["best_val_loss"] + [f"resolution_{p}" for p in PARAM_NAMES]
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    acts_ref = acts_ref or {}

    # --- Combined 2x3 figure ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, metric, color in zip(axes, metrics, colors):
        sub = df.dropna(subset=[x_col, metric]).sort_values(x_col)
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        param = metric.replace("resolution_", "")
        _plot_one_metric(ax, sub, x_col, metric, x_label, color, acts_ref.get(param))

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    # --- Separate single-metric figures ---
    sep_dir = out_dir / f"{sweep_name}_individual"
    sep_dir.mkdir(exist_ok=True)
    for metric, color in zip(metrics, colors):
        sub = df.dropna(subset=[x_col, metric]).sort_values(x_col)
        if len(sub) == 0:
            continue
        param = metric.replace("resolution_", "")
        fig, ax = plt.subplots(figsize=(6, 5))
        _plot_one_metric(ax, sub, x_col, metric, x_label, color, acts_ref.get(param))
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout()
        sep_path = sep_dir / f"{metric}.png"
        fig.savefig(sep_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote individual plots to {sep_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-sweep-dirs", nargs="+", required=True)
    ap.add_argument("--data-sweep-dirs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--acts-ref-dir", default=None,
                     help="Run dir with acts_comparison.json, for a flat ACTS reference line. Omit to drop it.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acts_ref = load_acts_ref(args.acts_ref_dir)

    model_df = pd.DataFrame([load_point(Path(d)) for d in args.model_sweep_dirs])
    data_df = pd.DataFrame([load_point(Path(d)) for d in args.data_sweep_dirs])

    model_df.to_csv(out_dir / "model_sweep_table.csv", index=False)
    data_df.to_csv(out_dir / "data_sweep_table.csv", index=False)
    print("\n--- Model-size sweep ---")
    print(model_df[["run_dir", "n_params", "best_val_loss", "resolution_qop"]].to_string(index=False))
    print("\n--- Data-size sweep ---")
    print(data_df[["run_dir", "train_events", "best_val_loss", "resolution_qop"]].to_string(index=False))

    plot_sweep(
        model_df, "n_params", "Model size (parameters)",
        "Scaling vs. Model Size", out_dir / "scaling_vs_model_size.png",
        "model_size", out_dir, acts_ref=acts_ref,
    )
    plot_sweep(
        data_df, "train_events", "Training events",
        "Scaling vs. Data Size", out_dir / "scaling_vs_data_size.png",
        "data_size", out_dir, acts_ref=acts_ref,
    )


if __name__ == "__main__":
    main()
