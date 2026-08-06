#!/usr/bin/env python
"""
compare_runs.py
=================
Aggregate several train_trackformer.py runs (e.g. the array tasks from
exp1_posenc.slurm or exp2_headmode.slurm) into one table + one plot, so you
can directly answer "does X make a difference, and which variant is best".

Usage:
    python compare_runs.py \
        --run-dirs runs/exp1_posenc_none runs/exp1_posenc_sinusoidal runs/exp1_posenc_learnable \
        --labels none sinusoidal learnable \
        --out-dir runs/exp1_posenc_comparison

Reads test_resolutions.json (+ final val_loss from training_history.json) from
each run dir. Writes:
    <out-dir>/comparison_table.csv
    <out-dir>/plot_comparison_resolution.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_trackformer import PARAM_NAMES
from plot_style import CATEGORICAL, PALETTE, apply_style, style_bars

apply_style()


def load_run(run_dir: Path, label: str) -> dict:
    row = {"label": label, "run_dir": str(run_dir)}

    res_path = run_dir / "test_resolutions.json"
    if res_path.exists():
        with open(res_path) as f:
            res = json.load(f)
        for name in PARAM_NAMES:
            row[f"resolution_{name}"] = res.get(name)
    else:
        print(f"WARNING: {res_path} missing -- has this run finished?")

    hist_path = run_dir / "training_history.json"
    if hist_path.exists():
        with open(hist_path) as f:
            hist = json.load(f)
        row["final_val_loss"] = hist["val_loss"][-1]
        row["best_val_loss"] = min(hist["val_loss"])

    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        row["best_epoch"] = summary.get("best_epoch")
        row["epochs_configured"] = summary.get("epochs_configured")
        row["training_minutes"] = summary.get("training_seconds", 0) / 60
        row["minutes_to_best_epoch"] = summary.get("seconds_to_reach_best_epoch", 0) / 60
        row["total_minutes"] = summary.get("total_seconds", 0) / 60

    speed_path = run_dir / "speed_benchmark.json"
    if speed_path.exists():
        with open(speed_path) as f:
            row["tracks_per_second"] = json.load(f)["tracks_per_second"]

    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        row["pos_encoding"] = cfg.get("pos_encoding")
        row["head_mode"] = cfg.get("head_mode")
        row["n_params"] = cfg.get("n_params")
        row["train_events"] = cfg.get("train_events")

    acts_path = run_dir / "acts_comparison.json"
    if acts_path.exists():
        with open(acts_path) as f:
            acts_comp = json.load(f)
        if "charge_misid_pct" in acts_comp:
            row["charge_misid_ml_pct"] = acts_comp["charge_misid_pct"]["ml"]
            row["charge_misid_acts_pct"] = acts_comp["charge_misid_pct"]["acts"]
        row["acts_match_rate"] = acts_comp.get("match_rate")

    return row


def aggregate_by_label(df: pd.DataFrame) -> pd.DataFrame:
    """If --labels has repeats (multiple seeds per config), collapse to
    mean +/- std per label. With no repeats, std is 0/NaN and this is a
    no-op reshuffle -- safe to always call."""
    numeric_cols = [c for c in df.columns if c.startswith("resolution_") or c in
                    ("final_val_loss", "best_val_loss", "tracks_per_second", "n_params",
                     "train_events", "charge_misid_ml_pct", "charge_misid_acts_pct", "acts_match_rate",
                     "best_epoch", "epochs_configured", "training_minutes", "minutes_to_best_epoch",
                     "total_minutes")]
    agg = df.groupby("label", sort=False)[numeric_cols].agg(["mean", "std", "count"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()
    # preserve first-seen order of labels rather than alphabetical
    order = list(dict.fromkeys(df["label"]))
    agg["_order"] = agg["label"].apply(order.index)
    return agg.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def plot_comparison(agg: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(4 * len(PARAM_NAMES), 5), sharex=False)
    if len(PARAM_NAMES) == 1:
        axes = [axes]
    x = np.arange(len(agg))
    n_seeds = int(agg.get("resolution_" + PARAM_NAMES[0] + "_count", pd.Series([1])).max())
    for ax, name in zip(axes, PARAM_NAMES):
        mean_col, std_col = f"resolution_{name}_mean", f"resolution_{name}_std"
        if mean_col not in agg.columns:
            continue
        colors = [CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(agg))]
        yerr = agg[std_col].fillna(0) if std_col in agg.columns else None
        bars = ax.bar(x, agg[mean_col], yerr=yerr, capsize=4, color=colors, alpha=0.9)
        style_bars(bars)
        best_idx = agg[mean_col].abs().idxmin()
        bars[list(agg.index).index(best_idx)].set_color(PALETTE["best"])
        ax.set_xticks(x)
        ax.set_xticklabels(agg["label"], rotation=25, ha="right")
        ax.set_title(name)
    axes[0].set_ylabel("Resolution (IQR/1.349, physical units)")
    suffix = f" (mean +/- std, n={n_seeds} seed{'s' if n_seeds != 1 else ''})" if n_seeds > 1 else ""
    fig.suptitle(f"Test-set resolution across variants{suffix} -- green = best per parameter", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_comparison_resolution.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {out_dir / 'plot_comparison_resolution.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument(
        "--labels", nargs="+", required=True,
        help="One label per --run-dirs entry. Repeat a label across multiple "
             "--run-dirs entries (different --seed per training run) to treat "
             "them as replicate seeds -- mean +/- std is then reported instead "
             "of a single noisy point estimate.",
    )
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    assert len(args.run_dirs) == len(args.labels), "--run-dirs and --labels must be the same length"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [load_run(Path(rd), label) for rd, label in zip(args.run_dirs, args.labels)]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "comparison_table_raw.csv", index=False)

    agg = aggregate_by_label(df)
    agg.to_csv(out_dir / "comparison_table.csv", index=False)

    print("\n" + agg.to_string(index=False))

    if any(f"resolution_{name}_mean" in agg.columns for name in PARAM_NAMES):
        plot_comparison(agg, out_dir)

    # Simple verdict: which variant wins on the most parameters (by mean)
    win_counts = {}
    for name in PARAM_NAMES:
        col = f"resolution_{name}_mean"
        if col in agg.columns and agg[col].notna().any():
            winner = agg.loc[agg[col].abs().idxmin(), "label"]
            win_counts[winner] = win_counts.get(winner, 0) + 1
    if win_counts:
        print("\nBest (mean) resolution per parameter, tallied by variant:")
        for label, count in sorted(win_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {label:<15} wins on {count}/{len(PARAM_NAMES)} parameters")

    print(f"\nWrote {out_dir / 'comparison_table.csv'} (aggregated) and comparison_table_raw.csv (per-run)")


if __name__ == "__main__":
    main()
