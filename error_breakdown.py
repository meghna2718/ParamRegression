#!/usr/bin/env python
"""
error_breakdown.py
=====================
"Which kind of track does this model do well/badly on?" -- breaks down
ML vs ACTS error by particle species and by a scattering proxy, on top of
the pT- and hit-count-binned plots analyze_results.py already produces.

Requires a run trained with --eval-acts (needs comparison_df.pkl).

Usage:
    python error_breakdown.py --run-dir runs/exp1_posenc_none_seed0
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import CATEGORICAL, PALETTE, apply_style, style_bars

apply_style()

# PDG particle codes -> readable names, for the subset likely to show up as
# primary charged tracks in a ttbar sample.
PDG_NAMES = {
    11: "electron", -11: "positron",
    13: "muon-", -13: "muon+",
    211: "pion+", -211: "pion-",
    321: "kaon+", -321: "kaon-",
    2212: "proton", -2212: "antiproton",
}


def robust_rmse(residuals):
    return float(np.sqrt(np.mean(np.asarray(residuals) ** 2)))


def robust_resolution(residuals):
    q75, q25 = np.percentile(residuals, [75, 25])
    return (q75 - q25) / 1.349


def plot_by_particle_type(run_dir, comp, min_group_size=15):
    if "pdg_id" not in comp.columns:
        print("No 'pdg_id' column; skipping particle-type breakdown.")
        return
    comp = comp.copy()
    comp["species"] = comp["pdg_id"].map(lambda p: PDG_NAMES.get(int(p), f"pdg {int(p)}"))
    # Group opposite-charge pairs together for readability (electron/positron
    # etc. should have symmetric tracking performance; splitting them just
    # adds noise unless you specifically care about charge asymmetry).
    comp["species_group"] = comp["pdg_id"].abs().map(
        lambda p: {11: "electron", 13: "muon", 211: "pion", 321: "kaon", 2212: "proton"}.get(int(p), f"pdg {int(p)}")
    )

    rows = []
    for species, g in comp.groupby("species_group"):
        if len(g) < min_group_size:
            continue
        rows.append({
            "species": species,
            "n_tracks": len(g),
            "ml_res_qop": robust_resolution(g["ml_qop"] - g["qop"]),
            "acts_res_qop": robust_resolution(g["acts_qop"] - g["qop"]),
            "ml_res_d0": robust_resolution(g["ml_d0"] - g["d0"]),
            "acts_res_d0": robust_resolution(g["acts_d0"] - g["d0"]),
        })
    if not rows:
        print("No species groups with enough tracks; skipping.")
        return
    df = pd.DataFrame(rows).sort_values("n_tracks", ascending=False)
    df.to_csv(run_dir / "breakdown_by_particle_type.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(df))
    for ax, param, ylabel in [
        (axes[0], "qop", r"Resolution for $q/p$ (1/GeV)"),
        (axes[1], "d0", r"Resolution for $d_0$ (mm)"),
    ]:
        b1 = ax.bar(x - 0.175, df[f"acts_res_{param}"], 0.35, label="ACTS", color=PALETTE["acts"], alpha=0.85)
        b2 = ax.bar(x + 0.175, df[f"ml_res_{param}"], 0.35, label="ML", color=PALETTE["ml"], alpha=0.85)
        style_bars(b1); style_bars(b2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(df["species"], df["n_tracks"])])
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.suptitle("Resolution by particle species", y=1.03)
    fig.tight_layout()
    fig.savefig(run_dir / "plot_breakdown_by_particle_type.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {run_dir / 'plot_breakdown_by_particle_type.png'} and breakdown_by_particle_type.csv")
    print("\nBy particle species:")
    print(df.to_string(index=False))


def plot_by_scattering(run_dir, comp, n_buckets=3):
    if "scatter_n_kinks" not in comp.columns:
        print("No 'scatter_n_kinks' column; skipping scattering breakdown.")
        return
    comp = comp.copy()
    # Bucket by number of direction-reversal "kinks" in the hit sequence --
    # a proxy for how non-helical (scattered) the track is. Most tracks will
    # have 0; separate 0 / 1 / 2+ rather than quantile-cutting a skewed count.
    def bucket(n):
        if n == 0:
            return "0 kinks (clean)"
        elif n == 1:
            return "1 kink"
        return "2+ kinks (scattered)"

    comp["scatter_bucket"] = comp["scatter_n_kinks"].map(bucket)
    order = ["0 kinks (clean)", "1 kink", "2+ kinks (scattered)"]

    rows = []
    for b in order:
        g = comp[comp["scatter_bucket"] == b]
        if len(g) < 10:
            continue
        rows.append({
            "bucket": b,
            "n_tracks": len(g),
            "ml_res_qop": robust_resolution(g["ml_qop"] - g["qop"]),
            "acts_res_qop": robust_resolution(g["acts_qop"] - g["qop"]),
        })
    if not rows:
        print("No scattering buckets with enough tracks; skipping.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "breakdown_by_scattering.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(df))
    b1 = ax.bar(x - 0.175, df["acts_res_qop"], 0.35, label="ACTS", color=PALETTE["acts"], alpha=0.85)
    b2 = ax.bar(x + 0.175, df["ml_res_qop"], 0.35, label="ML", color=PALETTE["ml"], alpha=0.85)
    style_bars(b1); style_bars(b2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(df["bucket"], df["n_tracks"])])
    ax.set_ylabel(r"Resolution for $q/p$ (1/GeV)")
    ax.set_title("Resolution vs. track scattering (direction reversals in hit sequence)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "plot_breakdown_by_scattering.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {run_dir / 'plot_breakdown_by_scattering.png'} and breakdown_by_scattering.csv")
    print("\nBy scattering bucket:")
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    comp_path = run_dir / "comparison_df.pkl"
    if not comp_path.exists():
        print(f"No comparison_df.pkl in {run_dir} -- run this run with --eval-acts first.")
        return
    comp = pd.read_pickle(comp_path)

    plot_by_particle_type(run_dir, comp)
    plot_by_scattering(run_dir, comp)
    print(f"\nAll breakdown artifacts in {run_dir}/")


if __name__ == "__main__":
    main()
