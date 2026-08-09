#!/usr/bin/env python
"""
analyze_results.py
===================
Post-processing for a run produced by train_trackformer.py:
  - loss curve (total + per-parameter, train vs val)
  - permutation feature importance on the held-out test set
  - ML vs ACTS resolution comparison bar chart (if the run used --eval-acts)

Usage:
    python analyze_results.py --run-dir runs/exp1

Reads:  runs/exp1/config.json, model_best.pt, test_data.pkl,
        training_history.json, [acts_comparison.json]
Writes: runs/exp1/plot_loss_curve.png
        runs/exp1/plot_feature_importance.png
        runs/exp1/feature_importance.json
        runs/exp1/plot_ml_vs_acts.png   (if acts_comparison.json present)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_trackformer import PARAM_NAMES, TrackDataset, TrackTransformer
from plot_style import CATEGORICAL, PALETTE, apply_style, style_bars

apply_style()


def robust_resolution(residuals):
    q75, q25 = np.percentile(residuals, [75, 25])
    return (q75 - q25) / 1.349


def load_model(run_dir, device):
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    angle_indices = tuple(sorted(PARAM_NAMES.index(p) for p in cfg["angle_params"]))
    model = TrackTransformer(
        input_dim=len(cfg["features"]),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        input_dropout=cfg["input_dropout"],
        max_hits=cfg["max_hits"],
        pos_encoding=cfg["pos_encoding"],
        head_mode=cfg["head_mode"],
        angle_param_indices=angle_indices,
        num_params=len(PARAM_NAMES),
        enforce_unit_circle=not cfg["no_unit_circle"],
    ).to(device)
    ckpt = torch.load(run_dir / "model_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded model_best.pt from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.5f})")
    return model, cfg


def plot_loss_curve(run_dir):
    with open(run_dir / "training_history.json") as f:
        hist = json.load(f)
    epochs = range(1, len(hist["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, hist["train_loss"], label="train", color=PALETTE["train"], linewidth=2.2)
    axes[0].plot(epochs, hist["val_loss"], label="val", color=PALETTE["val"], linewidth=2.2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Aggregate loss")
    axes[0].set_title("Training loss (geometric mean over params)")
    axes[0].legend()

    val_param = np.array(hist["val_param_loss"])  # [epochs, n_params]
    for i, name in enumerate(PARAM_NAMES):
        axes[1].plot(epochs, val_param[:, i], label=name, color=CATEGORICAL[i % len(CATEGORICAL)], linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val loss (per parameter)")
    axes[1].set_title("Per-parameter validation loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(run_dir / "plot_loss_curve.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {run_dir / 'plot_loss_curve.png'}")


def permutation_importance(model, test_df, cfg, device, n_repeats=3, seed=0):
    features = cfg["features"]
    ds = TrackDataset(
        test_df["hits_sequence"],
        targets=test_df[PARAM_NAMES].values.astype(np.float32),
        max_hits=cfg["max_hits"],
        input_dim=len(features),
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    rng = np.random.default_rng(seed)

    def score_loader(permute_idx=None):
        """Mean, over parameters, of (resolution / baseline_resolution) - 1."""
        all_preds, all_truths = [], []
        with torch.no_grad():
            for x, mask, y in loader:
                if permute_idx is not None:
                    perm = torch.tensor(rng.permutation(x.size(0)))
                    x[:, :, permute_idx] = x[perm][:, :, permute_idx]
                preds = model(x.to(device), mask.to(device)).cpu().numpy()
                all_preds.append(preds)
                all_truths.append(y.numpy())
        preds = np.vstack(all_preds)
        truths = np.vstack(all_truths)
        res = {}
        for i, name in enumerate(PARAM_NAMES):
            if name == "phi":
                r = np.remainder(preds[:, i] - truths[:, i] + np.pi, 2 * np.pi) - np.pi
            else:
                r = preds[:, i] - truths[:, i]
            res[name] = robust_resolution(r)
        return res

    baseline = score_loader(None)
    importances = {}
    for idx, fname in enumerate(features):
        degradations = []
        for _ in range(n_repeats):
            permuted = score_loader(idx)
            rel = np.mean(
                [(permuted[p] - baseline[p]) / max(abs(baseline[p]), 1e-9) for p in PARAM_NAMES]
            )
            degradations.append(rel)
        importances[fname] = float(np.mean(degradations))

    return importances, baseline


def plot_feature_importance(importances, run_dir):
    items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE["ml"] if v > 0 else PALETTE["neutral"] for v in vals]
    bars = ax.barh(names, vals, color=colors)
    style_bars(bars)
    ax.set_xlabel("Mean relative resolution degradation when shuffled")
    ax.set_title("Permutation feature importance (higher = more important)")
    ax.axvline(0, color="#334155", linewidth=1)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(run_dir / "plot_feature_importance.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {run_dir / 'plot_feature_importance.png'}")


def plot_pred_vs_true(run_dir, test_df):
    """2D density heatmap of ML prediction vs. ground truth, one panel per
    parameter, using ALL test tracks (not just the ACTS-matched subset
    comparison_df.pkl is restricted to -- more statistics, and doesn't
    require --eval-acts). A raw scatter would overplot badly at this many
    points, hence a density heatmap (hist2d) instead. The y=x reference line
    makes systematic bias/scale issues (e.g. a mismatched sign or unit
    convention) visually obvious in a way a single resolution number isn't --
    useful for sanity-checking a surprisingly-good result before trusting it."""
    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(4.3 * len(PARAM_NAMES), 4.3))
    for ax, name in zip(axes, PARAM_NAMES):
        truth = test_df[name].values
        pred = test_df[f"ml_{name}"].values
        lo, hi = np.percentile(np.concatenate([truth, pred]), [0.5, 99.5])
        hb = ax.hist2d(truth, pred, bins=80, range=[[lo, hi], [lo, hi]], cmap="viridis", cmin=1)
        ax.plot([lo, hi], [lo, hi], color="#ef4444", linestyle="--", linewidth=1.3, label="y = x (perfect)")
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(name)
        ax.legend(fontsize=8, loc="upper left")
        fig.colorbar(hb[3], ax=ax, label="tracks", shrink=0.85)
    fig.suptitle(f"Predicted vs. true, all {len(test_df):,} test tracks", y=1.02)
    fig.tight_layout()
    fig.savefig(run_dir / "plot_pred_vs_true.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {run_dir / 'plot_pred_vs_true.png'}")


def plot_qop_sign_check(run_dir, test_df):
    """Zoomed-in view of predicted vs. true qop near zero (i.e. near the
    charge sign boundary, since sign(qop) == sign(charge)) plus the
    misidentification rate broken out into narrow |true_qop| bins. If ML's
    charge misID rate looks suspiciously good (e.g. 0%) in the headline
    number, this is where a real sign-flip problem OR a genuine floor effect
    would actually show up -- 0% averaged over all tracks can still hide a
    real problem concentrated in the lowest-momentum tracks specifically."""
    truth = test_df["qop"].values
    pred = test_df["ml_qop"].values
    misid = np.sign(pred) != np.sign(truth)
    overall_misid_pct = 100 * misid.mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    zoom = np.percentile(np.abs(truth), 15)  # zoom to the lowest-|qop| ~15% of tracks
    m = np.abs(truth) <= zoom
    axes[0].scatter(truth[m], pred[m], s=4, alpha=0.35, color=PALETTE["ml"])
    axes[0].axhline(0, color="#334155", linewidth=1)
    axes[0].axvline(0, color="#334155", linewidth=1)
    axes[0].plot([-zoom, zoom], [-zoom, zoom], color="#ef4444", linestyle="--", linewidth=1.2, label="y = x")
    axes[0].set_xlabel("True qop")
    axes[0].set_ylabel("Predicted qop")
    axes[0].set_title(f"Zoom near qop=0 (lowest {m.sum():,} |qop| tracks) -- sign flips would show as points\n"
                       f"in the wrong (top-left / bottom-right) quadrants")
    axes[0].legend(fontsize=8)

    n_bins = 10
    bin_edges = np.quantile(np.abs(truth), np.linspace(0, 1, n_bins + 1))
    bin_idx = np.clip(np.digitize(np.abs(truth), bin_edges[1:-1]), 0, n_bins - 1)
    misid_per_bin = [100 * misid[bin_idx == i].mean() if (bin_idx == i).any() else np.nan for i in range(n_bins)]
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]
    bars = axes[1].bar(range(n_bins), misid_per_bin, color=PALETTE["ml"])
    style_bars(bars)
    axes[1].set_xticks(range(n_bins))
    axes[1].set_xticklabels([f"{c:.3f}" for c in bin_centers], rotation=45, ha="right")
    axes[1].set_xlabel("|true qop| (bin center, equal-count bins, low -> high momentum)")
    axes[1].set_ylabel("Charge misID rate (%)")
    axes[1].set_title(f"Overall: {overall_misid_pct:.3f}% ({int(misid.sum())}/{len(misid)} tracks)")

    fig.tight_layout()
    fig.savefig(run_dir / "plot_qop_sign_check.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {run_dir / 'plot_qop_sign_check.png'} "
          f"(overall charge misID: {overall_misid_pct:.3f}%, {int(misid.sum())}/{len(misid)} tracks)")


def plot_ml_vs_acts(run_dir):
    path = run_dir / "acts_comparison.json"
    if not path.exists():
        print("No acts_comparison.json found (run was not launched with --eval-acts); skipping.")
        return
    with open(path) as f:
        comp = json.load(f)

    names = [n for n in PARAM_NAMES if n in comp]
    ml_vals = [comp[n]["ml"] for n in names]
    acts_vals = [comp[n]["acts"] for n in names]
    # Bootstrap 68% CIs -> symmetric-ish error bars around the point estimate,
    # reshaped from per-name lists into [2, n] arrays for errorbar-style yerr.
    has_ci = all("ml_ci68" in comp[n] and "acts_ci68" in comp[n] for n in names)
    if has_ci:
        ml_yerr = np.array([[comp[n]["ml"] - comp[n]["ml_ci68"][0] for n in names],
                             [comp[n]["ml_ci68"][1] - comp[n]["ml"] for n in names]])
        acts_yerr = np.array([[comp[n]["acts"] - comp[n]["acts_ci68"][0] for n in names],
                               [comp[n]["acts_ci68"][1] - comp[n]["acts"] for n in names]])
    else:
        ml_yerr = acts_yerr = None
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - 0.175, acts_vals, 0.35, yerr=acts_yerr, capsize=3,
                label="ACTS (classical)", color=PALETTE["acts"], alpha=0.85)
    b2 = ax.bar(x + 0.175, ml_vals, 0.35, yerr=ml_yerr, capsize=3,
                label="ML (this model)", color=PALETTE["ml"], alpha=0.85)
    style_bars(b1); style_bars(b2)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Resolution (IQR/1.349, physical units)")
    ax.set_title("ML vs ACTS resolution per parameter (error bars: bootstrap 68% CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "plot_ml_vs_acts.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {run_dir / 'plot_ml_vs_acts.png'}")

    if "match_rate" in comp:
        print(
            f"\nACTS match rate: {comp['n_matched_tracks']}/{comp['n_test_tracks']} "
            f"tracks ({comp['match_rate']:.1%}) -- resolution numbers are conditioned on this subset."
        )
    for n in names:
        if "wilcoxon_pvalue" in comp[n]:
            sig = "significant (p<0.05)" if comp[n]["wilcoxon_pvalue"] < 0.05 else "not significant"
            print(
                f"  {n:<8} ML outlier {comp[n]['ml_outlier_pct']:.2f}% | "
                f"ACTS outlier {comp[n]['acts_outlier_pct']:.2f}% | "
                f"Wilcoxon p={comp[n]['wilcoxon_pvalue']:.4g} ({sig})"
            )


def robust_rmse(residuals):
    return float(np.sqrt(np.mean(np.asarray(residuals) ** 2)))


def plot_error_vs_pt(run_dir, comp, pt_bins=(1.0, 2.0, 5.0, 10.0, 50.0)):
    labels = [f"{pt_bins[i]}-{pt_bins[i+1]} GeV" for i in range(len(pt_bins) - 1)]
    comp = comp.copy()
    comp["pt_bin"] = pd.cut(comp["pt"], bins=pt_bins, labels=labels)

    rows = []
    for b, g in comp.groupby("pt_bin", observed=False):
        if len(g) == 0:
            continue
        rows.append({
            "bin": b,
            "ml_rmse": robust_rmse(g["ml_qop"] - g["qop"]),
            "acts_rmse": robust_rmse(g["acts_qop"] - g["qop"]),
            "n": len(g),
        })
    if not rows:
        print("No tracks after pT binning; skipping plot_error_vs_pt.")
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(df))
    b1 = ax.bar(x - 0.175, df["acts_rmse"], 0.35, label="ACTS (classical)", color=PALETTE["acts"], alpha=0.85)
    b2 = ax.bar(x + 0.175, df["ml_rmse"], 0.35, label="ML (this model)", color=PALETTE["ml"], alpha=0.85)
    style_bars(b1); style_bars(b2)
    ax.set_xticks(x)
    ax.set_xticklabels(df["bin"].astype(str))
    ax.set_ylabel(r"RMSE for $q/p$ (1/GeV)")
    ax.set_xlabel(r"True transverse momentum $p_T$")
    ax.set_title(r"Tracking error vs. particle $p_T$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "plot_error_vs_pt.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {run_dir / 'plot_error_vs_pt.png'}")


def plot_error_vs_hits(run_dir, comp, min_group_size=10):
    rows = []
    for hits, g in comp.groupby("calculated_hits"):
        if len(g) < min_group_size:
            continue
        rows.append({
            "hits": hits,
            "ml_rmse": robust_rmse(g["ml_d0"] - g["d0"]),
            "acts_rmse": robust_rmse(g["acts_d0"] - g["d0"]),
        })
    if not rows:
        print("No hit-count bins with enough tracks; skipping plot_error_vs_hits.")
        return
    df = pd.DataFrame(rows).sort_values("hits")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(df["hits"], df["acts_rmse"], marker="o", label="ACTS (classical)", color=PALETTE["acts"], linewidth=2.2, markersize=7)
    ax.plot(df["hits"], df["ml_rmse"], marker="s", label="ML (this model)", color=PALETTE["ml"], linewidth=2.2, markersize=7)
    ax.set_xticks(df["hits"])
    ax.set_ylabel(r"RMSE for $d_0$ (mm)")
    ax.set_xlabel("Number of detector hits")
    ax.set_title("Robustness to missing data: impact parameter vs. hit count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "plot_error_vs_hits.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {run_dir / 'plot_error_vs_hits.png'}")


def plot_error_vs_eta(run_dir, comp, eta_bins=(-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)):
    """Detector acceptance / material budget is strongly eta-dependent -- a
    standard tracking-physics breakdown that pT- and hit-count-binning alone
    don't capture."""
    if "eta" not in comp.columns:
        print("No 'eta' column in comparison_df; skipping plot_error_vs_eta.")
        return
    labels = [f"{eta_bins[i]} to {eta_bins[i+1]}" for i in range(len(eta_bins) - 1)]
    comp = comp.copy()
    comp["eta_bin"] = pd.cut(comp["eta"], bins=eta_bins, labels=labels)

    rows = []
    for b, g in comp.groupby("eta_bin", observed=False):
        if len(g) == 0:
            continue
        rows.append({
            "bin": b,
            "ml_rmse": robust_rmse(g["ml_d0"] - g["d0"]),
            "acts_rmse": robust_rmse(g["acts_d0"] - g["d0"]),
            "n": len(g),
        })
    if not rows:
        print("No tracks after eta binning; skipping plot_error_vs_eta.")
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(range(len(df)), df["acts_rmse"], marker="o", label="ACTS (classical)", color=PALETTE["acts"], linewidth=2.2, markersize=7)
    ax.plot(range(len(df)), df["ml_rmse"], marker="s", label="ML (this model)", color=PALETTE["ml"], linewidth=2.2, markersize=7)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["bin"], rotation=20)
    ax.set_ylabel(r"RMSE for $d_0$ (mm)")
    ax.set_xlabel(r"True pseudorapidity ($\eta$)")
    ax.set_title(r"Impact parameter resolution vs. particle $\eta$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "plot_error_vs_eta.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {run_dir / 'plot_error_vs_eta.png'}")


def plot_fractional_error(run_dir, comp):
    ml_frac = (comp["ml_qop"] - comp["qop"]) / np.abs(comp["qop"])
    acts_frac = (comp["acts_qop"] - comp["qop"]) / np.abs(comp["qop"])
    q1, q3 = np.percentile(ml_frac.dropna(), [2, 98])
    mask = (ml_frac >= q1) & (ml_frac <= q3) & (acts_frac >= q1) & (acts_frac <= q3)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(acts_frac[mask], bins=80, alpha=0.55, color=PALETTE["acts"], label="ACTS (classical)", density=True)
    ax.hist(ml_frac[mask], bins=80, alpha=0.6, color=PALETTE["ml"], label="ML (this model)", density=True)
    ax.axvline(0, color="#334155", linestyle="--", linewidth=1.6)
    ax.set_ylabel("Density")
    ax.set_xlabel("Fractional error [(pred - true) / true]")
    ax.set_title(r"Momentum ($q/p$) fractional error distributions")
    ax.legend()
    vals = ax.get_xticks()
    ax.set_xticklabels(["{:,.0%}".format(v) for v in vals])
    fig.tight_layout()
    fig.savefig(run_dir / "plot_fractional_error.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {run_dir / 'plot_fractional_error.png'}")


def report_speed(run_dir):
    path = run_dir / "speed_benchmark.json"
    if not path.exists():
        print("No speed_benchmark.json found; skipping.")
        return
    with open(path) as f:
        speed = json.load(f)
    print(
        f"\nInference speed ({speed['device']}): {speed['tracks_per_second']:.0f} tracks/s "
        f"({speed['ms_per_track']:.4f} ms/track, {speed['total_tracks']} tracks)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--skip-importance", action="store_true", help="Skip permutation importance (slow-ish for many features)")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (run_dir / "training_history.json").exists():
        plot_loss_curve(run_dir)
    else:
        # Expected for a run evaluated via evaluate_checkpoint.py after being
        # killed mid-training (e.g. a SLURM --time limit) BEFORE the
        # checkpoint_latest.pt / --resume feature existed for it -- per-epoch
        # history was never persisted for such a run, so there is genuinely
        # nothing to plot here. Every other analysis below only needs the
        # trained model + test set, not training history, so it still runs.
        print("No training_history.json found -- skipping loss-curve plot "
              "(expected for a run recovered via evaluate_checkpoint.py from "
              "a run that never finished its training loop).")
    report_speed(run_dir)

    model, cfg = load_model(run_dir, device)
    test_df = pd.read_pickle(run_dir / "test_data.pkl")

    if not args.skip_importance:
        importances, baseline = permutation_importance(model, test_df, cfg, device)
        with open(run_dir / "feature_importance.json", "w") as f:
            json.dump({"importances": importances, "baseline_resolution": baseline}, f, indent=2)
        plot_feature_importance(importances, run_dir)
        print("\nFeature importance (higher = more important):")
        for name, val in sorted(importances.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {name:<12} {val:+.4f}")

    preds_path = run_dir / "test_data_with_predictions.pkl"
    if preds_path.exists():
        test_preds_df = pd.read_pickle(preds_path)
        plot_pred_vs_true(run_dir, test_preds_df)
        plot_qop_sign_check(run_dir, test_preds_df)
    else:
        print(f"No {preds_path.name} found; skipping pred-vs-true heatmap and qop sign-check plots.")

    plot_ml_vs_acts(run_dir)

    comp_path = run_dir / "comparison_df.pkl"
    if comp_path.exists():
        comp = pd.read_pickle(comp_path)
        plot_error_vs_pt(run_dir, comp)
        plot_error_vs_eta(run_dir, comp)
        plot_error_vs_hits(run_dir, comp)
        plot_fractional_error(run_dir, comp)
    else:
        print(
            "No comparison_df.pkl found (run was not launched with --eval-acts); "
            "skipping pT-binned, hit-count-binned, and fractional-error plots."
        )

    print(f"\nAll plots/artifacts in {run_dir}/")


if __name__ == "__main__":
    main()
