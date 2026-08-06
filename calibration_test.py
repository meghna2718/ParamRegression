#!/usr/bin/env python
"""
calibration_test.py
=====================
Standalone uncertainty-calibration check via MC-Dropout (Gal & Ghahramani,
2016). This is deliberately a SEPARATE script from train_trackformer.py /
analyze_results.py:
  - it does not change the training loop, loss, architecture, or config
  - it does not modify or overwrite model_best.pt / model_last.pt
  - it only requires the run's --dropout to be > 0 (the default is 0.1, so
    this works on runs you've already trained without retraining anything)

What it does: normally dropout is OFF at eval time (model.eval()). Here we
keep the model in eval() for everything except the Dropout layers, which we
force back into train() mode. Running the same input through the network N
times then gives N slightly different predictions per track (because a
different random subset of units is dropped each pass) -- the spread of
that mini-ensemble is used as a predictive uncertainty estimate. We then
check calibration: if the model says "68% confidence interval", does the
truth actually fall inside that interval ~68% of the time on the test set?

Usage:
    python calibration_test.py --run-dir runs/exp1_posenc_none_seed0 --n-mc 30

Writes (into <run-dir>/calibration/):
    calibration_results.json
    plot_reliability_diagram.png   -- nominal vs. empirical coverage
    plot_uncertainty_vs_error.png  -- does higher predicted spread track
                                       higher actual error? (sharpness check)
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
import torch.nn as nn
from scipy.stats import norm
from torch.utils.data import DataLoader

from train_trackformer import PARAM_NAMES, TrackDataset
from analyze_results import load_model
from plot_style import CATEGORICAL, PALETTE, apply_style

apply_style()

CONFIDENCE_LEVELS = [0.5, 0.68, 0.8, 0.9, 0.95]


def enable_mc_dropout(model):
    n_dropout = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            n_dropout += 1
    return n_dropout


@torch.no_grad()
def mc_predict(model, loader, device, n_mc):
    """Returns preds [n_mc, N, n_params] and truths [N, n_params]."""
    all_truths = None
    all_preds = []
    for _ in range(n_mc):
        batch_preds = []
        truths = []
        for x, mask, y in loader:
            batch_preds.append(model(x.to(device), mask.to(device)).cpu().numpy())
            truths.append(y.numpy())
        all_preds.append(np.vstack(batch_preds))
        if all_truths is None:
            all_truths = np.vstack(truths)
    return np.stack(all_preds, axis=0), all_truths


def angular_diff(a, b):
    """a - b wrapped to [-pi, pi]."""
    d = a - b
    return np.remainder(d + np.pi, 2 * np.pi) - np.pi


def summarize_mc(preds, phi_idx):
    """Mean/std per track per param, handling phi's circularity via a
    circular mean (caveat: still an approximation for large MC spread)."""
    mean = np.median(preds, axis=0)  # robust to occasional dropout outlier passes
    if phi_idx is not None:
        sin_mean = np.mean(np.sin(preds[:, :, phi_idx]), axis=0)
        cos_mean = np.mean(np.cos(preds[:, :, phi_idx]), axis=0)
        mean[:, phi_idx] = np.arctan2(sin_mean, cos_mean)
        # circular std estimate
        R = np.sqrt(sin_mean ** 2 + cos_mean ** 2).clip(1e-9, 1)
        std_phi = np.sqrt(-2 * np.log(R))
    std = np.std(preds, axis=0)
    if phi_idx is not None:
        std[:, phi_idx] = std_phi
    return mean, std


def compute_calibration(mean, std, truth, phi_idx):
    n_tracks, n_params = mean.shape
    results = {}
    for i, name in enumerate(PARAM_NAMES):
        if i == phi_idx:
            err = np.abs(angular_diff(truth[:, i], mean[:, i]))
        else:
            err = np.abs(truth[:, i] - mean[:, i])
        sigma = np.maximum(std[:, i], 1e-9)
        z = err / sigma  # "how many predicted sigmas away is the truth"

        coverage = {}
        for conf in CONFIDENCE_LEVELS:
            z_thresh = norm.ppf((1 + conf) / 2)
            empirical = float((z <= z_thresh).mean())
            coverage[str(conf)] = empirical

        # Sharpness/correlation check: does higher predicted sigma track
        # higher actual error? Spearman-style rank correlation via numpy.
        rank_corr = float(np.corrcoef(np.argsort(np.argsort(sigma)), np.argsort(np.argsort(err)))[0, 1])

        results[name] = {
            "coverage": coverage,
            "mean_predicted_sigma": float(np.mean(sigma)),
            "mean_abs_error": float(np.mean(err)),
            "sigma_error_rank_correlation": rank_corr,
        }
    return results


def plot_reliability_diagram(results, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#334155", linewidth=1.2, label="Perfect calibration")
    for i, name in enumerate(PARAM_NAMES):
        nominal = CONFIDENCE_LEVELS
        empirical = [results[name]["coverage"][str(c)] for c in CONFIDENCE_LEVELS]
        ax.plot(nominal, empirical, marker="o", label=name, color=CATEGORICAL[i % len(CATEGORICAL)], linewidth=2)
    ax.set_xlabel("Nominal confidence level")
    ax.set_ylabel("Empirical coverage (fraction of truths inside interval)")
    ax.set_title("MC-Dropout calibration: reliability diagram")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_reliability_diagram.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {out_dir / 'plot_reliability_diagram.png'}")


def plot_uncertainty_vs_error(mean, std, truth, phi_idx, out_dir):
    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(4 * len(PARAM_NAMES), 4.2))
    for i, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        if i == phi_idx:
            err = np.abs(angular_diff(truth[:, i], mean[:, i]))
        else:
            err = np.abs(truth[:, i] - mean[:, i])
        ax.scatter(std[:, i], err, s=6, alpha=0.35, color=PALETTE["ml"])
        ax.set_xlabel("Predicted sigma (MC-Dropout)")
        if i == 0:
            ax.set_ylabel("Actual |error|")
        ax.set_title(name)
    fig.suptitle("Predicted uncertainty vs. actual error (sharpness check)", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_uncertainty_vs_error.png", dpi=200)
    plt.close(fig)
    print(f"Wrote {out_dir / 'plot_uncertainty_vs_error.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-mc", type=int, default=30, help="Number of MC-Dropout stochastic forward passes")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "calibration"
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(run_dir, device)

    if cfg.get("dropout", 0.0) == 0.0 and cfg.get("input_dropout", 0.0) == 0.0:
        print(
            "WARNING: this run was trained with dropout=0 and input_dropout=0. "
            "MC-Dropout has no stochasticity to sample from -- every pass will "
            "be identical and predicted sigma will be ~0 everywhere. This test "
            "is not meaningful for this run; retrain with --dropout > 0 if you "
            "want calibration results (unrelated to any other pipeline change)."
        )

    n_dropout = enable_mc_dropout(model)
    print(f"Enabled MC-Dropout on {n_dropout} Dropout module(s); running {args.n_mc} stochastic passes...")

    test_df = pd.read_pickle(run_dir / "test_data.pkl")
    ds = TrackDataset(
        test_df["hits_sequence"], targets=test_df[PARAM_NAMES].values.astype(np.float32),
        max_hits=cfg["max_hits"], input_dim=len(cfg["features"]),
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    preds, truth = mc_predict(model, loader, device, args.n_mc)  # preds: [n_mc, N, n_params]
    phi_idx = PARAM_NAMES.index("phi")
    mean, std = summarize_mc(preds, phi_idx)

    results = compute_calibration(mean, std, truth, phi_idx)
    with open(out_dir / "calibration_results.json", "w") as f:
        json.dump({"n_mc": args.n_mc, "n_test_tracks": len(test_df), "per_param": results}, f, indent=2)

    print("\n" + "-" * 90)
    print(f"{'Parameter':<10} | " + " | ".join(f"cov@{c}" for c in CONFIDENCE_LEVELS) + " | sigma-err corr")
    print("-" * 90)
    for name in PARAM_NAMES:
        cov_str = " | ".join(f"{results[name]['coverage'][str(c)]:.3f}" for c in CONFIDENCE_LEVELS)
        print(f"{name:<10} | {cov_str} | {results[name]['sigma_error_rank_correlation']:.3f}")
    print("-" * 90)
    print(
        "Reading this: coverage@0.68 should be close to 0.68 if well calibrated. "
        "Well below -> model is overconfident (intervals too narrow). "
        "Well above -> underconfident (intervals too wide). "
        "sigma-err correlation should be positive -- the model should be more "
        "uncertain exactly where it's more wrong."
    )

    plot_reliability_diagram(results, out_dir)
    plot_uncertainty_vs_error(mean, std, truth, phi_idx, out_dir)
    print(f"\nAll calibration artifacts in {out_dir}/ -- model_best.pt was not modified.")


if __name__ == "__main__":
    main()
