#!/usr/bin/env python
"""
evaluate_checkpoint.py
=======================
Re-evaluates an already-trained model_best.pt without retraining. Rebuilds
the same test set from config.json (deterministic split), runs it through
the checkpoint, and writes the same output files train_trackformer.py would
have written: test_data.pkl, test_data_with_predictions.pkl,
test_resolutions.json, speed_benchmark.json, comparison_light.parquet /
acts_comparison.json (if --eval-acts), run_summary.json.

Use this whenever you want to regenerate a run's evaluation artifacts with
the current code (e.g. after a pipeline fix) without retraining -- or to
recover a run that was killed after training finished but before eval ran.

USAGE:
  python evaluate_checkpoint.py --run-dir runs/exp1_posenc_none_seed0
  python evaluate_checkpoint.py --run-dir runs/exp1_posenc_none_seed0 --checkpoint runs/exp1_posenc_none_seed0/checkpoint_latest.pt
  python evaluate_checkpoint.py --run-dir runs/exp1_posenc_none_seed0 --out-log TF_exp1_posenc_48305_0.out

Then: python analyze_results.py --run-dir <run-dir>
"""
import argparse
import gc
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train_trackformer as tf
from plot_style import PALETTE, apply_style

apply_style()

_EPOCH_LINE_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s*\|\s*train\s+([\d.eE+-]+)\s*\|\s*val\s+([\d.eE+-]+)\s*\|"
)


def plot_loss_curve_from_log(log_path, run_dir):
    """Reconstruct an approximate (sparse, aggregate-only) loss curve from a
    SLURM .out log, for runs whose training_history.json is missing."""
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"WARNING: --out-log {log_path} not found; skipping.")
        return

    epochs, train_loss, val_loss = [], [], []
    with open(log_path) as f:
        for line in f:
            m = _EPOCH_LINE_RE.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(3)))
                val_loss.append(float(m.group(4)))

    if not epochs:
        print(f"WARNING: no epoch lines found in {log_path}; skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, "o-", label="train", color=PALETTE["train"], linewidth=2, markersize=4)
    ax.plot(epochs, val_loss, "o-", label="val", color=PALETTE["val"], linewidth=2, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Aggregate loss")
    ax.set_title(f"Loss curve reconstructed from {log_path.name} ({len(epochs)} points, sparse)")
    ax.legend()
    fig.tight_layout()
    out_path = run_dir / "plot_loss_curve_from_log.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")

    with open(run_dir / "loss_curve_from_log.json", "w") as f:
        json.dump({"epoch": epochs, "train_loss": train_loss, "val_loss": val_loss}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None, help="Default: <run-dir>/model_best.pt")
    ap.add_argument("--out-log", default=None, help="SLURM .out log, for reconstructing a sparse loss curve")
    cli = ap.parse_args()

    run_dir = Path(cli.run_dir)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found -- needed to reconstruct the run's data/model config.")
    with open(config_path) as f:
        config = json.load(f)
    config.pop("n_params", None)  # not a real CLI arg
    args = argparse.Namespace(**config)

    ckpt_path = Path(cli.checkpoint) if cli.checkpoint else run_dir / "model_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found -- nothing to evaluate.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Reconstructing run from {config_path}, evaluating checkpoint {ckpt_path}")

    t_script_start = time.time()

    # Rebuild the same test set (deterministic sort+slice on event IDs).
    t0 = time.time()
    particles_flat, hits_flat, train_event_ids, test_event_ids = tf.load_colliderml(args)
    all_data = tf.build_dataset_frames(particles_flat, hits_flat, args)
    data_load_seconds = time.time() - t0
    print(f"Data ready in {data_load_seconds:.1f}s: {len(all_data)} tracks total")

    test_df = all_data[all_data["event_id"].isin(test_event_ids)].copy()
    del all_data, particles_flat, hits_flat
    gc.collect()
    print(f"Test tracks: {len(test_df)}")

    test_df.to_pickle(run_dir / "test_data.pkl")

    # Rebuild the model and load the checkpoint.
    angle_indices = tuple(sorted(tf.PARAM_NAMES.index(p) for p in args.angle_params))
    model = tf.TrackTransformer(
        input_dim=len(args.features),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        max_hits=args.max_hits,
        pos_encoding=args.pos_encoding,
        head_mode=args.head_mode,
        angle_param_indices=angle_indices,
        num_params=len(tf.PARAM_NAMES),
        enforce_unit_circle=not args.no_unit_circle,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    ckpt_epoch = ckpt.get("epoch")
    ckpt_val = ckpt.get("val_loss", ckpt.get("best_val"))
    print(f"Loaded checkpoint from epoch {ckpt_epoch + 1 if ckpt_epoch is not None else '?'} (val_loss={ckpt_val})")

    # Run inference on the test set.
    test_ds = tf.TrackDataset(test_df["hits_sequence"], max_hits=args.max_hits, input_dim=len(args.features))
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for x, mask in test_loader:
            all_preds.append(model(x.to(device), mask.to(device)).cpu().numpy())
    preds = np.vstack(all_preds)
    for i, name in enumerate(tf.PARAM_NAMES):
        test_df[f"ml_{name}"] = preds[:, i]
    tf.restore_absolute_phi(test_df)
    test_df.to_pickle(run_dir / "test_data_with_predictions.pkl")

    # Speed benchmark.
    warm_x, warm_mask = next(iter(test_loader))
    with torch.no_grad():
        _ = model(warm_x.to(device), warm_mask.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.time()
    with torch.no_grad():
        for x, mask in test_loader:
            _ = model(x.to(device), mask.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - t_start
    n_tracks = len(test_df)
    speed = {
        "device": str(device),
        "total_tracks": n_tracks,
        "total_time_seconds": total_time,
        "tracks_per_second": n_tracks / total_time if total_time > 0 else None,
        "ms_per_track": (total_time / n_tracks) * 1000 if n_tracks > 0 else None,
    }
    with open(run_dir / "speed_benchmark.json", "w") as f:
        json.dump(speed, f, indent=2)
    print(f"Speed: {speed['tracks_per_second']:.0f} tracks/s ({speed['ms_per_track']:.4f} ms/track) on {device}")

    def robust_resolution(residuals):
        q75, q25 = np.percentile(residuals, [75, 25])
        return (q75 - q25) / 1.349

    print("\n" + "-" * 55)
    print(f"{'Parameter':<10} | {'ML resolution':<15}")
    print("-" * 55)
    resolutions = {}
    for name in tf.PARAM_NAMES:
        if name == "phi":
            res = np.remainder(test_df[f"ml_{name}"] - test_df[name] + np.pi, 2 * np.pi) - np.pi
        else:
            res = test_df[f"ml_{name}"] - test_df[name]
        resolutions[name] = robust_resolution(res)
        print(f"{name:<10} | {resolutions[name]:<15.5f}")
    print("-" * 55)
    with open(run_dir / "test_resolutions.json", "w") as f:
        json.dump(resolutions, f, indent=2)

    if getattr(args, "eval_acts", False):
        tf._eval_vs_acts(test_df, test_event_ids, args, run_dir)

    if cli.out_log:
        plot_loss_curve_from_log(cli.out_log, run_dir)

    # Write a run_summary.json (best-effort -- training-time fields only
    # populated if training_history.json already exists from the original run).
    history_path = run_dir / "training_history.json"
    history = None
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    run_summary = {
        "train_events": args.train_events,
        "test_events": args.test_events,
        "n_test_tracks": len(test_df),
        "epochs_configured": args.epochs,
        "best_epoch": (ckpt_epoch + 1) if ckpt_epoch is not None else None,
        "best_val_loss": ckpt_val,
        "note": "Re-evaluated via evaluate_checkpoint.py -- training_seconds reflects the original run if training_history.json exists, else null.",
        "training_seconds": sum(history["epoch_seconds"]) if history else None,
        "data_load_seconds": data_load_seconds,
    }
    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    total_seconds = time.time() - t_script_start
    print(f"\nDone. Total wall time: {total_seconds/60:.1f} min. Artifacts in {run_dir}/.")
    print(f"Now run: python analyze_results.py --run-dir {run_dir}")


if __name__ == "__main__":
    main()
