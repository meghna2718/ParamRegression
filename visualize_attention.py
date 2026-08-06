#!/usr/bin/env python
"""
visualize_attention.py
========================
Qualitative interpretability plot: for a handful of example test tracks, show
which hits the model's attention actually attends to, layer by layer.

This is read-only analysis on an already-trained checkpoint -- it does not
modify train_trackformer.py, does not retrain anything, and does not change
model_best.pt. Safe to run any time after a run has finished.

How it works: nn.TransformerEncoderLayer normally discards attention weights
for speed (need_weights=False internally). To recover them we manually
replay each encoder layer's forward pass (same math as PyTorch's internal
_sa_block/_ff_block, post-LN since the model was built with norm_first=False)
but call self_attn(..., need_weights=True) ourselves, so we can capture the
per-head attention matrix at each layer without touching the model's weights
or forward() at all.

Usage:
    python visualize_attention.py --run-dir runs/exp1_posenc_none_seed0 --n-tracks 4
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

from train_trackformer import PARAM_NAMES, TrackDataset
from analyze_results import load_model
from plot_style import apply_style

apply_style()


@torch.no_grad()
def get_attention_maps(model, x, pad_mask):
    """Replay the encoder stack, capturing per-layer attention weights.
    x: [1, seq_len, input_dim], pad_mask: [1, seq_len] (True = padding).
    Returns list of [seq_len, seq_len] arrays, one per encoder layer,
    averaged over attention heads.
    """
    h = model.embedding(x)
    h = model.pos_encoding(h)
    maps = []
    for layer in model.transformer.layers:
        attn_out, attn_weights = layer.self_attn(
            h, h, h, key_padding_mask=pad_mask, need_weights=True, average_attn_weights=True
        )
        h = layer.norm1(h + layer.dropout1(attn_out))
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
        h = layer.norm2(h + layer.dropout2(ff))
        maps.append(attn_weights[0].cpu().numpy())  # [seq_len, seq_len]
    return maps


def plot_track_attention(run_dir, track_idx, features, x_raw, pad_mask, maps, out_path):
    n_valid = int((~pad_mask[0]).sum())
    n_layers = len(maps)

    fig, axes = plt.subplots(1, n_layers, figsize=(4.2 * n_layers, 4.5))
    if n_layers == 1:
        axes = [axes]
    for li, (ax, m) in enumerate(zip(axes, maps)):
        m_valid = m[:n_valid, :n_valid]
        im = ax.imshow(m_valid, cmap="viridis", vmin=0)
        ax.set_title(f"Layer {li + 1}")
        ax.set_xlabel("Hit index (sorted by radius, inside->out)")
        if li == 0:
            ax.set_ylabel("Query hit index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Attention weights -- test track #{track_idx} ({n_valid} hits)", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_attention_vs_radius(run_dir, track_idx, r_values, maps, out_path):
    """Average, over query positions and heads/layers, how much attention
    each hit *receives* -- then plot against that hit's radius. A model that
    ignores position should show a flat line; one that leans on inner/outer
    hits should show a trend."""
    n_valid = len(r_values)
    avg_received = np.mean([m[:n_valid, :n_valid].mean(axis=0) for m in maps], axis=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    order = np.argsort(r_values)
    ax.plot(np.array(r_values)[order], avg_received[order], marker="o", linewidth=2)
    ax.set_xlabel("Hit radius r (mm)")
    ax.set_ylabel("Mean attention received (avg over layers, heads, queries)")
    ax.set_title(f"Attention vs. hit radius -- test track #{track_idx}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-tracks", type=int, default=4, help="Number of example test tracks to visualize")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(run_dir, device)
    model.eval()

    test_df = pd.read_pickle(run_dir / "test_data.pkl").reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(test_df), size=min(args.n_tracks, len(test_df)), replace=False)

    features = cfg["features"]
    r_idx = features.index("r") if "r" in features else None

    out_dir = run_dir / "attention_maps"
    out_dir.mkdir(exist_ok=True)

    ds = TrackDataset(test_df["hits_sequence"], max_hits=cfg["max_hits"], input_dim=len(features))
    for track_idx in idxs:
        x, pad_mask = ds[track_idx]
        x, pad_mask = x.unsqueeze(0).to(device), pad_mask.unsqueeze(0).to(device)
        maps = get_attention_maps(model, x, pad_mask)
        plot_track_attention(run_dir, int(track_idx), features, x, pad_mask, maps, out_dir / f"attention_track{track_idx}.png")

        if r_idx is not None:
            n_valid = int((~pad_mask[0]).sum().item())
            # x was divided by coord_scale (1000.0) in TrackDataset; undo for a physical-units x-axis
            r_values = (x[0, :n_valid, r_idx].cpu().numpy()) * 1000.0
            plot_attention_vs_radius(run_dir, int(track_idx), r_values, maps, out_dir / f"attention_vs_radius_track{track_idx}.png")

    print(f"\nAll attention plots in {out_dir}/")


if __name__ == "__main__":
    main()
