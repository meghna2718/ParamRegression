#!/usr/bin/env python
"""
train_trackformer.py
=====================
Standalone ColliderML -> track-parameter-regression training pipeline.

This is a rebuild of your two Colab notebooks (Collider_ml_pu200_3_feature_eng.ipynb,
End_to_End_Thesis_Pipeline.ipynb), restructured to run from the command line on a
SLURM/GPU cluster, with four fixes ported in from Jeremy Couthures' TrackFormer repo
(github.com/CouthuresJeremy/TrackFormer, branch all_params) that his own code relies on:

  1. Rotation-invariant input feature `dphi` (phi relative to each track's innermost
     hit) instead of, or in addition to, raw absolute phi/u/v.
  2. Angle outputs (phi, theta) predicted as normalized (cos, sin) pairs, decoded via
     atan2, trained with a periodic loss `2*(1-cos(pred-target))` -- avoids the
     wraparound / boundary issues of regressing a raw angle.
  3. Best-val-loss checkpointing (not just "whatever the last epoch produced").
  4. Geometric-mean loss aggregation across the 5 track parameters (you'd already
     built this independently -- kept as-is).

It also exposes the two experiment axes you asked about as CLI flags, so a SLURM
array job can sweep them without editing code:
  --pos-encoding {none,sinusoidal,learnable}
  --head-mode    {shared,separate}
plus standard architecture knobs (--d-model, --nhead, --num-layers, --dropout, ...).

USAGE (see accompanying .slurm scripts for full examples):
  python train_trackformer.py \
      --channel ttbar --pileup pu200 --max-events 1700 --train-events 1500 --test-events 150 \
      --features r dphi z --d-model 128 --nhead 4 --num-layers 4 \
      --pos-encoding none --head-mode shared \
      --batch-size 128 --epochs 400 --lr 1e-3 --out-dir runs/exp1
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Physical target parameter order used throughout this script.
# (Doesn't need to match Jeremy's internal order -- only needs to be consistent
# within this pipeline.)
PARAM_NAMES = ["d0", "z0", "phi", "theta", "qop"]
DEFAULT_ANGLE_PARAMS = ["phi", "theta"]


# ============================================================================
# 1. Data loading + feature engineering
# ============================================================================

def load_colliderml(args):
    from colliderml.core import load_tables, collect_tables
    from colliderml.polars import explode_particles, explode_tracker_hits

    cfg = {
        "dataset_id": args.dataset_id,
        "channels": args.channel,
        "pileup": args.pileup,
        "objects": ["particles", "tracker_hits"],
        "split": "train",
        "lazy": True,
        "max_events": args.max_events,
        "data_dir": args.data_dir,
    }
    tables = load_tables(cfg)
    all_event_ids = tables["particles"].select("event_id").unique().collect()["event_id"].to_list()
    all_event_ids.sort()

    n_needed = args.train_events + args.test_events
    if len(all_event_ids) < n_needed:
        raise ValueError(
            f"Only {len(all_event_ids)} events available, need "
            f"{n_needed} (train {args.train_events} + test {args.test_events}). "
            f"Increase --max-events."
        )
    train_event_ids = all_event_ids[: args.train_events]
    test_event_ids = all_event_ids[args.train_events : args.train_events + args.test_events]
    selected_event_ids = train_event_ids + test_event_ids

    tables["particles"] = tables["particles"].filter(pl.col("event_id").is_in(selected_event_ids))
    tables["tracker_hits"] = tables["tracker_hits"].filter(pl.col("event_id").is_in(selected_event_ids))
    frames = collect_tables(tables)
    del tables
    gc.collect()

    particles_flat = explode_particles(frames["particles"])
    hits_flat = explode_tracker_hits(frames["tracker_hits"])
    del frames
    gc.collect()

    return particles_flat, hits_flat, train_event_ids, test_event_ids


def engineer_features(hits_flat: pd.DataFrame) -> pd.DataFrame:
    """Compute every candidate hit-level feature; caller selects a subset via --features."""
    for col in ["x", "y", "z"]:
        hits_flat[col] = hits_flat[col].astype(np.float32)

    hits_flat["r"] = np.sqrt(hits_flat["x"] ** 2 + hits_flat["y"] ** 2).astype(np.float32)
    hits_flat["s"] = np.clip(
        np.sqrt(hits_flat["x"] ** 2 + hits_flat["y"] ** 2 + hits_flat["z"] ** 2), 1e-7, None
    ).astype(np.float32)
    hits_flat["theta_hit"] = np.clip(
        np.arccos(hits_flat["z"] / hits_flat["s"]), 1e-7, np.pi - 1e-7
    ).astype(np.float32)
    hits_flat["phi"] = np.arctan2(hits_flat["y"], hits_flat["x"]).astype(np.float32)
    hits_flat["eta"] = -np.log(np.tan(hits_flat["theta_hit"] / 2)).astype(np.float32)
    r_sq = np.clip(hits_flat["x"] ** 2 + hits_flat["y"] ** 2, 1e-7, None)
    hits_flat["u"] = (hits_flat["x"] / r_sq).astype(np.float32)
    hits_flat["v"] = (hits_flat["y"] / r_sq).astype(np.float32)

    # Rotation-invariant relative azimuth, a la Jeremy's TrackFormer "dphi" input:
    # sort by radius (inside-out) then subtract each track's innermost-hit phi.
    hits_flat = hits_flat.sort_values(["event_id", "particle_id", "r"])
    first_phi = hits_flat.groupby(["event_id", "particle_id"])["phi"].transform("first")
    dphi = hits_flat["phi"] - first_phi
    dphi = np.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
    hits_flat["dphi"] = dphi.astype(np.float32)

    return hits_flat


def _track_scatter_metrics(dphi_seq):
    """Cheap proxy for 'how non-helical / scattered is this track', reusing
    the dphi-vs-radius sequence: a clean helix has dphi changing smoothly
    and monotonically (in sign of curvature) as you move outward; multiple
    scattering shows up as direction reversals ("kinks"). Not used as a
    training feature -- purely for post-hoc error-breakdown analysis."""
    dphi_seq = np.asarray(dphi_seq)
    if len(dphi_seq) < 3:
        return 0, 0.0
    d1 = np.diff(dphi_seq)
    d2 = np.diff(d1)
    signs = np.sign(d1)
    signs = signs[signs != 0]
    n_kinks = int(np.sum(signs[1:] != signs[:-1])) if len(signs) >= 2 else 0
    max_kink = float(np.max(np.abs(d2))) if len(d2) else 0.0
    return n_kinks, max_kink


def build_dataset_frames(particles_flat, hits_flat, args):
    hits_flat = engineer_features(hits_flat)

    missing = [f for f in args.features if f not in hits_flat.columns]
    if missing:
        raise ValueError(f"Unknown feature(s) {missing}; available: {sorted(hits_flat.columns)}")

    # Per-track scattering diagnostic (see _track_scatter_metrics docstring).
    # hits_flat is already sorted by (event_id, particle_id, r) inside
    # engineer_features, so groupby preserves radius order.
    scatter_rows = [
        {"event_id": eid, "particle_id": pid, **dict(zip(
            ("scatter_n_kinks", "scatter_max_kink"), _track_scatter_metrics(g["dphi"].values)
        ))}
        for (eid, pid), g in hits_flat.groupby(["event_id", "particle_id"], sort=False)
    ]
    scatter_df = pd.DataFrame(scatter_rows)

    hit_counts = hits_flat.groupby(["event_id", "particle_id"]).size().reset_index(name="calculated_hits")
    raw_vals = hits_flat[args.features].values
    split_indices = np.cumsum(hit_counts["calculated_hits"].values)[:-1]
    seqs = np.split(raw_vals, split_indices)
    hits_grouped = hit_counts[["event_id", "particle_id"]].copy()
    hits_grouped["hits_sequence"] = seqs
    del hits_flat, raw_vals
    gc.collect()

    for col in ["px", "py", "pz", "perigee_d0", "perigee_z0"]:
        particles_flat[col] = particles_flat[col].astype(np.float32)
    particles_flat = particles_flat.merge(hit_counts, on=["event_id", "particle_id"], how="left")
    particles_flat["pt"] = np.sqrt(particles_flat["px"] ** 2 + particles_flat["py"] ** 2)

    mask_valid = (
        (particles_flat.get("primary", True) == True)  # noqa: E712
        & (particles_flat["pt"] > args.min_pt)
        & (particles_flat["calculated_hits"].fillna(0) >= args.min_hits)
        & (particles_flat["calculated_hits"].fillna(0) <= args.max_hits)
    )
    if args.particle_types:
        # e.g. --particle-types 13 to reproduce Jeremy's actual single-muon
        # training set (his queued .sub job used dataset_single_muons_...,
        # not the ttbar dataset yaml -- see chat). Compares |pdg_id| so both
        # charge signs of a species are included.
        allowed = set(abs(t) for t in args.particle_types)
        mask_valid &= particles_flat["pdg_id"].abs().isin(allowed)
    good = particles_flat[mask_valid].copy()
    del particles_flat, mask_valid
    gc.collect()

    p = np.clip(np.sqrt(good["px"] ** 2 + good["py"] ** 2 + good["pz"] ** 2), 1e-7, None)
    good["d0"] = good["perigee_d0"]
    good["z0"] = good["perigee_z0"]
    good["phi"] = np.arctan2(good["py"], good["px"])
    good["theta"] = np.arccos(np.clip(good["pz"] / p, -1.0, 1.0))
    good["eta"] = -np.log(np.tan(good["theta"] / 2.0))
    good["qop"] = good["charge"] / p

    all_data = pd.merge(good, hits_grouped, on=["event_id", "particle_id"], how="inner")
    all_data = pd.merge(all_data, scatter_df, on=["event_id", "particle_id"], how="left")
    all_data = all_data.dropna(subset=PARAM_NAMES).copy()
    del good, hits_grouped
    gc.collect()
    return all_data


# ============================================================================
# 2. Dataset
# ============================================================================

class TrackDataset(Dataset):
    def __init__(self, hit_sequences, targets=None, max_hits=20, input_dim=3, coord_scale=1000.0):
        self.hit_sequences = hit_sequences.reset_index(drop=True)
        self.targets = targets
        self.max_hits = max_hits
        self.input_dim = input_dim
        self.coord_scale = coord_scale

    def __len__(self):
        return len(self.hit_sequences)

    def __getitem__(self, idx):
        seq = self.hit_sequences.iloc[idx]
        actual_len = min(len(seq), self.max_hits)
        padded = np.zeros((self.max_hits, self.input_dim), dtype=np.float32)
        pad_mask = np.ones(self.max_hits, dtype=bool)  # True = padding (PyTorch convention)
        padded[:actual_len, :] = np.asarray(seq[:actual_len])
        pad_mask[:actual_len] = False

        x = torch.tensor(padded / self.coord_scale, dtype=torch.float32)
        mask = torch.tensor(pad_mask, dtype=torch.bool)
        if self.targets is not None:
            return x, mask, torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, mask


# ============================================================================
# 3. Model  (embedding / pooling / angle head ported from Jeremy's TrackFormer)
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, mode="sinusoidal"):
        super().__init__()
        self.mode = mode
        if mode == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))
            self.learnable = None
        elif mode == "learnable":
            self.pe = None
            self.learnable = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        else:
            self.pe = None
            self.learnable = None

    def forward(self, x):
        if self.mode == "sinusoidal":
            return x + self.pe[:, : x.size(1), :]
        if self.mode == "learnable":
            return x + self.learnable[:, : x.size(1), :]
        return x


class TrackTransformer(nn.Module):
    """
    Architecture mirrors src/my_model/transformer.py::TrackFormer:
      - 2-layer MLP embedding (Linear -> LeakyReLU -> Linear), optional input dropout
      - optional positional encoding (none / sinusoidal / learnable), applied pre-encoder
      - standard post-LN transformer encoder (norm_first=False, matches Jeremy's EncoderBlock)
      - masked mean pooling over the sequence
      - regression head(s): either one shared head for all params ("shared"), or an
        independent small MLP per physical parameter ("separate") -- this is the
        joint-vs-separate-heads experiment axis.
      - angle parameters are predicted as (cos, sin) pairs, L2-normalized to the unit
        circle, and decoded to radians via atan2 at forward() time.
    """

    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=None,
        dropout=0.1,
        input_dropout=0.0,
        max_hits=20,
        pos_encoding="none",
        head_mode="shared",
        angle_param_indices=(2, 3),  # indices into PARAM_NAMES that are angles
        num_params=5,
        enforce_unit_circle=True,
    ):
        super().__init__()
        dim_feedforward = dim_feedforward or 2 * d_model
        self.num_params = num_params
        self.angle_indices = sorted(set(angle_param_indices))
        self.angle_index_set = set(self.angle_indices)
        self.scalar_indices = [i for i in range(num_params) if i not in self.angle_index_set]
        self.num_angles = len(self.angle_indices)
        self.enforce_unit_circle = enforce_unit_circle
        self.head_mode = head_mode

        self.embedding = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(input_dim, d_model),
            nn.LeakyReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        self.pos_encoding = PositionalEncoding(d_model, max_hits, mode=pos_encoding)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,  # post-LN, matches Jeremy's EncoderBlock
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Augmented output width: scalars (1 each) + angles (2 each, cos/sin)
        augmented_out = len(self.scalar_indices) + 2 * self.num_angles

        if head_mode == "shared":
            self.head = nn.Sequential(
                nn.Linear(d_model, 64), nn.LeakyReLU(inplace=True), nn.Linear(64, augmented_out)
            )
        elif head_mode == "separate":
            # One small head per physical parameter; angle params output 2 dims (cos,sin).
            heads = []
            for i in range(num_params):
                out_i = 2 if i in self.angle_index_set else 1
                heads.append(
                    nn.Sequential(nn.Linear(d_model, 64), nn.LeakyReLU(inplace=True), nn.Linear(64, out_i))
                )
            self.heads = nn.ModuleList(heads)
        else:
            raise ValueError(f"Unknown head_mode: {head_mode}")

    def _pool(self, x, pad_mask):
        # pad_mask: True = padding (PyTorch convention)
        valid = (~pad_mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

    def _assemble_augmented(self, x):
        """Run the head(s) and return augmented output [scalars..., (cos,sin) per angle...]."""
        if self.head_mode == "shared":
            y_aug = self.head(x)
        else:
            scalar_outs, angle_outs = [], []
            for i in range(self.num_params):
                out_i = self.heads[i](x)
                if i in self.angle_index_set:
                    angle_outs.append(out_i)
                else:
                    scalar_outs.append(out_i)
            parts = []
            if scalar_outs:
                parts.append(torch.cat(scalar_outs, dim=-1))
            if angle_outs:
                parts.append(torch.cat(angle_outs, dim=-1))
            y_aug = torch.cat(parts, dim=-1)
        return y_aug

    def _decode(self, y_aug):
        n_scalars = len(self.scalar_indices)
        out = torch.empty(y_aug.size(0), self.num_params, device=y_aug.device, dtype=y_aug.dtype)
        scalars = y_aug[:, :n_scalars]
        angle_block = y_aug[:, n_scalars:]
        if n_scalars:
            out[:, self.scalar_indices] = scalars
        if self.num_angles:
            pairs = angle_block.view(-1, self.num_angles, 2)
            if self.enforce_unit_circle:
                pairs = F.normalize(pairs, dim=-1, eps=1e-6)
            angles = torch.atan2(pairs[..., 1], pairs[..., 0])  # atan2(sin, cos)
            out[:, self.angle_indices] = angles
        return out

    def forward(self, x, pad_mask):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self._pool(x, pad_mask)
        y_aug = self._assemble_augmented(x)
        return self._decode(y_aug)  # [Batch, num_params], angles already in radians


# ============================================================================
# 4. Loss + LR schedule  (ported from src/my_model/utils/modules.py)
# ============================================================================

class ParamLoss:
    """mse: plain MSE. mse_angle: 2*(1-cos(pred-target)) -- periodic, no wraparound issue."""

    def __init__(self, mode="mse"):
        self.mode = mode
        if mode == "mse":
            self.fn = F.mse_loss
        elif mode == "mse_angle":
            self.fn = lambda p, t: torch.mean(2 * (1 - torch.cos(p - t)))
        else:
            raise ValueError(f"Unknown loss mode: {mode}")

    def __call__(self, preds, targets):
        return self.fn(preds, targets)


def geometric_multi_task_loss(preds, targets, criteria, aggregate="geometric_mean", norm_loss="std"):
    losses = []
    for i, crit in enumerate(criteria):
        p_i, t_i = preds[:, i], targets[:, i]
        if norm_loss == "std":
            std = torch.std(t_i).clamp(min=1e-6)
            p_i, t_i = p_i / std, t_i / std
        losses.append(crit(p_i, t_i))
    losses = torch.stack(losses)
    if aggregate == "mean":
        total = losses.mean()
    elif aggregate == "sum":
        total = losses.sum()
    elif aggregate == "geometric_mean":
        total = torch.prod(losses.clamp(min=1e-12)) ** (1.0 / len(losses))
    else:
        raise ValueError(f"Unknown aggregate: {aggregate}")
    return total, losses.detach().cpu().numpy()


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Exact port of Jeremy's scheduler: linear warmup (in optimizer steps) then cosine decay."""

    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=0.0):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]
        factor = 0.5 * (1 + np.cos(np.pi * self.last_epoch / self.max_steps))
        if self.last_epoch <= self.warmup_steps:
            factor *= self.last_epoch / max(self.warmup_steps, 1)
        return [max(base_lr * factor, self.min_lr) for base_lr in self.base_lrs]


# ============================================================================
# 5. Train / eval
# ============================================================================

def run_epoch(model, loader, criteria, args, device, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, total_param_losses, n_batches = 0.0, np.zeros(len(criteria)), 0

    with torch.set_grad_enabled(train):
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            preds = model(x, mask)
            loss, param_losses = geometric_multi_task_loss(
                preds, y, criteria, aggregate=args.aggregate_loss, norm_loss=args.norm_loss
            )
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item()
            total_param_losses += param_losses
            n_batches += 1
    return total_loss / n_batches, total_param_losses / n_batches


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    ap.add_argument("--dataset-id", default="CERN/ColliderML-Release-1")
    ap.add_argument("--data-dir", default="./collider_data")
    ap.add_argument("--channel", default="ttbar")
    ap.add_argument("--pileup", default="pu200")
    ap.add_argument("--max-events", type=int, default=1700)
    ap.add_argument("--train-events", type=int, default=1500)
    ap.add_argument("--test-events", type=int, default=150)
    ap.add_argument("--min-pt", type=float, default=1.0)
    ap.add_argument("--min-hits", type=int, default=8)
    ap.add_argument("--max-hits", type=int, default=20)
    ap.add_argument(
        "--particle-types", type=int, nargs="+", default=None,
        help="Restrict to these |pdg_id| species (e.g. 13 for muons only, to "
             "match Jeremy's actual single-muon training set). Default: no "
             "species filter (all primary particle types), matching the "
             "original notebooks this pipeline was built from.",
    )
    ap.add_argument("--val-frac", type=float, default=0.15)
    # Features
    ap.add_argument(
        "--features", nargs="+", default=["r", "dphi", "z"],
        help="Any of: x y z r s theta_hit phi eta u v dphi",
    )
    ap.add_argument("--angle-params", nargs="+", default=DEFAULT_ANGLE_PARAMS, choices=PARAM_NAMES)
    # Architecture
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--dim-feedforward", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--input-dropout", type=float, default=0.0)
    ap.add_argument("--pos-encoding", choices=["none", "sinusoidal", "learnable"], default="none")
    ap.add_argument("--head-mode", choices=["shared", "separate"], default="shared")
    ap.add_argument("--no-unit-circle", action="store_true", help="Disable cos/sin normalization for angle outputs")
    # Loss
    ap.add_argument(
        "--criterion", nargs="+", default=None,
        help="Per-param loss, in PARAM_NAMES order (d0 z0 phi theta qop). "
             "Default: mse for scalars, mse_angle for --angle-params.",
    )
    ap.add_argument("--aggregate-loss", choices=["mean", "sum", "geometric_mean"], default="geometric_mean")
    ap.add_argument("--norm-loss", choices=["none", "std"], default="std")
    # Optimization
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    # Eval / output
    ap.add_argument("--eval-acts", action="store_true", help="Download ACTS reco tracks for test events and compare")
    ap.add_argument("--out-dir", default="runs/exp")
    args = ap.parse_args()
    if args.norm_loss == "none":
        args.norm_loss = None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    t_script_start = time.time()

    # ---- Data ----
    t0 = time.time()
    particles_flat, hits_flat, train_event_ids, test_event_ids = load_colliderml(args)
    all_data = build_dataset_frames(particles_flat, hits_flat, args)
    data_load_seconds = time.time() - t0
    print(f"Data ready in {data_load_seconds:.1f}s: {len(all_data)} tracks total")

    train_val_df = all_data[all_data["event_id"].isin(train_event_ids)].copy()
    test_df = all_data[all_data["event_id"].isin(test_event_ids)].copy()
    del all_data
    gc.collect()

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    targets_scaled_for_split_only = scaler.fit_transform(train_val_df[PARAM_NAMES].values)
    # NOTE: unlike the original notebooks, the model is trained directly on physical
    # units (loss normalizes per-batch by label std, matching Jeremy's norm_loss="std"),
    # so we keep the scaler only to save alongside artifacts / for reference.
    joblib.dump(scaler, out_dir / "target_scaler.pkl")

    targets = train_val_df[PARAM_NAMES].values.astype(np.float32)
    dataset = TrackDataset(
        train_val_df["hits_sequence"], targets, max_hits=args.max_hits, input_dim=len(args.features)
    )
    n_train = int((1 - args.val_frac) * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Train tracks: {n_train} | Val tracks: {n_val} | Test tracks: {len(test_df)}")

    # ---- Model ----
    angle_indices = tuple(sorted(PARAM_NAMES.index(p) for p in args.angle_params))
    model = TrackTransformer(
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
        num_params=len(PARAM_NAMES),
        enforce_unit_circle=not args.no_unit_circle,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    with open(out_dir / "config.json", "w") as f:
        json.dump({**vars(args), "n_params": n_params}, f, indent=2, default=str)

    if args.criterion is None:
        criteria = [
            ParamLoss("mse_angle" if name in args.angle_params else "mse") for name in PARAM_NAMES
        ]
    else:
        assert len(args.criterion) == len(PARAM_NAMES)
        criteria = [ParamLoss(m) for m in args.criterion]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    max_steps = len(train_loader) * min(1000, args.epochs)
    scheduler = CosineWarmupScheduler(optimizer, args.warmup_steps, max_steps, args.min_lr)

    # ---- Train ----
    history = {"train_loss": [], "val_loss": [], "train_param_loss": [], "val_param_loss": [], "epoch_seconds": []}
    best_val = float("inf")
    best_epoch = -1
    best_path = out_dir / "model_best.pt"
    t_train_start = time.time()

    for epoch in range(args.epochs):
        t_epoch = time.time()
        tr_loss, tr_param = run_epoch(model, train_loader, criteria, args, device, optimizer, scheduler)
        val_loss, val_param = run_epoch(model, val_loader, criteria, args, device)
        epoch_seconds = time.time() - t_epoch

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_param_loss"].append(tr_param.tolist())
        history["val_param_loss"].append(val_param.tolist())
        history["epoch_seconds"].append(epoch_seconds)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                best_path,
            )

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(
                f"Epoch {epoch+1:03d}/{args.epochs} | train {tr_loss:.5f} | val {val_loss:.5f} "
                f"| best {best_val:.5f} (epoch {best_epoch+1}) | {epoch_seconds:.1f}s/epoch"
            )

    training_seconds = time.time() - t_train_start
    torch.save({"model_state": model.state_dict(), "epoch": args.epochs - 1, "val_loss": val_loss}, out_dir / "model_last.pt")
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    test_df.to_pickle(out_dir / "test_data.pkl")
    print(
        f"Best val loss: {best_val:.5f} at epoch {best_epoch+1}/{args.epochs} (saved to {best_path}) | "
        f"training took {training_seconds/60:.1f} min ({training_seconds/args.epochs:.2f} s/epoch avg)"
    )

    # Timing/convergence summary -- read this before sizing up other runs.
    # best_epoch tells you how many epochs were actually needed for THIS
    # dataset size to reach its best val_loss; if it's a lot less than
    # --epochs, you likely wasted GPU time and can shorten future runs at
    # this data size (or the reverse: if best_epoch == epochs-1, val_loss
    # was probably still improving and you should train longer).
    run_summary = {
        "train_events": args.train_events,
        "test_events": args.test_events,
        "n_train_tracks": n_train,
        "n_val_tracks": n_val,
        "n_test_tracks": len(test_df),
        "epochs_configured": args.epochs,
        "best_epoch": best_epoch + 1,  # 1-indexed for readability
        "best_val_loss": best_val,
        "final_val_loss": val_loss,
        "data_load_seconds": data_load_seconds,
        "training_seconds": training_seconds,
        "avg_seconds_per_epoch": training_seconds / args.epochs,
        "seconds_to_reach_best_epoch": sum(history["epoch_seconds"][: best_epoch + 1]),
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    # ---- Test-set evaluation, optionally vs ACTS ----
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = TrackDataset(test_df["hits_sequence"], max_hits=args.max_hits, input_dim=len(args.features))
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for x, mask in test_loader:
            all_preds.append(model(x.to(device), mask.to(device)).cpu().numpy())
    preds = np.vstack(all_preds)
    for i, name in enumerate(PARAM_NAMES):
        test_df[f"ml_{name}"] = preds[:, i]

    # Save the per-track predictions (not just aggregate resolutions) -- needed
    # for pT-binned / hit-count-binned / fractional-error plots downstream.
    test_df.to_pickle(out_dir / "test_data_with_predictions.pkl")

    # ---- Inference speed benchmark ----
    warm_x, warm_mask = next(iter(test_loader))
    with torch.no_grad():
        _ = model(warm_x.to(device), warm_mask.to(device))  # warm up
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
    with open(out_dir / "speed_benchmark.json", "w") as f:
        json.dump(speed, f, indent=2)
    print(f"Speed: {speed['tracks_per_second']:.0f} tracks/s ({speed['ms_per_track']:.4f} ms/track) on {device}")

    def robust_resolution(residuals):
        q75, q25 = np.percentile(residuals, [75, 25])
        return (q75 - q25) / 1.349

    print("\n" + "-" * 55)
    print(f"{'Parameter':<10} | {'ML resolution':<15}")
    print("-" * 55)
    resolutions = {}
    for name in PARAM_NAMES:
        if name == "phi":
            res = np.remainder(test_df[f"ml_{name}"] - test_df[name] + np.pi, 2 * np.pi) - np.pi
        else:
            res = test_df[f"ml_{name}"] - test_df[name]
        resolutions[name] = robust_resolution(res)
        print(f"{name:<10} | {resolutions[name]:<15.5f}")
    print("-" * 55)
    with open(out_dir / "test_resolutions.json", "w") as f:
        json.dump(resolutions, f, indent=2)

    if args.eval_acts:
        _eval_vs_acts(test_df, test_event_ids, args, out_dir)

    total_seconds = time.time() - t_script_start
    run_summary["total_seconds"] = total_seconds
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)
    print(
        f"\nDone. Total wall time: {total_seconds/60:.1f} min "
        f"(data {data_load_seconds/60:.1f} min, training {training_seconds/60:.1f} min, "
        f"eval {(total_seconds - data_load_seconds - training_seconds)/60:.1f} min). "
        f"Artifacts in {out_dir}/ (see run_summary.json for the numbers above)."
    )


def sigma_clipped_resolution(residuals, max_iterations=100):
    """Iterative sigma-clipping resolution: repeatedly drop residuals beyond
    3 std devs (recomputing mean/std each pass) until convergence, return
    the std of the surviving residuals. A standard robust-statistics
    technique (also used in ATLAS/CMS tracking performance studies); used
    alongside -- not instead of -- our IQR-based robust_resolution."""
    residuals = np.array(residuals)
    for _ in range(max_iterations):
        mean_r, std_r = np.mean(residuals), np.std(residuals)
        inlier = np.abs(residuals - mean_r) <= 3 * std_r
        filtered = residuals[inlier]
        if len(filtered) == len(residuals):
            break
        residuals = filtered
    return float(np.std(residuals))


def shortest_interval_resolution(data, alpha=0.6827):
    """Shortest-interval resolution/bias: the narrowest interval of the
    sorted data containing `alpha` fraction of points -- does not assume
    symmetry around the median, unlike IQR/1.349. Returns (resolution,
    bias) = (half-width, center); this is the standard ATLAS/CMS-style
    tracking-resolution convention."""
    data = np.sort(np.asarray(data))
    n = len(data)
    n_in_interval = int(alpha * n)
    if n_in_interval <= 0 or n_in_interval >= n:
        return float(np.std(data)), float(np.median(data))
    best_width = np.inf
    best_lo, best_hi = data[0], data[-1]
    for i in range(n - n_in_interval):
        width = data[i + n_in_interval] - data[i]
        if width < best_width:
            best_width = width
            best_lo, best_hi = data[i], data[i + n_in_interval]
    center = (best_lo + best_hi) / 2.0  # simplified vs. his index-midpoint lookup; equivalent for continuous data
    return float(best_width / 2.0), float(center)


def _eval_vs_acts(test_df, test_event_ids, args, out_dir):
    from colliderml.core import load_tables, collect_tables

    cfg = {
        "dataset_id": args.dataset_id, "channels": args.channel, "pileup": args.pileup,
        "objects": ["tracks"], "split": "train", "lazy": True,
        "max_events": args.max_events, "data_dir": args.data_dir,
    }
    tables = load_tables(cfg)
    tables["tracks"] = tables["tracks"].filter(pl.col("event_id").is_in(test_event_ids))
    acts_pl = collect_tables(tables)["tracks"]
    acts_pl.write_parquet(out_dir / "acts_benchmark_tracks.parquet")

    acts_pl = acts_pl.explode(["track_id", "majority_particle_id", "d0", "z0", "phi", "theta", "qop"])
    acts_df = acts_pl.to_pandas().rename(
        columns={
            "majority_particle_id": "particle_id", "d0": "acts_d0", "z0": "acts_z0",
            "phi": "acts_phi", "theta": "acts_theta", "qop": "acts_qop",
        }
    ).dropna(subset=["particle_id"])

    test_df = test_df.copy()
    test_df["event_id"], test_df["particle_id"] = test_df["event_id"].astype(str), test_df["particle_id"].astype(str)
    acts_df["event_id"], acts_df["particle_id"] = acts_df["event_id"].astype(str), acts_df["particle_id"].astype(str)
    comp = pd.merge(test_df, acts_df, on=["event_id", "particle_id"], how="inner").dropna(subset=["acts_d0"])

    def robust_resolution(residuals):
        q75, q25 = np.percentile(residuals, [75, 25])
        return (q75 - q25) / 1.349

    def bootstrap_resolution_ci(residuals, n_boot=1000, seed=0):
        """68% CI on the IQR-based resolution via bootstrap resampling of tracks."""
        rng = np.random.default_rng(seed)
        residuals = np.asarray(residuals)
        n = len(residuals)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            sample = residuals[rng.integers(0, n, n)]
            boots[b] = robust_resolution(sample)
        lo, hi = np.percentile(boots, [16, 84])  # 68% CI, comparable to +/-1 sigma
        return float(lo), float(hi)

    # Match/finding-rate: ML gets a prediction for every truth track by
    # construction, but ACTS only reconstructs (and gets truth-matched to)
    # some fraction of them. Resolution numbers below are conditioned on
    # ACTS having found the track -- report that rate explicitly so the
    # comparison isn't silently biased in ACTS's favor.
    match_rate = len(comp) / len(test_df) if len(test_df) > 0 else float("nan")
    print(f"\nACTS match rate on test set: {len(comp)}/{len(test_df)} tracks ({match_rate:.1%})")
    print("(Resolution comparison below is computed only on ACTS-matched tracks.)")

    from scipy.stats import wilcoxon

    print("\n" + "-" * 100)
    print(
        f"{'Parameter':<10} | {'ML resolution':<16} | {'ACTS resolution':<16} | "
        f"{'ML outlier%':<12} | {'ACTS outlier%':<13} | {'p-value':<10}"
    )
    print("-" * 100)
    comparison = {"match_rate": match_rate, "n_test_tracks": len(test_df), "n_matched_tracks": len(comp)}
    for name in PARAM_NAMES:
        if name == "phi":
            ml_res = np.remainder(comp[f"ml_{name}"] - comp[name] + np.pi, 2 * np.pi) - np.pi
            acts_res = np.remainder(comp[f"acts_{name}"] - comp[name] + np.pi, 2 * np.pi) - np.pi
        else:
            ml_res = comp[f"ml_{name}"] - comp[name]
            acts_res = comp[f"acts_{name}"] - comp[name]

        ml_val, acts_val = robust_resolution(ml_res), robust_resolution(acts_res)
        ml_ci = bootstrap_resolution_ci(ml_res)
        acts_ci = bootstrap_resolution_ci(acts_res)

        # Outlier rate: fraction of tracks with >10% fractional error on qop
        # (only meaningful for qop; reported for all params on absolute
        # residual > 5x the ML resolution, a simple relative-tail definition).
        thresh = 5 * max(ml_val, acts_val, 1e-9)
        ml_outlier = float((np.abs(ml_res) > thresh).mean() * 100)
        acts_outlier = float((np.abs(acts_res) > thresh).mean() * 100)

        # Paired significance test: is ML's |error| distribution different
        # from ACTS's on the same tracks? (Wilcoxon signed-rank, robust to
        # non-normal error distributions.)
        try:
            stat, pval = wilcoxon(np.abs(ml_res), np.abs(acts_res))
        except ValueError:
            pval = float("nan")  # e.g. all differences are zero

        # Shortest-interval resolution/bias (standard ATLAS/CMS tracking
        # convention: narrowest 68.27% interval + iterative sigma-clipped
        # std), for numbers comparable to typical tracking-performance
        # papers -- our IQR-based number above is a different, simpler
        # estimator (assumes rough symmetry; this one doesn't).
        ml_interval_res, ml_interval_bias = shortest_interval_resolution(ml_res)
        acts_interval_res, acts_interval_bias = shortest_interval_resolution(acts_res)
        ml_clipped_std = sigma_clipped_resolution(ml_res)
        acts_clipped_std = sigma_clipped_resolution(acts_res)

        comparison[name] = {
            "ml": ml_val, "ml_ci68": ml_ci,
            "acts": acts_val, "acts_ci68": acts_ci,
            "ml_outlier_pct": ml_outlier, "acts_outlier_pct": acts_outlier,
            "wilcoxon_pvalue": float(pval),
            "ml_interval_resolution": ml_interval_res, "ml_interval_bias": ml_interval_bias,
            "acts_interval_resolution": acts_interval_res, "acts_interval_bias": acts_interval_bias,
            "ml_sigma_clipped_resolution": ml_clipped_std, "acts_sigma_clipped_resolution": acts_clipped_std,
        }
        print(
            f"{name:<10} | {ml_val:<16.5f} | {acts_val:<16.5f} | "
            f"{ml_outlier:<12.2f} | {acts_outlier:<13.2f} | {pval:<10.4g}"
        )
        print(
            f"{'':<10} | Shortest-interval: ML res={ml_interval_res:.5f} bias={ml_interval_bias:+.5f} | "
            f"ACTS res={acts_interval_res:.5f} bias={acts_interval_bias:+.5f}"
        )
    print("-" * 100)

    # Charge misidentification rate: sign(qop) encodes particle charge, so a
    # sign flip is a qualitatively different (and physically more serious)
    # failure than an ordinary magnitude error -- worth its own metric rather
    # than being folded into RMSE/resolution on qop.
    ml_charge_misid = float((np.sign(comp["ml_qop"]) != np.sign(comp["qop"])).mean() * 100)
    acts_charge_misid = float((np.sign(comp["acts_qop"]) != np.sign(comp["qop"])).mean() * 100)
    comparison["charge_misid_pct"] = {"ml": ml_charge_misid, "acts": acts_charge_misid}
    print(
        f"\nCharge misidentification rate: ML {ml_charge_misid:.2f}% | ACTS {acts_charge_misid:.2f}% "
        f"(fraction of tracks where sign(q/p) prediction disagrees with truth)"
    )

    with open(out_dir / "acts_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Save the full per-track merged dataframe (truth + ml_* + acts_* + pt +
    # calculated_hits). This is what the pT-binned, hit-count-binned, and
    # fractional-error-distribution plots need -- the aggregate resolutions
    # above aren't enough for those.
    comp.to_pickle(out_dir / "comparison_df.pkl")
    print(f"Saved per-track comparison dataframe ({len(comp)} tracks) to comparison_df.pkl")


if __name__ == "__main__":
    main()
