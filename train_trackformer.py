#!/usr/bin/env python
"""
train_trackformer.py
=====================
Standalone ColliderML -> track-parameter-regression training pipeline.

This is a rebuild of your two Colab notebooks (Collider_ml_pu200_3_feature_eng.ipynb,
End_to_End_Thesis_Pipeline.ipynb), restructured to run from the command line on a
SLURM/GPU cluster, with fixes ported in from Jeremy Couthures' TrackFormer repo
(github.com/CouthuresJeremy/TrackFormer, branch all_params) that his own code relies on:

  1. Rotation-invariant input feature `dphi` (phi relative to each track's innermost
     hit) instead of, or in addition to, raw absolute phi/u/v.
  2. `phi` target trained relative to the same innermost-hit reference angle used by
     `dphi` (Jeremy's `dphi0` mechanism) -- an absolute lab-frame angle isn't
     recoverable from rotation-invariant inputs. See restore_absolute_phi().
  3. z-symmetry canonicalization (Jeremy's `apply_z_symmetry`) -- every track is
     flipped, per-track, to a canonical +z-heading orientation, removing the
     detector's forward/backward mirror symmetry the model would otherwise have to
     learn to handle for free. Applied to the `z` hit feature and to the `z0`/`theta`
     targets consistently. See restore_absolute_z().
  4. Angle outputs (phi, theta) predicted as normalized (cos, sin) pairs, decoded via
     atan2, trained with a periodic loss `2*(1-cos(pred-target))` -- avoids the
     wraparound / boundary issues of regressing a raw angle.
  5. Best-val-loss checkpointing (not just "whatever the last epoch produced").
  6. Geometric-mean loss aggregation across the 5 track parameters (you'd already
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

    # Drop columns nothing in this pipeline ever reads (confirmed via a live
    # check: raw particles_flat/hits_flat carry ~19/~14 columns respectively,
    # e.g. mass, energy, vertex position, true_x/y/z, detector/volume/layer/
    # surface ids -- none of which are used anywhere below). At pu200 scale
    # (hundreds of millions of raw rows before any event/track filtering),
    # keeping only what's needed noticeably cuts memory for everything that
    # touches these tables downstream.
    particles_keep = [
        "event_id", "particle_id", "pdg_id", "charge",
        "px", "py", "pz", "perigee_d0", "perigee_z0", "primary",
    ]
    hits_keep = ["event_id", "particle_id", "x", "y", "z"]
    particles_flat = particles_flat[[c for c in particles_keep if c in particles_flat.columns]]
    hits_flat = hits_flat[[c for c in hits_keep if c in hits_flat.columns]]
    gc.collect()

    return particles_flat, hits_flat, train_event_ids, test_event_ids


def engineer_features(hits_flat: pd.DataFrame) -> pd.DataFrame:
    """Compute every candidate hit-level feature; caller selects a subset via --features."""
    for col in ["x", "y", "z"]:
        hits_flat[col] = hits_flat[col].astype(np.float32)

    hits_flat["r"] = np.sqrt(hits_flat["x"] ** 2 + hits_flat["y"] ** 2).astype(np.float32)

    # Sort inside-out (ascending radius) per track FIRST -- needed both for the
    # z-symmetry reference (first 3 hits, below) and for the dphi reference
    # (first hit) further down, and this is also the order hits_sequence is
    # fed to the model in.
    hits_flat = hits_flat.sort_values(["event_id", "particle_id", "r"])

    # z-symmetry canonicalization, a la Jeremy's TrackFormer apply_z_symmetry()
    # (src/datasets/datamodules.py, all_params branch): a collider detector is
    # forward/backward symmetric in z, so "track heading toward +z" and "track
    # heading toward -z" are physically equivalent up to an overall sign flip
    # of every z-dependent quantity. Without canonicalizing this, the model
    # has to learn both mirror-image cases as if they were unrelated, which
    # inflates the effective spread of z0 (and any z-dependent hit feature)
    # it has to fit. z_sign is computed once per track, from the mean dz
    # (relative to the first hit) of that track's first 3 hits, and applied
    # here to the z hit feature; the SAME z_sign is threaded through to
    # build_dataset_frames() (via split_indices, exactly like phi_offset) so
    # the z0/theta TARGETS get the identical flip -- restore_absolute_z()
    # undoes it afterward for reporting/ACTS comparison.
    rank = hits_flat.groupby(["event_id", "particle_id"]).cumcount()
    first_z = hits_flat.groupby(["event_id", "particle_id"])["z"].transform("first")
    dz_from_first = hits_flat["z"] - first_z
    mean_dz_first3 = (
        dz_from_first.where(rank < 3)
        .groupby([hits_flat["event_id"], hits_flat["particle_id"]])
        .transform("mean")
    )
    z_sign = np.sign(mean_dz_first3.to_numpy())
    z_sign = np.where(z_sign == 0, 1.0, z_sign).astype(np.float32)  # exact tie -> no flip
    hits_flat["z_sign"] = z_sign
    hits_flat["z"] = (hits_flat["z"] * z_sign).astype(np.float32)

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
    # subtract each track's innermost-hit phi (unaffected by the z flip above --
    # phi only depends on x, y, which are never touched).
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


def restore_absolute_phi(df):
    """Convert `phi` (target) and `ml_phi` (prediction), if present, from the
    relative-to-phi_offset frame they're trained/predicted in (see
    build_dataset_frames()) back to absolute lab-frame angles, IN PLACE.
    Call this once, right after test-set predictions are computed, before
    any resolution/ACTS-comparison/reporting code -- NOT during training
    (train/val loss is correctly computed in the relative frame the model
    was trained on; the offset cancels out in a same-frame residual anyway,
    so this conversion is only needed for physically-meaningful reporting
    and comparison against ACTS, which reports absolute phi)."""
    for col in ("phi", "ml_phi"):
        if col in df.columns:
            wrapped = df[col] + df["phi_offset"]
            wrapped = np.where(wrapped > np.pi, wrapped - 2 * np.pi, wrapped)
            wrapped = np.where(wrapped < -np.pi, wrapped + 2 * np.pi, wrapped)
            df[col] = wrapped.astype(np.float32)
    return df


def restore_absolute_z(df):
    """Undo the per-track z-symmetry flip (see engineer_features()) on `z0`/`ml_z0`
    and `theta`/`ml_theta`, IN PLACE, converting them from the canonical-+z training
    frame back to the true lab frame. Call alongside restore_absolute_phi(), once,
    right after test-set predictions are computed -- NOT during training (the flip
    is a bijection with itself, i.e. z_sign*z_sign == 1, so it doesn't change what
    the model can learn; it's only wrong to leave in place when reporting physical
    numbers or comparing against ACTS, which reports untouched lab-frame z0/theta).

    z0 is a simple sign flip (z0_signed = z0_true * z_sign  =>  z0_true =
    z0_signed * z_sign). theta is not: theta = arccos(pz/p), and flipping pz's
    sign maps theta -> pi - theta (not -theta), so it's undone via
    arccos(z_sign * cos(theta_signed)) instead of a plain sign flip."""
    if "z_sign" not in df.columns:
        return df
    z_sign = df["z_sign"].to_numpy()
    for col in ("z0", "ml_z0"):
        if col in df.columns:
            df[col] = (df[col] * z_sign).astype(np.float32)
    for col in ("theta", "ml_theta"):
        if col in df.columns:
            df[col] = np.arccos(np.clip(z_sign * np.cos(df[col]), -1.0, 1.0)).astype(np.float32)
    return df


def build_dataset_frames(particles_flat, hits_flat, args):
    n_raw_particles, n_raw_hits = len(particles_flat), len(hits_flat)
    print(f"[filters] Raw particles: {n_raw_particles:,} | Raw hits: {n_raw_hits:,}")

    # Per-particle hit count, computed on the FULL raw hits table -- this is
    # just a groupby().size() (a single cheap hash-aggregation pass, no
    # sorting or per-hit feature math), so it's fine to do this before any
    # filtering. We need it now because --min-hits/--max-hits depend on it.
    hit_counts_all = hits_flat.groupby(["event_id", "particle_id"]).size().reset_index(name="calculated_hits")

    for col in ["px", "py", "pz", "perigee_d0", "perigee_z0"]:
        particles_flat[col] = particles_flat[col].astype(np.float32)
    particles_flat = particles_flat.merge(hit_counts_all, on=["event_id", "particle_id"], how="left")
    particles_flat["pt"] = np.sqrt(particles_flat["px"] ** 2 + particles_flat["py"] ** 2)
    del hit_counts_all

    # Filters applied one at a time (rather than one combined boolean AND)
    # purely so each cut's effect can be logged individually -- identical
    # final result to the equivalent single combined mask, just with
    # visibility into where tracks actually get lost. Useful both for
    # sanity-checking filter choices and for reporting dataset provenance.
    mask = particles_flat.get("primary", True) == True  # noqa: E712
    if isinstance(mask, bool):  # no "primary" column at all -> vacuously true
        mask = pd.Series(mask, index=particles_flat.index)
    n = len(particles_flat)
    n_after = int(mask.sum())
    print(f"[filters] primary == True:        {n_after:>12,} particles  (-{n - n_after:,})")
    n = n_after

    mask &= particles_flat["pt"] > args.min_pt
    n_after = int(mask.sum())
    print(f"[filters] pt > {args.min_pt:<6g}          {n_after:>12,} particles  (-{n - n_after:,})")
    n = n_after

    mask &= particles_flat["calculated_hits"].fillna(0) >= args.min_hits
    n_after = int(mask.sum())
    print(f"[filters] hits >= {args.min_hits:<6}       {n_after:>12,} particles  (-{n - n_after:,})")
    n = n_after

    mask &= particles_flat["calculated_hits"].fillna(0) <= args.max_hits
    n_after = int(mask.sum())
    print(f"[filters] hits <= {args.max_hits:<6}       {n_after:>12,} particles  (-{n - n_after:,})")
    n = n_after

    if args.particle_types:
        # e.g. --particle-types 13 to reproduce Jeremy's actual single-muon
        # training set (his queued .sub job used dataset_single_muons_...,
        # not the ttbar dataset yaml -- see chat). Compares |pdg_id| so both
        # charge signs of a species are included.
        allowed = set(abs(t) for t in args.particle_types)
        mask &= particles_flat["pdg_id"].abs().isin(allowed)
        n_after = int(mask.sum())
        print(f"[filters] |pdg_id| in {sorted(allowed)}: {n_after:>12,} particles  (-{n - n_after:,})")
        n = n_after

    good = particles_flat[mask].copy()
    del particles_flat, mask
    gc.collect()
    print(f"[filters] Surviving particles (pre hit-level dropna): {len(good):,} / {n_raw_particles:,}")

    p = np.clip(np.sqrt(good["px"] ** 2 + good["py"] ** 2 + good["pz"] ** 2), 1e-7, None)
    good["d0"] = good["perigee_d0"]
    # z0/theta are NOT finalized here -- they depend on the per-track z_sign
    # computed from hits in engineer_features(), which isn't available until
    # after the hits merge below. Kept as "_unsigned"/"_mag" for now; the
    # actual z0/theta targets are set right after the z_sign_df merge further
    # down (mirrors how the phi target isn't finalized until after the
    # phi_offset_df merge). d0 needs no such treatment -- it's a transverse-
    # plane quantity, untouched by the z-symmetry flip.
    good["z0_unsigned"] = good["perigee_z0"]
    good["pz_unsigned"] = good["pz"]
    good["p_mag"] = p
    # ABSOLUTE lab-frame azimuthal angle of the initial momentum direction --
    # kept under this name deliberately, NOT "phi" (see below for why).
    good["phi0_absolute"] = np.arctan2(good["py"], good["px"])
    good["qop"] = good["charge"] / p  # p is a magnitude -- unaffected by the z-sign flip

    # Restrict hits to only the particles that survived the cuts above --
    # BEFORE running the expensive per-hit feature engineering (sort_values +
    # groupby.transform over the full hit table). In a pu200 event, the vast
    # majority of hits belong to pileup particles the primary/pt/hit-count
    # mask discards anyway (confirmed: 950 events loaded ~224M raw hit rows
    # here, versus the low hundreds of thousands that actually survive these
    # cuts). Doing the expensive per-hit math on ~everything first and
    # throwing most of it away afterward -- the original ordering -- was
    # almost certainly the dominant cost behind the multi-hour runtimes and
    # OOM failures, well beyond the earlier scatter-metrics fix alone. This
    # produces an IDENTICAL final result to filtering after feature
    # engineering: engineer_features()'s per-hit math and the dphi
    # computation only ever depend on a track's own hits, never on any other
    # track's, so restricting to survivors first vs. last cannot change any
    # surviving track's computed values -- it only skips computing values
    # for tracks that get discarded either way.
    hits_flat = hits_flat.merge(good[["event_id", "particle_id"]], on=["event_id", "particle_id"], how="inner")
    print(f"[filters] Hits after restricting to surviving particles: {len(hits_flat):,} / {n_raw_hits:,}")
    hits_flat = engineer_features(hits_flat)

    missing = [f for f in args.features if f not in hits_flat.columns]
    if missing:
        raise ValueError(f"Unknown feature(s) {missing}; available: {sorted(hits_flat.columns)}")

    # hit_counts/split_indices assume hits_flat is grouped in the SAME order
    # groupby(sort=True) (the default) would produce -- true here because
    # engineer_features() already sorted hits_flat by (event_id, particle_id, r),
    # so consecutive rows for a given track are already contiguous and in
    # ascending (event_id, particle_id) order, matching hit_counts' row order.
    # (This recomputes counts on the now-filtered hits_flat; the values are
    # identical to hit_counts_all restricted to survivors, since filtering
    # only ever removes whole discarded particles, never individual hits
    # belonging to a surviving one.)
    hit_counts = hits_flat.groupby(["event_id", "particle_id"]).size().reset_index(name="calculated_hits")
    split_indices = np.cumsum(hit_counts["calculated_hits"].values)[:-1]

    raw_vals = hits_flat[args.features].values
    seqs = np.split(raw_vals, split_indices)
    hits_grouped = hit_counts[["event_id", "particle_id"]].copy()
    hits_grouped["hits_sequence"] = seqs

    # Per-track scattering diagnostic (see _track_scatter_metrics docstring).
    # Reuses the same split_indices as hits_sequence above instead of a
    # second full groupby pass building a Python dict per track.
    dphi_seqs = np.split(hits_flat["dphi"].values, split_indices)
    scatter_metrics = [_track_scatter_metrics(s) for s in dphi_seqs]
    scatter_df = hits_grouped[["event_id", "particle_id"]].copy()
    scatter_df["scatter_n_kinks"] = [m[0] for m in scatter_metrics]
    scatter_df["scatter_max_kink"] = [m[1] for m in scatter_metrics]

    # Reference angle for the `phi` target below -- the SAME innermost-hit
    # phi already used as the reference for the `dphi` INPUT feature inside
    # engineer_features() (there: `first_phi`). Extracted here via the same
    # split_indices used for hits_sequence/scatter_df above, so it's
    # guaranteed to be the identical value by construction, not just by
    # convention -- directly answers "are we using the same offset as the
    # input features?": yes, deliberately, this IS that value.
    phi_offset_vals = np.array([seq[0] for seq in np.split(hits_flat["phi"].values, split_indices)], dtype=np.float32)
    phi_offset_df = hits_grouped[["event_id", "particle_id"]].copy()
    phi_offset_df["phi_offset"] = phi_offset_vals

    # z_sign for the z0/theta targets -- SAME per-track value already applied
    # to the `z` hit feature in engineer_features() (constant across all of a
    # track's hits by construction there, so seq[0] == every other entry).
    z_sign_vals = np.array([seq[0] for seq in np.split(hits_flat["z_sign"].values, split_indices)], dtype=np.float32)
    z_sign_df = hits_grouped[["event_id", "particle_id"]].copy()
    z_sign_df["z_sign"] = z_sign_vals

    del hits_flat, raw_vals, dphi_seqs, scatter_metrics
    gc.collect()

    all_data = pd.merge(good, hits_grouped, on=["event_id", "particle_id"], how="inner")
    all_data = pd.merge(all_data, scatter_df, on=["event_id", "particle_id"], how="left")
    all_data = pd.merge(all_data, phi_offset_df, on=["event_id", "particle_id"], how="left")
    all_data = pd.merge(all_data, z_sign_df, on=["event_id", "particle_id"], how="left")

    # z0/theta targets: finalized now that z_sign (from the track's own hits)
    # is available -- see engineer_features()'s z-symmetry docstring and
    # restore_absolute_z(). z0 is a plain sign flip; theta isn't (flipping
    # pz's sign maps theta -> pi - theta, not -theta).
    all_data["z0"] = (all_data["z0_unsigned"] * all_data["z_sign"]).astype(np.float32)
    pz_signed = all_data["pz_unsigned"] * all_data["z_sign"]
    all_data["theta"] = np.arccos(
        np.clip(pz_signed / all_data["p_mag"], -1.0, 1.0)
    ).astype(np.float32)

    # `phi` target: expressed RELATIVE to phi_offset (the SAME reference the
    # `dphi` input feature uses), matching Jeremy's "dphi0" output variable
    # (src/datasets/datamodules.py, all_params branch): merged_df["dphi0"] =
    # merged_df["phi0"] - merged_df["phi_offset"]. An ABSOLUTE lab-frame angle
    # is not learnable from rotation-invariant inputs (r, dphi, z) -- those
    # inputs deliberately discard which way the track points in the
    # transverse plane, so the model has no way to recover it. The relative
    # angle IS learnable, since it only depends on the track's own shape.
    # `phi_offset` itself is kept as a column (not dropped) so predictions
    # can be converted back to absolute phi afterward for physical reporting
    # and ACTS comparison -- see main()'s test-eval section and _eval_vs_acts().
    dphi0 = all_data["phi0_absolute"] - all_data["phi_offset"]
    dphi0 = np.where(dphi0 > np.pi, dphi0 - 2 * np.pi, dphi0)
    dphi0 = np.where(dphi0 < -np.pi, dphi0 + 2 * np.pi, dphi0)
    all_data["phi"] = dphi0.astype(np.float32)

    n_before_dropna = len(all_data)
    all_data = all_data.dropna(subset=PARAM_NAMES).copy()
    n_dropped = n_before_dropna - len(all_data)
    if n_dropped:
        print(f"[filters] Dropped {n_dropped:,} tracks with a NaN target ({sorted(PARAM_NAMES)}): "
              f"{n_before_dropna:,} -> {len(all_data):,}")
    del good, hits_grouped
    gc.collect()
    print(f"[filters] FINAL dataset: {len(all_data):,} tracks "
          f"({len(all_data) / n_raw_particles:.2%} of {n_raw_particles:,} raw particles)")
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


def geometric_multi_task_loss(preds, targets, criteria, aggregate="geometric_mean", norm_loss="std", target_std=None):
    losses = []
    for i, crit in enumerate(criteria):
        p_i, t_i = preds[:, i], targets[:, i]
        if norm_loss == "std":
            # Fixed, precomputed per-parameter std (see main(): target_std is
            # computed once from the full train pool). Previously this was
            # recomputed per-batch via torch.std(t_i), which is both noisy
            # (small/unlucky batches give an unstable normalization) and
            # outright broken for a batch of size 1 -- torch.std() of a
            # single element is NaN (n-1=0 denominator), and .clamp(min=...)
            # does not fix NaN. That NaN then poisons the whole epoch's
            # running loss sum. A DataLoader without drop_last=True (as used
            # for validation) will hit a size-1 last batch whenever
            # len(dataset) % batch_size == 1, exactly what happened here.
            std = target_std[i] if target_std is not None else torch.std(t_i).clamp(min=1e-6)
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

def run_epoch(model, loader, criteria, args, device, optimizer=None, scheduler=None, target_std=None):
    train = optimizer is not None
    model.train(train)
    total_loss, total_param_losses, n_batches = 0.0, np.zeros(len(criteria)), 0

    with torch.set_grad_enabled(train):
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            preds = model(x, mask)
            loss, param_losses = geometric_multi_task_loss(
                preds, y, criteria, aggregate=args.aggregate_loss, norm_loss=args.norm_loss,
                target_std=target_std,
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
    ap.add_argument(
        "--num-layers", type=int, default=8,
        help="Jeremy's 5-parameter ODD/ttbar model (the closest architectural comparison "
             "to this pipeline) uses 16 transformer layers at 32 embedding dims -- much "
             "deeper than his simpler 2-parameter (q/p, pz) model, which uses only 2 layers "
             "at 128 dims and does beat ACTS. Depth appears to matter more than width for "
             "the harder multi-parameter task. Bumped the default here from 4 -> 8 (a "
             "middle ground, not jumping straight to 16, since our train set is ~800 "
             "events vs. the millions of events his fine-tuning stage used -- a much "
             "deeper model risks being harder to train well on less data). Override with "
             "--num-layers if you want to try 4, 16, or anything else.",
    )
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
    ap.add_argument(
        "--resume", default=None,
        help="Path to a checkpoint_latest.pt from a previous (e.g. killed by a "
             "SLURM --time limit) run of this SAME config/--out-dir to continue "
             "training from, instead of starting over. Restores model, optimizer, "
             "scheduler, and history state; training resumes at the next epoch "
             "after the checkpoint's. Does NOT reload data -- data loading always "
             "reruns (fast now: ~7 min at 950 events), so this only saves the "
             "training-time cost, which is the dominant cost for a long run.",
    )
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
    # units (loss normalizes by a fixed per-parameter label std, matching Jeremy's
    # norm_loss="std"), so we keep the scaler only to save alongside artifacts /
    # for reference. scaler.scale_ (the per-parameter std over the full train
    # pool) doubles as that fixed normalization constant below -- computed once
    # here rather than per-batch, which used to be both statistically noisy and
    # capable of producing NaN outright for degenerate (e.g. size-1) batches.
    joblib.dump(scaler, out_dir / "target_scaler.pkl")
    target_std = torch.tensor(scaler.scale_, dtype=torch.float32, device=device).clamp(min=1e-6)

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
    resume_path = out_dir / "checkpoint_latest.pt"
    start_epoch = 0

    if args.resume:
        ckpt_path = Path(args.resume)
        print(f"Resuming from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        best_epoch = ckpt["best_epoch"]
        history = ckpt["history"]
        print(
            f"Resumed at epoch {start_epoch}/{args.epochs} "
            f"(best_val={best_val:.5f} at epoch {best_epoch+1})."
        )
        if start_epoch >= args.epochs:
            print("Resume checkpoint's epoch already >= --epochs; nothing left to train.")

    t_train_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()
        tr_loss, tr_param = run_epoch(
            model, train_loader, criteria, args, device, optimizer, scheduler, target_std=target_std
        )
        val_loss, val_param = run_epoch(model, val_loader, criteria, args, device, target_std=target_std)
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

        # Full resumable checkpoint, overwritten every epoch (not just on
        # improvement) -- unlike model_best.pt, this includes optimizer and
        # scheduler state plus the running history, so a job killed mid-run
        # (e.g. hitting a SLURM --time limit) can be continued with
        # --resume checkpoint_latest.pt instead of restarting from scratch.
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "history": history,
            },
            resume_path,
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
    if best_epoch == -1:
        # val_loss was never a valid, finite number better than the initial
        # +inf (e.g. every epoch's val loss came back NaN) -- model_best.pt
        # was NEVER written, unlike what a generic "Best val loss: ..."
        # message might imply. Surface this loudly instead of silently
        # claiming a save that didn't happen.
        print(
            f"WARNING: val_loss was never finite/improving across all {args.epochs} epochs "
            f"(best_val={best_val}) -- {best_path} was NOT created. Check training_history.json's "
            f"val_param_loss for NaN/inf and fix the underlying data/loss issue before relying on "
            f"downstream scripts (analyze_results.py etc.) that expect model_best.pt to exist."
        )
    else:
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

    # Convert phi/ml_phi from the relative-to-phi_offset training frame, and
    # z0/theta (+ ml_z0/ml_theta) from the z-symmetry-canonicalized training
    # frame, back to physical lab-frame values -- see restore_absolute_phi()
    # and restore_absolute_z() docstrings. Everything from here on
    # (resolutions, ACTS comparison, saved pickles, error-breakdown plots)
    # sees ordinary absolute phi/z0/theta transparently.
    restore_absolute_phi(test_df)
    restore_absolute_z(test_df)

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
