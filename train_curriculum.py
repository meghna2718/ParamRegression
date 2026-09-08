#!/usr/bin/env python
"""
train_curriculum.py -- Precision curriculum sweep (Transformer vs. least-
squares baseline) across increasing physical complexity. Cluster-runnable
version of the Colab notebook: same functions, Google Drive replaced by
--out-dir, plots saved to disk instead of displayed.

Usage:
    python train_curriculum.py --out-dir runs/curriculum_seed_sweep --seeds 0 1 2

Resumable: rerunning with the same --out-dir skips (stage, label, seed)
combinations already present in sweep_results.csv.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # headless -- no display on a cluster node
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import time
import os
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cpu":
    print("WARNING: no GPU detected -- check the SLURM --gres request.")
import argparse
import sys


# ============================================================================
# Part 1 -- Synthetic track generator
# ============================================================================

C = 0.299792458  # GeV / (T * m)


def rho_from_pt(pt_gev, B_tesla):
    """Transverse radius of curvature [mm] from transverse momentum."""
    rho_m = pt_gev / (C * B_tesla)
    return rho_m * 1000.0


def generate_batch(
    n_tracks, rng,
    n_hits_range=(6, 12), pt_range=(1.0, 10.0), pz_range=(-5.0, 5.0),
    vertex_spread_xy=0.0, vertex_spread_z=0.0, phi0_range=(-np.pi, np.pi),
    B_tesla=2.0, charge_choices=(-1.0, 1.0),
    hit_noise_std=0.0, scatter_std=0.0, alpha_span_frac=(0.15, 0.6),
    dtype=np.float32,
):
    """Generate a batch of synthetic helical tracks. Vectorized across
    tracks.

    Returns
    -------
    hits : (n_tracks, max_hits, 3) array - padded (x, y, z) hit positions
    mask : (n_tracks, max_hits) bool - True where padded
    targets : dict of (n_tracks,) arrays - true track parameters
    """
    n_hits = rng.integers(n_hits_range[0], n_hits_range[1] + 1, size=n_tracks)
    max_hits = int(n_hits.max())

    pt = rng.uniform(*pt_range, size=n_tracks)
    pz = rng.uniform(*pz_range, size=n_tracks)
    q = rng.choice(charge_choices, size=n_tracks)
    phi0 = rng.uniform(*phi0_range, size=n_tracks)

    vx = rng.normal(0.0, vertex_spread_xy, size=n_tracks) if vertex_spread_xy > 0 else np.zeros(n_tracks)
    vy = rng.normal(0.0, vertex_spread_xy, size=n_tracks) if vertex_spread_xy > 0 else np.zeros(n_tracks)
    vz = rng.normal(0.0, vertex_spread_z, size=n_tracks) if vertex_spread_z > 0 else np.zeros(n_tracks)

    rho = rho_from_pt(pt, B_tesla)
    cot_theta = pz / pt
    span_frac = rng.uniform(*alpha_span_frac, size=n_tracks)

    # Per-track turning angle, built for every track at once. Rows with
    # fewer than max_hits real hits get filled out to max_hits anyway -
    # the extra entries are masked out below, matching the original
    # per-track linspace(0, span, k) for each row's first k entries.
    j = np.arange(max_hits)
    valid = j[None, :] < n_hits[:, None]                       # (n_tracks, max_hits) bool
    denom = np.maximum(n_hits - 1, 1)[:, None]
    frac = j[None, :] / denom
    alphas = frac * (span_frac[:, None] * 2 * np.pi)
    alphas = alphas + rng.normal(0.0, 0.02 * span_frac[:, None], size=(n_tracks, max_hits))

    q_b, phi0_b, rho_b = q[:, None], phi0[:, None], rho[:, None]
    x = vx[:, None] + rho_b * (np.sin(phi0_b + q_b * alphas) - np.sin(phi0_b))
    y = vy[:, None] - rho_b * (np.cos(phi0_b + q_b * alphas) - np.cos(phi0_b))

    if scatter_std > 0.0:
        kicks = np.cumsum(rng.normal(0.0, scatter_std, size=(n_tracks, max_hits)), axis=1)
        x = x + rho_b * np.sin(kicks) * 0.05
        y = y + rho_b * np.cos(kicks) * 0.05

    z = vz[:, None] + rho_b * alphas * cot_theta[:, None]

    hits = np.stack([x, y, z], axis=-1)
    if hit_noise_std > 0.0:
        hits = hits + rng.normal(0.0, hit_noise_std, size=hits.shape)
    hits = hits.astype(dtype)
    hits[~valid] = 0.0
    mask = ~valid

    targets = {
        "q_over_pt": (q / pt).astype(dtype), "pz": pz.astype(dtype),
        "q": q.astype(dtype), "pt": pt.astype(dtype),
        "vertex_x": vx.astype(dtype), "vertex_y": vy.astype(dtype), "vertex_z": vz.astype(dtype),
        "phi0": phi0.astype(dtype),
    }
    return hits, mask, targets


STAGES = {
    "stage0_everything_fixed": dict(
        pt_range=(2.0, 2.0), pz_range=(0.0, 0.0), vertex_spread_xy=0.0, vertex_spread_z=0.0,
        hit_noise_std=0.0, scatter_std=0.0, phi0_range=(0.0, 0.0), charge_choices=(1.0,),
    ),
    "stage0b_pt_only": dict(
        pt_range=(1.0, 10.0), pz_range=(0.0, 0.0), vertex_spread_xy=0.0, vertex_spread_z=0.0,
        hit_noise_std=0.0, scatter_std=0.0, phi0_range=(0.0, 0.0), charge_choices=(1.0,),
    ),
    "stage1_charge_only": dict(
        pt_range=(2.0, 2.0), pz_range=(0.0, 0.0), vertex_spread_xy=0.0, vertex_spread_z=0.0,
        hit_noise_std=0.0, scatter_std=0.0, phi0_range=(0.0, 0.0),
    ),
    "stage2_varying_momentum": dict(
        pt_range=(1.0, 10.0), pz_range=(-5.0, 5.0), vertex_spread_xy=0.0, vertex_spread_z=0.0,
        hit_noise_std=0.0, scatter_std=0.0,
    ),
    "stage3_vertex_offset": dict(
        pt_range=(1.0, 10.0), pz_range=(-5.0, 5.0), vertex_spread_xy=1.0, vertex_spread_z=50.0,
        hit_noise_std=0.0, scatter_std=0.0,
    ),
    "stage4_measurement_noise": dict(
        pt_range=(1.0, 10.0), pz_range=(-5.0, 5.0), vertex_spread_xy=1.0, vertex_spread_z=50.0,
        hit_noise_std=0.05, scatter_std=0.0,
    ),
    "stage5_scattering": dict(
        pt_range=(1.0, 10.0), pz_range=(-5.0, 5.0), vertex_spread_xy=1.0, vertex_spread_z=50.0,
        hit_noise_std=0.05, scatter_std=0.01,
    ),
}
STAGE_ORDER = ["stage0_everything_fixed", "stage0b_pt_only", "stage1_charge_only",
               "stage2_varying_momentum", "stage3_vertex_offset", "stage4_measurement_noise",
               "stage5_scattering"]
TARGET = "q_over_pt"
print("Stages:", STAGE_ORDER)


# ============================================================================
# Part 2 -- Least-squares baseline (vectorized)
# ============================================================================

def fit_batch_vectorized(hits, mask, B_tesla=2.0, dtype=torch.float32, max_iter=15, lam=1e-6):
    """Batched circle-and-line fit for ALL tracks at once -- replaces the
    per-track Python loop entirely.

    hits: (N, max_hits, 3) numpy array
    mask: (N, max_hits) bool, True = padding
    Returns dict of (N,) numpy arrays: q_over_pt, pz, pt, q
    """
    C = 0.299792458
    N = hits.shape[0]
    x = torch.tensor(hits[..., 0], dtype=dtype)
    y = torch.tensor(hits[..., 1], dtype=dtype)
    z = torch.tensor(hits[..., 2], dtype=dtype)
    pad = torch.tensor(mask)
    valid = (~pad).to(dtype)

    # --- Kasa algebraic initial guess, batched ---
    xm, ym = x * valid, y * valid
    A = torch.stack([xm, ym, valid], dim=-1)             # (N, max_hits, 3)
    b = -(xm**2 + ym**2) * valid
    sol = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)  # (N, 3)
    xc, yc = -sol[:, 0] / 2, -sol[:, 1] / 2
    R = torch.sqrt(torch.clamp(xc**2 + yc**2 - sol[:, 2], min=1e-9))

    # --- Batched Gauss-Newton refinement, closed-form Jacobian (no autograd needed) ---
    for _ in range(max_iter):
        dx, dy = x - xc.unsqueeze(1), y - yc.unsqueeze(1)
        dist = torch.sqrt(dx**2 + dy**2 + 1e-12)
        r = (dist - R.unsqueeze(1)) * valid
        J = torch.stack([-dx / dist * valid, -dy / dist * valid, -valid], dim=-1)  # (N, max_hits, 3)

        JTJ = torch.einsum('nij,nik->njk', J, J)
        JTr = torch.einsum('nij,ni->nj', J, r)
        damped = JTJ + lam * torch.eye(3, dtype=dtype)
        delta = torch.linalg.solve(damped, (-JTr).unsqueeze(-1)).squeeze(-1)
        xc, yc, R = xc + delta[:, 0], yc + delta[:, 1], R + delta[:, 2]

    pt_pred = C * B_tesla * (R / 1000.0)

    # --- Charge sign + relative angle, via cross/dot (avoids per-track np.unwrap) ---
    dx, dy = x - xc.unsqueeze(1), y - yc.unsqueeze(1)
    cross = dx[:, :-1] * dy[:, 1:] - dy[:, :-1] * dx[:, 1:]
    dot = dx[:, :-1] * dx[:, 1:] + dy[:, :-1] * dy[:, 1:]
    dtheta = torch.atan2(cross, dot)
    pair_valid = (~pad[:, :-1]) & (~pad[:, 1:])
    dtheta = dtheta * pair_valid.to(dtype)
    n_pairs = pair_valid.sum(dim=1).clamp(min=1).to(dtype)
    q_pred = torch.sign(dtheta.sum(dim=1) / n_pairs)
    ang_rel = torch.cat([torch.zeros(N, 1, dtype=dtype), torch.cumsum(dtheta, dim=1)], dim=1)

    # --- z vs. arc-length line fit, batched ---
    s = R.unsqueeze(1) * ang_rel
    s_m, ones_m, z_m = s * valid, valid, z * valid
    A2 = torch.stack([s_m, ones_m], dim=-1)
    ATA = torch.einsum('nij,nik->njk', A2, A2)
    ATb = torch.einsum('nij,ni->nj', A2, z_m)
    sol2 = torch.linalg.solve(ATA + 1e-12 * torch.eye(2, dtype=dtype), ATb.unsqueeze(-1)).squeeze(-1)
    slope = sol2[:, 0]
    pz_pred = slope * pt_pred * q_pred

    q_over_pt_pred = torch.where(pt_pred > 0, q_pred / pt_pred, torch.zeros_like(pt_pred))

    return {
        "q_over_pt": q_over_pt_pred.numpy().astype(np.float64),
        "pz": pz_pred.numpy().astype(np.float64),
        "pt": pt_pred.numpy().astype(np.float64),
        "q": q_pred.numpy().astype(np.float64),
    }


def robust_resolution(residuals):
    """IQR/1.349 -- the same resolution metric used throughout Study 1,
    chosen so both studies report resolution the same way without relying
    on a Gaussian-fit assumption."""
    r = np.asarray(residuals)
    q75, q25 = np.percentile(r, [75, 25])
    return (q75 - q25) / 1.349


def robust_bias(residuals):
    """Median residual -- paired with robust_resolution as the location
    estimate, matching the median+IQR robust-statistics convention (in
    place of mean+std)."""
    return float(np.median(residuals))


# Sanity check: fit correctness at high precision, independent of the
# float32 comparison feature above.
_rng = np.random.default_rng(1)
_hits, _mask, _targets = generate_batch(500, _rng, **STAGES["stage2_varying_momentum"], dtype=np.float64)
_fit = fit_batch_vectorized(_hits, _mask, dtype=torch.float64)
_err = _fit["q_over_pt"] - _targets["q_over_pt"]
print(f"Sanity check (should be ~1e-10 or smaller): mean={_err.mean():.3e}  std={_err.std():.3e}")


# ============================================================================
# Part 3 -- Model, precision, data utilities
# ============================================================================

PRECISION = torch.float32
NUMPY_DTYPE = np.float32
print(f"Working precision: {PRECISION} (model and baseline both use this)")


class TrackTransformer(nn.Module):
    """Transformer encoder -> masked mean pooling -> regression head.
    Permutation invariant over hits.

    Input features (r, dphi, z) match train_trackformer.py's default
    --features r dphi z: r and z scaled by /1000 (mm -> ~O(1-30)), dphi is
    the rotation-invariant azimuth relative to each track's first hit,
    left unscaled (already O(1) radians)."""

    def __init__(self, input_dim=3, embed_dim=64, n_layers=2, n_heads=4, ff_dim=128,
                 dropout=0.0, targets=("q_over_pt",), head_mode="shared"):
        super().__init__()
        self.targets = list(targets)
        self.head_mode = head_mode
        self.embed = nn.Linear(input_dim, embed_dim)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim,
                                            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, len(self.targets)))

    def forward(self, hits, mask):
        x_coord, y_coord, z_coord = hits[..., 0], hits[..., 1], hits[..., 2]
        r = torch.sqrt(x_coord**2 + y_coord**2)
        phi = torch.atan2(y_coord, x_coord)
        first_phi = phi[:, :1]
        dphi = phi - first_phi
        dphi = torch.where(dphi > torch.pi, dphi - 2 * torch.pi, dphi)
        dphi = torch.where(dphi < -torch.pi, dphi + 2 * torch.pi, dphi)
        hits = torch.stack([r / 1000.0, dphi, z_coord / 1000.0], dim=-1)

        x = self.embed(hits)
        x = self.encoder(x, src_key_padding_mask=mask)
        real = (~mask).float().unsqueeze(-1)
        pooled = (x * real).sum(dim=1) / real.sum(dim=1).clamp(min=1.0)
        out = self.head(pooled)
        return {t: out[:, i] for i, t in enumerate(self.targets)}


def make_stream(stage_cfg, batch_size, seed, extra_targets=("q_over_pt",), dtype=np.float32):
    """Infinite generator of fresh training batches (no fixed training set)."""
    rng = np.random.default_rng(seed)
    while True:
        hits, mask, targets = generate_batch(batch_size, rng, dtype=dtype, **stage_cfg)
        yield (torch.from_numpy(hits), torch.from_numpy(mask),
               {k: torch.from_numpy(targets[k]) for k in extra_targets})


def make_fixed_eval_set(stage_cfg, n, seed, extra_targets=("q_over_pt",), dtype=np.float32):
    """One fixed, seeded evaluation set -- reused across runs for comparable metrics."""
    rng = np.random.default_rng(seed)
    hits, mask, targets = generate_batch(n, rng, dtype=dtype, **stage_cfg)
    return (torch.from_numpy(hits), torch.from_numpy(mask),
            {k: torch.from_numpy(targets[k]) for k in extra_targets})


def batched_forward(model, hits, mask, device, chunk_size=4096):
    """Chunked inference -- avoids a CUDA kernel limit hit by very large batches."""
    model.eval()
    n = hits.shape[0]
    all_preds = {}
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_pred = model(hits[start:end].to(device), mask[start:end].to(device))
            for k, v in chunk_pred.items():
                all_preds.setdefault(k, []).append(v.cpu())
    model.train()
    return {k: torch.cat(v, dim=0) for k, v in all_preds.items()}


def evaluate(model, hits, mask, targets, device, chunk_size=4096):
    pred = batched_forward(model, hits, mask, device, chunk_size=chunk_size)
    return {k: torch.mean((pred[k] - targets[k]) ** 2).item() for k in targets}

print("Model and data utilities defined.")


# ============================================================================
# Part 4 -- Training and refinement
# ============================================================================

EVAL_EVERY = 250  # shared constant -- also used to label the kink plot's x-axis correctly


def set_warmup_lr(opt, base_lr, step, warmup_steps):
    """Linear warmup, applied manually so it can hand off cleanly to
    ReduceLROnPlateau afterward."""
    warmup_lr = base_lr * (step + 1) / max(warmup_steps, 1)
    for g in opt.param_groups:
        g["lr"] = warmup_lr


def is_plateaued(history, patience=15, rel_tol=1e-5):
    """Compares the average of the last `patience` evals against the
    average of the `patience` before that - robust to a noisy loss curve."""
    if len(history) < 2 * patience:
        return False
    older_avg = np.mean(history[-2 * patience:-patience])
    recent_avg = np.mean(history[-patience:])
    if older_avg <= 0:
        return True
    return (older_avg - recent_avg) / abs(older_avg) < rel_tol


def train_until_plateau(stage, targets=("q_over_pt",), embed_dim=64, n_layers=2,
                         batch_size=512, max_steps=50000, lr=3e-4, eval_every=EVAL_EVERY,
                         eval_size=20000, patience=15, rel_tol=1e-5, seed=0, warmup_steps=200,
                         run_name=None, ckpt_dir=None, ckpt_every=1000, verbose=True):
    """Trains a TrackTransformer with Adam until eval loss plateaus.
    Resumable via checkpoint. Returns (model, eval_loss_history)."""
    ckpt_dir = ckpt_dir or "./checkpoints"  # no fallback Drive path in script context
    stage_cfg = STAGES[stage]
    run_name = run_name or stage
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pt")
    os.makedirs(ckpt_dir, exist_ok=True)

    stream = make_stream(stage_cfg, batch_size, seed=seed, extra_targets=targets, dtype=NUMPY_DTYPE)
    eval_hits, eval_mask, eval_targets = make_fixed_eval_set(stage_cfg, eval_size, seed=12345, extra_targets=targets, dtype=NUMPY_DTYPE)

    model = TrackTransformer(embed_dim=embed_dim, n_layers=n_layers, targets=targets).to(device=device, dtype=PRECISION)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Reduces LR when eval loss stops improving -- tied to actual training
    # progress, not a fixed step-count curve. patience here (in EVAL checks,
    # not steps) is deliberately shorter than is_plateaued's, so LR drops
    # BEFORE training gives up, not after.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, threshold=1e-4, min_lr=1e-12)

    start_step = 0
    eval_history = []
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"]
        eval_history = ckpt.get("eval_history", [])  # restore curve, not just weights

        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        # else: no fast-forward possible for a plateau-based scheduler
        # (needs real loss history, not a step count) -- starts fresh.

        if verbose: print(f"resumed '{run_name}' from step {start_step}")

    final_step, plateaued = start_step, False
    t0 = time.time()

    for step in range(start_step, max_steps):
        hits, mask, batch_targets = next(stream)
        hits, mask = hits.to(device), mask.to(device)
        pred = model(hits, mask)
        loss = sum(torch.mean((pred[k] - batch_targets[k].to(device)) ** 2) for k in targets)

        if not torch.isfinite(loss):
            print(f"[{run_name}] step {step}: non-finite loss, stopping.")
            break
        if step < warmup_steps:
            set_warmup_lr(opt, lr, step, warmup_steps)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        final_step = step

        if step % eval_every == 0:
            eval_losses = evaluate(model, eval_hits, eval_mask, eval_targets, device)
            combined = sum(eval_losses.values())
            cur_lr = opt.param_groups[0]["lr"]
            eval_history.append({"step": step, "lr": cur_lr, "combined": combined, **eval_losses})
            if step >= warmup_steps:
                scheduler.step(combined)
            if verbose:
                print(f"[{run_name}] step {step:6d} | lr {cur_lr:.2e} | eval {eval_losses} | {time.time()-t0:.1f}s")
            if is_plateaued([h["combined"] for h in eval_history], patience=patience, rel_tol=rel_tol):
                plateaued = True
                if verbose: print(f"[{run_name}] plateaued at step {step}")
                break
        if step % ckpt_every == 0 and step > start_step:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "scheduler": scheduler.state_dict(),
                        "eval_history": eval_history}, ckpt_path)

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": final_step + 1,
                "scheduler": scheduler.state_dict(),
                "eval_history": eval_history}, ckpt_path)
    if verbose and not plateaued:
        print(f"[{run_name}] hit max_steps without plateauing - consider raising max_steps")
    return model, eval_history


def refine_with_lbfgs(model, stage, targets=("q_over_pt",), n=8192, seed=777,
                             maxiter=5000, gtol=1e-20, verbose=True):
    """Second-order refinement via PyTorch L-BFGS, run in one call with
    tolerances forced near zero so it cannot quit early on its own
    convergence check.

    Limited-memory L-BFGS (PyTorch has no full-BFGS option); history_size
    approximates it."""
    refined = copy.deepcopy(model)
    stage_cfg = STAGES[stage]
    hits, mask, batch_targets = make_fixed_eval_set(stage_cfg, n, seed=seed, extra_targets=targets, dtype=NUMPY_DTYPE)
    hits, mask = hits.to(device), mask.to(device)
    batch_targets = {k: v.to(device) for k, v in batch_targets.items()}

    opt = torch.optim.LBFGS(refined.parameters(), lr=1.0, max_iter=maxiter, history_size=100,
                             line_search_fn="strong_wolfe",
                            tolerance_grad=1e-20, tolerance_change=1e-20)

    loss_curve = []

    def closure():
        opt.zero_grad()
        pred = refined(hits, mask)
        loss = sum(torch.mean((pred[k] - batch_targets[k]) ** 2) for k in targets)
        loss.backward()
        loss_curve.append(loss.item())

        # if len(loss_curve) % 10 == 0:
        print(f"  L-BFGS step {len(loss_curve)} | loss: {loss.item():.6e}")

        return loss

    t0 = time.time()
    final_loss = opt.step(closure)
    if not torch.isfinite(final_loss):
        print("  WARNING: L-BFGS diverged. Returning ORIGINAL model unchanged.")
        return model, loss_curve
    if verbose:
        print(f"  PyTorch L-BFGS: {len(loss_curve)} closure evals | {time.time()-t0:.1f}s | "
              f"final loss {loss_curve[-1]:.6e}")
    return refined, loss_curve

print("Training and refinement functions defined.")


# ============================================================================
# Part 5 -- Evaluation
# ============================================================================

def evaluate_full(model, stage, target="q_over_pt", eval_size=10000, seed=12345, label=""):
    """Evaluates a model against the fixed eval set and the matched-precision
    least-squares baseline. Returns a dict of summary metrics."""
    stage_cfg = STAGES[stage]
    hits, mask, truth_t = make_fixed_eval_set(stage_cfg, eval_size, seed=seed, extra_targets=(target,), dtype=NUMPY_DTYPE)
    pred = batched_forward(model, hits, mask, device)
    pred_np, truth_np = pred[target].numpy().astype(np.float64), truth_t[target].numpy().astype(np.float64)

    mse = float(np.mean((pred_np - truth_np) ** 2))
    rmse = float(np.sqrt(mse))
    base = fit_batch_vectorized(hits.numpy(), mask.numpy(), dtype=PRECISION)
    base_mse = float(np.mean((base[target].astype(np.float64) - truth_np) ** 2))
    base_rmse = float(np.sqrt(base_mse))
    ratio = mse / max(base_mse, 1e-300)
    beats_baseline = mse < base_mse

    rel = (pred_np - truth_np) / np.where(truth_np != 0, truth_np, 1.0)
    bias, resolution = robust_bias(rel), robust_resolution(rel)

    # RMSE reported alongside MSE -- matches the Precision ML paper's own
    # unit (e.g. "SciPy BFGS achieves 1e-7 RMSE"), so this is the number to
    # compare against that expectation, not MSE.
    print(f"{label:20s} RMSE={rmse:.3e}  baseline_RMSE={base_rmse:.3e}  MSE={mse:.3e}  baseline_MSE={base_mse:.3e}  "
          f"ratio={ratio:.3e}  res={resolution*100:.3f}%  bias={bias*100:.3f}%  beats_baseline={beats_baseline}")

    return dict(stage=stage, label=label, mse=mse, rmse=rmse, baseline_mse=base_mse, baseline_rmse=base_rmse,
                ratio=ratio, resolution_pct=resolution*100, bias_pct=bias*100, beats_baseline=beats_baseline)



# ============================================================================
# Part 6 -- Sweep, summary, and plots
# ============================================================================

def run_sweep(out_dir, seeds, stages, eval_every=EVAL_EVERY):
    """Trains and evaluates every (stage, seed) combination. Resumable:
    rerunning with the same out_dir skips combinations already in
    sweep_results.csv. Seed-aware -- without this, finishing seed 0 would
    make later seeds silently get skipped too."""
    ckpt_dir = f"{out_dir}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    results_path = f"{out_dir}/sweep_results.csv"

    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        print(f"Loaded {len(results_df)} existing results - resuming.")
    else:
        results_df = pd.DataFrame(columns=["stage", "label", "seed", "wall_time_sec", "mse", "rmse", "baseline_mse",
                                            "baseline_rmse", "ratio", "resolution_pct", "bias_pct", "beats_baseline"])

    sweep_start = time.time()
    total_combos = len(seeds) * len(stages) * 2
    completed_combos = [0]

    def log_progress(tag, block_elapsed):
        completed_combos[0] += 1
        total_elapsed = time.time() - sweep_start
        avg = total_elapsed / completed_combos[0]
        eta = avg * (total_combos - completed_combos[0])
        print(f"  [{tag}] this step: {block_elapsed/60:.1f} min | "
              f"total elapsed: {total_elapsed/60:.1f} min | "
              f"ETA remaining: {eta/60:.1f} min ({completed_combos[0]}/{total_combos} combos)")

    def stage_done(stage, label, seed):
        if len(results_df) == 0:
            return False
        return ((results_df["stage"] == stage) & (results_df["label"] == label) & (results_df["seed"] == seed)).any()

    def save_curve(stage, name, seed, curve):
        """CSV, not .npy -- directly reusable for a different plot later
        without rerunning."""
        path = f"{ckpt_dir}/{stage}_{name}_seed{seed}_curve.csv"
        if len(curve) > 0 and isinstance(curve[0], dict):
            pd.DataFrame(curve).to_csv(path, index=False)
        else:
            pd.DataFrame({"value": curve}).to_csv(path, index=False)

    def load_curve(stage, name, seed, as_dicts=False):
        path = f"{ckpt_dir}/{stage}_{name}_seed{seed}_curve.csv"
        if not os.path.exists(path):
            return []
        df = pd.read_csv(path)
        return df.to_dict("records") if as_dicts else df["value"].tolist()

    adam_curves, bfgs_curves, refined_models = {}, {}, {}

    for seed in seeds:
        for stage in stages:
            print(f"\n=== {stage} (seed {seed}) ===")

            t0 = time.time()
            if not stage_done(stage, "adam_only", seed):
                model, eval_history = train_until_plateau(
                    stage=stage, targets=(TARGET,), seed=seed,
                    run_name=f"{stage}_adam_seed{seed}", ckpt_dir=ckpt_dir, eval_every=eval_every, verbose=True)
                save_curve(stage, "adam", seed, eval_history)
                if seed == seeds[0]:
                    adam_curves[stage] = eval_history
                row = evaluate_full(model, stage, label="adam_only"); row["seed"] = seed
                row["wall_time_sec"] = time.time() - t0
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
                results_df.to_csv(results_path, index=False)
                torch.save(model.state_dict(), f"{ckpt_dir}/{stage}_adam_final_seed{seed}.pt")
            else:
                print("  adam_only: already done, loading saved curve")
                if seed == seeds[0]:
                    adam_curves[stage] = load_curve(stage, "adam", seed, as_dicts=True)
                model = TrackTransformer(targets=(TARGET,)).to(device=device, dtype=PRECISION)
                model.load_state_dict(torch.load(f"{ckpt_dir}/{stage}_adam_final_seed{seed}.pt", map_location=device, weights_only=False))
            log_progress(f"{stage} seed{seed} adam_only", time.time() - t0)

            t0 = time.time()
            if not stage_done(stage, "adam_plus_bfgs", seed):
                refined, bfgs_curve = refine_with_lbfgs(model, stage, targets=(TARGET,), seed=777 + seed, verbose=True)
                save_curve(stage, "bfgs", seed, bfgs_curve)
                if seed == seeds[0]:
                    bfgs_curves[stage] = bfgs_curve
                    refined_models[stage] = refined
                row = evaluate_full(refined, stage, label="adam_plus_bfgs"); row["seed"] = seed
                row["wall_time_sec"] = time.time() - t0
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
                results_df.to_csv(results_path, index=False)
                torch.save(refined.state_dict(), f"{ckpt_dir}/{stage}_refined_final_seed{seed}.pt")
            else:
                print("  adam_plus_bfgs: already done, loading saved curve")
                if seed == seeds[0]:
                    bfgs_curves[stage] = load_curve(stage, "bfgs", seed)
                    refined = TrackTransformer(targets=(TARGET,)).to(device=device, dtype=PRECISION)
                    refined.load_state_dict(torch.load(f"{ckpt_dir}/{stage}_refined_final_seed{seed}.pt", map_location=device, weights_only=False))
                    refined_models[stage] = refined
            log_progress(f"{stage} seed{seed} adam_plus_bfgs", time.time() - t0)

    total_sweep_time = time.time() - sweep_start
    print(f"\nSweep complete in {total_sweep_time/60:.1f} min ({total_sweep_time/3600:.2f} hr). "
          f"{len(results_df)} results saved to {results_path}")
    return results_df, adam_curves, bfgs_curves, refined_models


def make_summary(results_df, stages, out_dir):
    """Aggregates across seeds, prints the summary table, plots RMSE with
    error bars, and reports which stages beat the baseline in every seed."""
    results_df = results_df.copy()
    results_df["stage"] = pd.Categorical(results_df["stage"], categories=stages, ordered=True)
    summary = results_df.groupby(["stage", "label"], observed=True).agg(
        n_seeds=("seed", "nunique"),
        wall_time_min_mean=("wall_time_sec", lambda s: s.mean() / 60),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        baseline_rmse_mean=("baseline_rmse", "mean"), baseline_mse_mean=("baseline_mse", "mean"),
        ratio_mean=("ratio", "mean"), resolution_pct_mean=("resolution_pct", "mean"),
        bias_pct_mean=("bias_pct", "mean"), beats_baseline_all=("beats_baseline", "all"),
    ).reset_index().sort_values(["stage", "label"])

    total_wall_hr = results_df["wall_time_sec"].sum() / 3600
    print(f"Total wall time across all logged combos: {total_wall_hr:.2f} hr\n")
    pd.set_option("display.float_format", lambda x: f"{x:.4g}")
    print(summary.to_string(index=False))
    results_df.to_csv(f"{out_dir}/sweep_summary_final.csv", index=False)
    summary.to_csv(f"{out_dir}/sweep_summary_aggregated.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(stages))
    for label, marker in [("adam_only", "o"), ("adam_plus_bfgs", "s")]:
        sub = summary[summary["label"] == label].set_index("stage").reindex(stages)
        ax.errorbar(x, sub["rmse_mean"], yerr=sub["rmse_std"], marker=marker, capsize=4, label=label)
    baseline_vals = summary[summary["label"] == "adam_only"].set_index("stage").reindex(stages)["baseline_rmse_mean"]
    ax.plot(x, baseline_vals, marker="*", color="green", linestyle="--", label="least-squares baseline")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([s.split("_")[0].replace("stage", "S") for s in stages])
    ax.set_xlabel("curriculum stage"); ax.set_ylabel("RMSE (log scale)")
    ax.set_title(f"Precision across {len(stages)} stages: Adam vs. Adam+BFGS vs. baseline")
    ax.legend()
    plt.savefig(f"{out_dir}/full_sweep_summary.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print("\nStages where the model beat the least-squares baseline in EVERY seed:")
    beats = summary[(summary["label"] == "adam_plus_bfgs") & (summary["beats_baseline_all"])]
    print(beats[["stage"]].to_string(index=False) if len(beats) else "  none")

    make_band_plot(summary, stages, out_dir)
    return summary


def make_band_plot(summary, stages, out_dir):
    """Same data as make_summary()'s errorbar plot, drawn instead as a
    shaded mean +/- std band -- easier to read the seed spread across all
    stages at a glance than discrete whiskers. With a single seed (std=NaN
    for each stage) the band collapses to zero width, same as the errorbar
    plot showing no visible whisker in that case."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(stages))
    colors = {"adam_only": "tab:blue", "adam_plus_bfgs": "tab:red"}
    for label in ["adam_only", "adam_plus_bfgs"]:
        sub = summary[summary["label"] == label].set_index("stage").reindex(stages)
        mean = sub["rmse_mean"].to_numpy()
        std = sub["rmse_std"].fillna(0.0).to_numpy()
        lo = np.maximum(mean - std, mean * 1e-6)  # keep >0 for the log-scale axis below
        ax.plot(x, mean, marker="o", color=colors[label], label=label)
        ax.fill_between(x, lo, mean + std, color=colors[label], alpha=0.2)
    baseline_vals = summary[summary["label"] == "adam_only"].set_index("stage").reindex(stages)["baseline_rmse_mean"]
    ax.plot(x, baseline_vals, marker="*", color="green", linestyle="--", label="least-squares baseline")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([s.split("_")[0].replace("stage", "S") for s in stages])
    ax.set_xlabel("curriculum stage"); ax.set_ylabel("RMSE (log scale)")
    ax.set_title(f"Precision across {len(stages)} stages: mean $\\pm$ std across seeds")
    ax.legend()
    plt.savefig(f"{out_dir}/full_sweep_summary_band.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_kink_plots(adam_curves, bfgs_curves, stages, out_dir, eval_every=EVAL_EVERY):
    n_cols = 3
    n_rows = -(-len(stages) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows))
    for ax in axes.flat[len(stages):]:
        ax.axis("off")
    for ax, stage in zip(axes.flat, stages):
        adam_c, bfgs_c = adam_curves.get(stage, []), bfgs_curves.get(stage, [])
        if not adam_c or not bfgs_c:
            ax.set_title(f"{stage}\n(no curve found)", fontsize=8)
            continue
        adam_losses = [h["combined"] for h in adam_c]  # eval_history entries are rich dicts
        adam_x = np.arange(len(adam_losses)) * eval_every
        bfgs_x = adam_x[-1] + np.arange(1, len(bfgs_c) + 1) if len(adam_x) else np.arange(len(bfgs_c))
        ax.plot(adam_x, adam_losses, label="Adam (real steps)", color="tab:blue", linewidth=1)
        ax.plot(bfgs_x, bfgs_c, label="BFGS (real iterations)", color="tab:red", linewidth=1)
        ax.axvline(adam_x[-1] if len(adam_x) else 0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_title(stage.replace("_", " "), fontsize=9)
        ax.set_xlabel("Adam: step | BFGS: iteration", fontsize=7)
        ax.set_ylabel("loss (log)", fontsize=8)
        if stage == stages[0]:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/kink_plots_all_stages.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)


def make_residual_plots(refined_models, stages, out_dir):
    n_cols = 3
    n_rows = -(-len(stages) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows))
    for ax in axes.flat[len(stages):]:
        ax.axis("off")
    for ax, stage in zip(axes.flat, stages):
        if stage not in refined_models:
            ax.set_title(f"{stage}\n(no model found)", fontsize=8)
            continue
        model = refined_models[stage]
        stage_cfg = STAGES[stage]
        hits, mask, truth_t = make_fixed_eval_set(stage_cfg, 50000, seed=12345, extra_targets=(TARGET,), dtype=NUMPY_DTYPE)
        pred = batched_forward(model, hits, mask, device)
        pred_np = pred[TARGET].numpy().astype(np.float64)
        truth_np = truth_t[TARGET].numpy().astype(np.float64)
        rel = (pred_np - truth_np) / np.where(truth_np != 0, truth_np, 1.0)
        bias, resolution = robust_bias(rel), robust_resolution(rel)
        lo, hi = np.percentile(rel, [0.5, 99.5])
        if lo == hi:
            lo, hi = -1e-3, 1e-3
        ax.hist(rel, bins=80, range=(lo, hi), density=True, alpha=0.7, color="tab:blue")
        half_iqr = resolution * 1.349 / 2
        ax.axvline(bias, color="red", linewidth=1.5, label="median")
        ax.axvline(bias - half_iqr, color="red", linewidth=1, linestyle="--", label="IQR/2")
        ax.axvline(bias + half_iqr, color="red", linewidth=1, linestyle="--")
        ax.set_title(f"{stage.replace(chr(95),chr(32))}\nres={resolution*100:.3f}%  bias={bias*100:.3f}%", fontsize=8)
        ax.set_xlabel("(pred - truth) / truth", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/residual_distributions_all_stages.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Precision curriculum sweep")
    parser.add_argument("--out-dir", required=True, help="output directory for checkpoints, results, plots")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--stages", nargs="+", default=STAGE_ORDER, choices=STAGE_ORDER,
                         help="subset of stages to run (default: all)")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float32")
    parser.add_argument("--make-plots", action="store_true", default=True)
    args = parser.parse_args()

    global PRECISION, NUMPY_DTYPE
    PRECISION = torch.float64 if args.precision == "float64" else torch.float32
    NUMPY_DTYPE = np.float64 if args.precision == "float64" else np.float32
    print(f"Working precision: {PRECISION}")

    os.makedirs(args.out_dir, exist_ok=True)

    class _Tee:
        """Mirrors every print() to both stdout and a durable log file --
        SLURM's --output already captures stdout, but this keeps a copy
        inside --out-dir itself, self-contained with the run's own artifacts."""
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data); s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    log_path = f"{args.out_dir}/run_log.txt"
    log_file = open(log_path, "a")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    print(f"Logging to: {log_path}")

    # Sanity check -- fit correctness at high precision, independent of the run's own precision.
    _rng = np.random.default_rng(1)
    _hits, _mask, _targets = generate_batch(500, _rng, **STAGES["stage2_varying_momentum"], dtype=np.float64)
    _fit = fit_batch_vectorized(_hits, _mask, dtype=torch.float64)
    _err = _fit["q_over_pt"] - _targets["q_over_pt"]
    print(f"Sanity check (should be ~1e-10 or smaller): mean={_err.mean():.3e}  std={_err.std():.3e}")

    results_df, adam_curves, bfgs_curves, refined_models = run_sweep(args.out_dir, args.seeds, args.stages)
    make_summary(results_df, args.stages, args.out_dir)

    if args.make_plots:
        make_kink_plots(adam_curves, bfgs_curves, args.stages, args.out_dir)
        make_residual_plots(refined_models, args.stages, args.out_dir)
        print(f"\nPlots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
