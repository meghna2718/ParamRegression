#!/usr/bin/env python
"""
diagnose_theta_phi.py
Standalone diagnostic -- checks whether restore_absolute_z()/restore_absolute_phi()
in train_trackformer.py are mathematically correct, by round-tripping the TRUTH
theta/phi values (not model predictions) through the same formulas and comparing
against what got saved. If truth round-trips cleanly, the restore code is fine and
the bad theta/phi resolutions are a model-learning problem, not a restore bug.

Usage:
    python diagnose_theta_phi.py --run-dir runs/exp1_posenc_sinusoidal_seed0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def wrap(x):
    """Wrap to (-pi, pi], same convention used throughout train_trackformer.py."""
    x = np.where(x > np.pi, x - 2 * np.pi, x)
    x = np.where(x < -np.pi, x + 2 * np.pi, x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    pre = pd.read_pickle(run_dir / "test_data.pkl")
    post = pd.read_pickle(run_dir / "test_data_with_predictions.pkl")
    print(f"Loaded {len(pre)} rows from test_data.pkl, {len(post)} from test_data_with_predictions.pkl")
    assert len(pre) == len(post), "Row count mismatch -- pre/post aren't the same test set!"

    # --- Step 1: z_sign sanity check ---
    print("\n--- z_sign distribution ---")
    print(pre["z_sign"].value_counts(dropna=False))

    # --- Step 2: phi_offset sanity check ---
    print("\n--- phi_offset stats ---")
    print(pre["phi_offset"].describe())

    # --- Step 3: round-trip TRUTH theta through the restore formula by hand ---
    # pre["theta"] = theta_signed (the z-symmetry-canonicalized TRAINING target)
    # post["theta"] = what restore_absolute_z() produced from it
    theta_recon = np.arccos(np.clip(pre["z_sign"].to_numpy() * np.cos(pre["theta"].to_numpy()), -1.0, 1.0))
    theta_diff = np.abs(theta_recon - post["theta"].to_numpy())
    print("\n--- theta round-trip check (manual formula vs. saved restored value) ---")
    print(f"max |diff|: {theta_diff.max():.6g}   mean |diff|: {theta_diff.mean():.6g}")
    print("PASS (formula matches saved output)" if theta_diff.max() < 1e-4 else "MISMATCH -- restore_absolute_z has a bug")

    # --- Step 4: round-trip TRUTH phi through the restore formula by hand ---
    # pre["phi"] = dphi0 (relative to phi_offset, the TRAINING target)
    # post["phi"] = what restore_absolute_phi() produced from it
    phi_recon = wrap(pre["phi"].to_numpy() + pre["phi_offset"].to_numpy())
    phi_diff = np.abs(wrap(phi_recon - post["phi"].to_numpy()))
    print("\n--- phi round-trip check (manual formula vs. saved restored value) ---")
    print(f"max |diff|: {phi_diff.max():.6g}   mean |diff|: {phi_diff.mean():.6g}")
    print("PASS (formula matches saved output)" if phi_diff.max() < 1e-4 else "MISMATCH -- restore_absolute_phi has a bug")

    # --- Step 5: what do the SIGNED/RELATIVE training targets actually look like? ---
    # (Independent of restore correctness -- tells us if the pre-restore target
    # itself is well-behaved, e.g. theta_signed should be within [0, pi/2].)
    print("\n--- pre-restore theta (training target, should be within [0, pi/2]) ---")
    print(pre["theta"].describe())
    print("\n--- pre-restore phi / dphi0 (training target, should be within (-pi, pi]) ---")
    print(pre["phi"].describe())

    # --- Step 6: how does ml_theta / ml_phi (absolute, post-restore) actually compare to truth? ---
    print("\n--- theta residual (ml_theta - theta), post-restore, absolute frame ---")
    theta_res = post["ml_theta"].to_numpy() - post["theta"].to_numpy()
    print(pd.Series(theta_res).describe())

    print("\n--- phi residual (wrapped), post-restore, absolute frame ---")
    phi_res = wrap(post["ml_phi"].to_numpy() - post["phi"].to_numpy())
    print(pd.Series(phi_res).describe())


if __name__ == "__main__":
    main()
