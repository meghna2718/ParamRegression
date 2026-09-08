"""Merge the three per-seed curriculum_sweep out-dirs (seed0/1/2) back into
one combined sweep_results.csv, then reruns make_summary() on the merged
data to get the proper cross-seed mean/std and the aggregate plot.

Usage (run from /home/xucabm80/ParamRegression, after all 3 array tasks finish):
    python merge_curriculum_seeds.py --seeds 0 1 2 \
        --seed-dir-pattern "runs/curriculum_sweep_seed{seed}" \
        --merged-out-dir runs/curriculum_sweep_merged
"""

import argparse
import os
import pandas as pd

from train_curriculum import make_summary, STAGE_ORDER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seed-dir-pattern", default="runs/curriculum_sweep_seed{seed}")
    parser.add_argument("--merged-out-dir", default="runs/curriculum_sweep_merged")
    parser.add_argument("--stages", nargs="+", default=STAGE_ORDER)
    args = parser.parse_args()

    frames = []
    for seed in args.seeds:
        seed_dir = args.seed_dir_pattern.format(seed=seed)
        path = f"{seed_dir}/sweep_results.csv"
        if not os.path.exists(path):
            print(f"  missing: {path} -- skipping seed {seed}")
            continue
        frames.append(pd.read_csv(path))
        print(f"  loaded {path} ({len(frames[-1])} rows)")

    if not frames:
        raise SystemExit("No per-seed results found -- check --seed-dir-pattern.")

    merged = pd.concat(frames, ignore_index=True)
    os.makedirs(args.merged_out_dir, exist_ok=True)
    merged.to_csv(f"{args.merged_out_dir}/sweep_results.csv", index=False)
    print(f"\nMerged {len(merged)} rows from {len(frames)} seed(s) -> {args.merged_out_dir}/sweep_results.csv")

    make_summary(merged, args.stages, args.merged_out_dir)


if __name__ == "__main__":
    main()
