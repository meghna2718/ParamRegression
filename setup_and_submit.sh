#!/bin/bash
set -e   # stop immediately if any step fails, so we don't submit jobs onto a bad cache

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env
cd /home/xucabm80/ParamRegression

echo "=== $(date) : cancelling old jobs ==="
scancel -u xucabm80 || true
sleep 5

echo "=== $(date) : downloading shards (particles, tracker_hits, tracks; ~7000 events) ==="
colliderml download --out collider_data --channels ttbar --pileup pu200 \
    --objects particles,tracker_hits,tracks --max-events 7000

echo "=== $(date) : verifying shard counts ==="
for cfg in ttbar_pu200_particles ttbar_pu200_tracker_hits ttbar_pu200_tracks; do
    n=$(find collider_data -path "*${cfg}/data/*/train-*.parquet" | wc -l)
    echo "$cfg: $n shard files found"
    if [ "$n" -lt 60 ]; then
        echo "FATAL: only $n shards for $cfg, expected ~70 -- aborting before submitting jobs."
        exit 1
    fi
done

echo "=== $(date) : submitting jobs ==="
sbatch exp2_headmode.slurm
sbatch --array=1,2 exp5_data_push.slurm

echo "=== $(date) : done, check with squeue -u xucabm80 ==="
