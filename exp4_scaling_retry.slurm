#!/bin/bash
#SBATCH --job-name=TF_exp4_retry
#SBATCH --output=TF_exp4_retry_%A_%a.out
#SBATCH --error=TF_exp4_retry_%A_%a.err
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-1            # data_large retry, model_large retry

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env

cd /home/xucabm80/ParamRegression

# Retry of the two exp4 points that failed to converge (see run_summary.json
# in the _v1 dirs: model_large never found a good minimum at all; data_large
# found one at epoch 5 then collapsed). New RUN_TAG so the failed v1 runs
# aren't overwritten -- keep them, they're useful evidence for the writeup
# either way.
#
# data_large (task 0): same config, just picks up the new default
# --warmup-frac 0.05 automatically (no flag needed) -- this was a steps-per-
# epoch accounting issue, should be directly fixed.
#
# model_large (task 1): warmup-frac alone likely isn't enough for a 12-layer/
# 256-dim model that never converged even once -- also drop LR and push
# warmup-frac higher as an explicit safety margin for this one point.

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    python -u train_trackformer.py \
        --channel ttbar --pileup pu200 \
        --max-events 1650 --train-events 1500 --test-events 100 \
        --features r dphi z \
        --d-model 128 --nhead 4 --num-layers 8 \
        --pos-encoding sinusoidal --head-mode shared \
        --norm-loss none --weight-decay 0.01 \
        --seed 0 --batch-size 128 --epochs 300 --lr 1e-3 \
        --early-stop-patience 20 \
        --eval-acts \
        --out-dir runs/exp4_scale_data_large_v2
    python analyze_results.py --run-dir runs/exp4_scale_data_large_v2
else
    python -u train_trackformer.py \
        --channel ttbar --pileup pu200 \
        --max-events 950 --train-events 800 --test-events 100 \
        --features r dphi z \
        --d-model 256 --nhead 4 --num-layers 12 \
        --pos-encoding sinusoidal --head-mode shared \
        --norm-loss none --weight-decay 0.01 \
        --seed 0 --batch-size 128 --epochs 300 --lr 5e-4 \
        --warmup-frac 0.10 \
        --early-stop-patience 20 \
        --eval-acts \
        --out-dir runs/exp4_scale_model_large_v2
    python analyze_results.py --run-dir runs/exp4_scale_model_large_v2
fi
