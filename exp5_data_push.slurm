#!/bin/bash
#SBATCH --job-name=TF_exp5_data
#SBATCH --output=TF_exp5_data_%A_%a.out
#SBATCH --error=TF_exp5_data_%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-2%1          # %1 = run one task at a time -- data_3000/6000 previously
                                # OOM'd (Rust allocator failing on ~1MB allocs) when both
                                # landed on the same node concurrently, combined memory
                                # exceeding what was actually available despite --mem=200G each

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env

cd /home/xucabm80/ParamRegression

# EXPERIMENT 5: push the data-size axis further, using the LR/warmup combo
# that fixed model_large's instability (lr=5e-4, warmup-frac=0.10) since
# data_large needed the same treatment, not just the warmup-frac auto-scale.
# Architecture, batch size, and everything else held fixed at the anchor's
# values so these stay directly comparable to the rest of the data-size
# sweep -- only train_events changes.
#
# --mem bumped to 200G and --time to 20h since larger event counts mean more
# RAM for the raw tables and longer data-download/load time up front.
#
# Task 2 (6000 events) is the real "find the limit" point -- if it OOMs or
# times out, that tells us where the ceiling is, which is itself useful
# information for the writeup.

TASKS=(
  "data_1500_v2   1500   1650"
  "data_3000       3000   3150"
  "data_6000       6000   6150"
)
read -r LABEL TRAIN_EVENTS MAX_EVENTS <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "Task $SLURM_ARRAY_TASK_ID: $LABEL  train_events=$TRAIN_EVENTS  max_events=$MAX_EVENTS"

python -u train_trackformer.py \
    --channel ttbar --pileup pu200 \
    --max-events ${MAX_EVENTS} --train-events ${TRAIN_EVENTS} --test-events 100 \
    --features r dphi z \
    --d-model 128 --nhead 4 --num-layers 8 \
    --pos-encoding sinusoidal --head-mode shared \
    --norm-loss none --weight-decay 0.01 \
    --seed 0 --batch-size 128 --epochs 300 --lr 5e-4 \
    --warmup-frac 0.10 \
    --early-stop-patience 20 \
    --eval-acts \
    --out-dir runs/exp5_${LABEL}

python analyze_results.py --run-dir runs/exp5_${LABEL}

# After all 3 finish, combine with the rest of the data-size sweep and replot:
#   python scaling_plot.py \
#       --model-sweep-dirs runs/exp4_scale_model_small_v1 runs/exp4_scale_model_medium_v1 \
#                           runs/exp3_fix_v3_stability_fix_seed0 runs/exp4_scale_model_large_v2 \
#       --data-sweep-dirs  runs/exp4_scale_data_small_v1 runs/exp4_scale_data_medium_v1 \
#                           runs/exp3_fix_v3_stability_fix_seed0 runs/exp5_data_1500_v2 \
#                           runs/exp5_data_3000 runs/exp5_data_6000 \
#       --out-dir runs/exp5_scaling_plots
