#!/bin/bash
#SBATCH --job-name=TF_exp3_fix
#SBATCH --output=TF_exp3_fix_%A_%a.out
#SBATCH --error=TF_exp3_fix_%A_%a.err
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0              # single seed for now -- bump to 0-2 later for mean+/-std

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env

cd /home/xucabm80/ParamRegression
pip install -q -r requirements.txt   # one-time; harmless if already installed

# EXPERIMENT 3: targeted fix for z0/qop + training-length uncertainty, on top
# of the best-so-far config (shared head, sinusoidal pos-enc, 128-dim/8-layer).
# Changes vs. exp2:
#   --norm-loss none      : his config never std-normalizes any parameter (not
#                            just angles); z0/qop are our two weak params, so
#                            this directly targets them.
#   --weight-decay 0.01    : matches AdamW's default (his config never
#                            overrides it); ours was 1e-4, 100x weaker.
#   --epochs 300            : raised ceiling since last run was still improving
#                            at 60/60 and 150/150 -- but paired with...
#   --early-stop-patience 20: ...so it stops automatically once val_loss
#                            plateaus instead of running the full 300 and
#                            risking the mid-training collapse seen before.
# Everything else (batch size, LR, architecture) held fixed vs. exp2 so this
# stays a clean, interpretable test of just these three changes.
RUN_TAG="v3_stability_fix"

SEEDS=(0 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python -u train_trackformer.py \
    --channel ttbar --pileup pu200 \
    --max-events 950 --train-events 800 --test-events 100 \
    --features r dphi z \
    --d-model 128 --nhead 4 --num-layers 8 \
    --pos-encoding sinusoidal \
    --head-mode shared \
    --norm-loss none \
    --weight-decay 0.01 \
    --seed ${SEED} \
    --batch-size 128 --epochs 300 --lr 1e-3 \
    --early-stop-patience 20 \
    --eval-acts \
    --out-dir runs/exp3_fix_${RUN_TAG}_seed${SEED}

python analyze_results.py --run-dir runs/exp3_fix_${RUN_TAG}_seed${SEED}

# After all 3 array tasks finish, aggregate mean +/- std:
#   python compare_runs.py \
#       --run-dirs runs/exp3_fix_v3_stability_fix_seed0 runs/exp3_fix_v3_stability_fix_seed1 runs/exp3_fix_v3_stability_fix_seed2 \
#       --labels   exp3_fix exp3_fix exp3_fix \
#       --out-dir runs/exp3_fix_comparison
