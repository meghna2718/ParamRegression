#!/bin/bash
#SBATCH --job-name=TF_exp2_head
#SBATCH --output=TF_exp2_head_%A_%a.out
#SBATCH --error=TF_exp2_head_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-1            # 2 seeds of "separate" only
#
# "shared" + pos-encoding none, seeds 0 and 1, are already covered by
# exp1_posenc_none_seed{0,1} (identical config) -- reused, not retrained.
#
# PREREQUISITE: exp1_posenc.slurm's "none" tasks (seeds 0 and 1) must have
# already finished and produced runs/exp1_posenc_none_seed{0,1}/ before you
# run the comparison step at the bottom of this file.

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env

cd /home/xucabm80/ParamRegression
pip install -q -r requirements.txt   # one-time; harmless if already installed

# EXPERIMENT 2: Shared regression head (Jeremy's design) vs. independent head
# per parameter, both with pos-encoding=none (matching Jeremy's actual
# configs -- positional_encoding: null in his model yamls), 2 seeds each.
SEEDS=(0 1)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python train_trackformer.py \
    --channel ttbar --pileup pu200 \
    --max-events 1700 --train-events 1500 --test-events 150 \
    --features r dphi z \
    --d-model 128 --nhead 4 --num-layers 4 \
    --pos-encoding none \
    --head-mode separate \
    --seed ${SEED} \
    --batch-size 128 --epochs 400 --lr 1e-3 \
    --eval-acts \
    --out-dir runs/exp2_head_separate_seed${SEED}

python analyze_results.py --run-dir runs/exp2_head_separate_seed${SEED}

# Aggregate against the reused "shared" runs from exp1 (do NOT retrain them):
#   python compare_runs.py \
#       --run-dirs runs/exp1_posenc_none_seed0  runs/exp1_posenc_none_seed1 \
#                  runs/exp2_head_separate_seed0 runs/exp2_head_separate_seed1 \
#       --labels   shared shared separate separate \
#       --out-dir runs/exp2_head_comparison
