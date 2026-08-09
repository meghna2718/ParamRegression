#!/bin/bash
#SBATCH --job-name=TF_exp2_head
#SBATCH --output=TF_exp2_head_%A_%a.out
#SBATCH --error=TF_exp2_head_%A_%a.err
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-1            # 2 seeds of "separate" only -- "shared" baseline == exp1_posenc_none_seed0

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env

cd /home/xucabm80/ParamRegression
pip install -q -r requirements.txt   # one-time; harmless if already installed

# EXPERIMENT 2: shared regression head vs. independent head per parameter, pos-encoding=none.
SEEDS=(0 1)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# --particle-types 13 to restrict to muons only; empty = all primary species.
PARTICLE_TYPE_FLAG=""

python -u train_trackformer.py \
    --channel ttbar --pileup pu200 \
    --max-events 950 --train-events 800 --test-events 100 \
    --features r dphi z \
    --d-model 128 --nhead 4 --num-layers 8 \
    --pos-encoding none \
    --head-mode separate \
    --seed ${SEED} \
    --batch-size 128 --epochs 400 --lr 1e-3 \
    --eval-acts \
    ${PARTICLE_TYPE_FLAG} \
    --out-dir runs/exp2_head_separate_seed${SEED}

python analyze_results.py --run-dir runs/exp2_head_separate_seed${SEED}

# Compare against exp1_posenc_none_seed0 (shared head, seed 0 only):
#   python compare_runs.py \
#       --run-dirs runs/exp1_posenc_none_seed0 runs/exp2_head_separate_seed0 runs/exp2_head_separate_seed1 \
#       --labels   shared                      separate                     separate \
#       --out-dir runs/exp2_head_comparison
