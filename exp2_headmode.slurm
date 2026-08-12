#!/bin/bash
#SBATCH --job-name=TF_exp2_head
#SBATCH --output=TF_exp2_head_%A_%a.out
#SBATCH --error=TF_exp2_head_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-5            # 2 head modes x 3 seeds

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env

cd /home/xucabm80/ParamRegression
pip install -q -r requirements.txt   # one-time; harmless if already installed

# EXPERIMENT 2: shared vs. per-parameter regression head, 3 seeds each for a
# real mean +/- std comparison. pos-encoding fixed to sinusoidal. Epochs
# bumped 60 -> 150: last run's best epoch was the LAST epoch trained in both
# cases (60/60), with val loss still dropping -- a sign training was cut off
# before it actually converged, not that it had plateaued.
#
# RUN_TAG: bump this string for any future rerun of this script so new
# results land in fresh runs/ directories instead of overwriting these ones
# (e.g. change to "v3" next time). No manual `mv`-ing of old run folders needed.
RUN_TAG="v2_150ep"

HEAD_MODES=(shared separate)
SEEDS=(0 1 2)
N_SEEDS=${#SEEDS[@]}
HEAD_MODE=${HEAD_MODES[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}

# --particle-types 13 to restrict to muons only; empty = all primary species.
PARTICLE_TYPE_FLAG=""

python -u train_trackformer.py \
    --channel ttbar --pileup pu200 \
    --max-events 950 --train-events 800 --test-events 100 \
    --features r dphi z \
    --d-model 128 --nhead 4 --num-layers 8 \
    --pos-encoding sinusoidal \
    --head-mode ${HEAD_MODE} \
    --seed ${SEED} \
    --batch-size 128 --epochs 150 --lr 1e-3 \
    --eval-acts \
    ${PARTICLE_TYPE_FLAG} \
    --out-dir runs/exp2_head_${HEAD_MODE}_${RUN_TAG}_seed${SEED}

python analyze_results.py --run-dir runs/exp2_head_${HEAD_MODE}_${RUN_TAG}_seed${SEED}

# After all 6 array tasks finish, compare (repeated labels -> mean +/- std per head_mode):
#   python compare_runs.py \
#       --run-dirs runs/exp2_head_shared_v2_150ep_seed0   runs/exp2_head_shared_v2_150ep_seed1   runs/exp2_head_shared_v2_150ep_seed2 \
#                  runs/exp2_head_separate_v2_150ep_seed0 runs/exp2_head_separate_v2_150ep_seed1 runs/exp2_head_separate_v2_150ep_seed2 \
#       --labels   shared shared shared separate separate separate \
#       --out-dir runs/exp2_head_comparison_v2_150ep
