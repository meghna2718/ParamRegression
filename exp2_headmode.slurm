#!/bin/bash
#SBATCH --job-name=TF_exp2_head
#SBATCH --output=TF_exp2_head_%A_%a.out
#SBATCH --error=TF_exp2_head_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-1            # 0 = shared head, 1 = separate head -- both trained fresh, identical config

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env

cd /home/xucabm80/ParamRegression
pip install -q -r requirements.txt   # one-time; harmless if already installed

# EXPERIMENT 2: shared vs. per-parameter regression head. pos-encoding fixed
# to sinusoidal (best of exp1's 3). Both trained fresh at 60 epochs (not
# reusing exp1's run) so head_mode is the only thing that differs.
HEAD_MODES=(shared separate)
HEAD_MODE=${HEAD_MODES[$SLURM_ARRAY_TASK_ID]}

# --particle-types 13 to restrict to muons only; empty = all primary species.
PARTICLE_TYPE_FLAG=""

python -u train_trackformer.py \
    --channel ttbar --pileup pu200 \
    --max-events 950 --train-events 800 --test-events 100 \
    --features r dphi z \
    --d-model 128 --nhead 4 --num-layers 8 \
    --pos-encoding sinusoidal \
    --head-mode ${HEAD_MODE} \
    --seed 0 \
    --batch-size 128 --epochs 60 --lr 1e-3 \
    --eval-acts \
    ${PARTICLE_TYPE_FLAG} \
    --out-dir runs/exp2_head_${HEAD_MODE}_seed0

python analyze_results.py --run-dir runs/exp2_head_${HEAD_MODE}_seed0

# After both tasks finish, compare:
#   python compare_runs.py \
#       --run-dirs runs/exp2_head_shared_seed0 runs/exp2_head_separate_seed0 \
#       --labels   shared                      separate \
#       --out-dir runs/exp2_head_comparison
