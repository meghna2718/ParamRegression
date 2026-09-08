#!/bin/bash
#SBATCH --job-name=TF_exp1_extra
#SBATCH --output=TF_exp1_extra_%A_%a.out
#SBATCH --error=TF_exp1_extra_%A_%a.err
#SBATCH --time=09:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-3

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env
cd /home/xucabm80/ParamRegression

# Seeds 1-2 for pos-encoding none/learnable only -- sinusoidal is excluded
# since it collapsed (Sec. exp2_posenc) and isn't reported beyond seed 0.
# Same recipe as exp1_posenc_fixed.slurm (seed 0, already used in the
# thesis): d128/8-layer, features r dphi z, norm_loss=none, weight_decay
# 0.01, epochs=150 @ lr=1e-3 -- only --pos-encoding and --seed vary.

TASKS=(
  "none       1"
  "none       2"
  "learnable  1"
  "learnable  2"
)
read -r POS_ENC SEED <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "Task $SLURM_ARRAY_TASK_ID: pos_encoding=$POS_ENC seed=$SEED"

python -u train_trackformer.py \
    --channel ttbar --pileup pu200 \
    --max-events 950 --train-events 800 --test-events 100 \
    --features r dphi z \
    --d-model 128 --nhead 4 --num-layers 8 \
    --pos-encoding ${POS_ENC} \
    --head-mode shared \
    --norm-loss none --weight-decay 0.01 \
    --seed ${SEED} --batch-size 128 --epochs 150 --lr 1e-3 \
    --early-stop-patience 20 \
    --eval-acts \
    --out-dir runs/exp1_fixed_posenc_${POS_ENC}_seed${SEED}

python analyze_results.py --run-dir runs/exp1_fixed_posenc_${POS_ENC}_seed${SEED}

# After all 4 finish, combine with the existing seed0 runs:
#   python compare_runs.py \
#       --run-dirs runs/exp1_fixed_posenc_none_seed0 runs/exp1_fixed_posenc_none_seed1 runs/exp1_fixed_posenc_none_seed2 \
#                   runs/exp1_fixed_posenc_learnable_seed0 runs/exp1_fixed_posenc_learnable_seed1 runs/exp1_fixed_posenc_learnable_seed2 \
#       --out-dir runs/exp1_fixed_comparison_3seed
