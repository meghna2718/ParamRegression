#!/bin/bash
#SBATCH --job-name=TF_exp9_scale
#SBATCH --output=TF_exp9_scale_%A_%a.out
#SBATCH --error=TF_exp9_scale_%A_%a.err
#SBATCH --time=72:00:00
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

# Model-size scaling sweep. Epochs/patience fixed across all configs so
# architecture is the only real variable; lr/warmup staggered down/up with
# size since that's needed for stability, not a free choice.

# label          d_model  num_layers  lr      warmup_frac
TASKS=(
  "model_large     256      12          5e-4    0.10"
  "model_xlarge    384      16          4e-4    0.10"
  "model_xxlarge   512      16          3.5e-4  0.12"
  "model_huge      640      20          3e-4    0.15"
)
read -r LABEL D_MODEL NUM_LAYERS LR WARMUP_FRAC <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "Task $SLURM_ARRAY_TASK_ID: $LABEL  d_model=$D_MODEL  num_layers=$NUM_LAYERS  lr=$LR  warmup_frac=$WARMUP_FRAC"

python -u train_trackformer.py \
    --channel ttbar --pileup pu200 \
    --max-events 950 --train-events 800 --test-events 100 \
    --features r dphi z \
    --d-model ${D_MODEL} --nhead 4 --num-layers ${NUM_LAYERS} \
    --pos-encoding sinusoidal --head-mode shared \
    --norm-loss none --weight-decay 0.01 \
    --seed 0 --batch-size 128 --epochs 200 --lr ${LR} \
    --warmup-frac ${WARMUP_FRAC} \
    --early-stop-patience 30 \
    --eval-acts \
    --out-dir runs/exp9_scale_${LABEL}

python analyze_results.py --run-dir runs/exp9_scale_${LABEL}

# Replot after all finish (anchor = exp2_head_shared_v3_150ep_fixed_seed1).
# --acts-ref-dir adds a flat ACTS reference line; drop the flag to remove it.
#   python scaling_plot.py \
#       --model-sweep-dirs runs/exp4_scale_model_small_v1 runs/exp4_scale_model_medium_v1 \
#                           runs/exp4_scale_model_gap_v1 runs/exp2_head_shared_v3_150ep_fixed_seed1 \
#                           runs/exp9_scale_model_large runs/exp9_scale_model_xlarge \
#                           runs/exp9_scale_model_xxlarge runs/exp9_scale_model_huge \
#       --out-dir runs/exp9_scaling_plots \
#       --acts-ref-dir runs/exp2_head_shared_v3_150ep_fixed_seed1
