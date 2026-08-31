#!/bin/bash
#SBATCH --job-name=TF_exp9b_lrfix
#SBATCH --output=TF_exp9b_lrfix_%A_%a.out
#SBATCH --error=TF_exp9b_lrfix_%A_%a.err
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

# d192/d256 rerun, each at two depths: 8 layers (same depth as the
# stable d128 baseline, width-only increase) and the original deeper
# config (10/12 layers) with a depth-corrected LR. d32-d128 from exp9
# were fine and are not touched here.
#
# LR correction rationale:
#  - width-only (8L): mild cut vs. d128's 1e-3 (~sqrt(d_base/d_new)),
#    since even at fixed depth, standard (non-muP) init needs a small
#    downward LR correction as width grows.
#  - depth-increased (10L/12L): much larger cut + longer warmup, since
#    exp9 showed collapse scales mainly with layer count (post-LN
#    residual-stream instability), not width.

# label            d_model  num_layers  lr      warmup_frac
TASKS=(
  "model_d192_L8     192      8           8e-4    0.05"
  "model_d192_L10    192      10          2e-4    0.15"
  "model_d256_L8     256      8           7e-4    0.05"
  "model_d256_L12    256      12          1e-4    0.15"
)
read -r LABEL D_MODEL NUM_LAYERS LR WARMUP_FRAC <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "Task $SLURM_ARRAY_TASK_ID: $LABEL  d_model=$D_MODEL  num_layers=$NUM_LAYERS  lr=$LR  warmup_frac=$WARMUP_FRAC"

python -u train_trackformer.py \
    --channel ttbar --pileup pu200 \
    --max-events 950 --train-events 800 --test-events 100 \
    --features r dphi z \
    --d-model ${D_MODEL} --nhead 4 --num-layers ${NUM_LAYERS} \
    --pos-encoding none --head-mode shared \
    --norm-loss none --weight-decay 0.01 \
    --seed 0 --batch-size 128 --epochs 200 --lr ${LR} \
    --warmup-frac ${WARMUP_FRAC} \
    --early-stop-patience 30 \
    --eval-acts \
    --out-dir runs/exp9b_scale_${LABEL}_lrfix_noposenc

python analyze_results.py --run-dir runs/exp9b_scale_${LABEL}_lrfix_noposenc

# Replot once you've picked which of these (if any) to keep:
#   python scaling_plot.py \
#       --model-sweep-dirs runs/exp9_scale_model_d32_noposenc runs/exp9_scale_model_d64_noposenc \
#                           runs/exp9_scale_model_d96_noposenc runs/exp9_scale_model_d128_noposenc \
#                           runs/exp9b_scale_model_d192_L8_lrfix_noposenc runs/exp9b_scale_model_d192_L10_lrfix_noposenc \
#                           runs/exp9b_scale_model_d256_L8_lrfix_noposenc runs/exp9b_scale_model_d256_L12_lrfix_noposenc \
#       --out-dir runs/exp9_scaling_plots_lrfix \
#       --acts-ref-dir runs/exp9_scale_model_d128_noposenc
