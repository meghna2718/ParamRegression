#!/bin/bash
#SBATCH --job-name=TF_exp9c_lrtest
#SBATCH --output=TF_exp9c_lrtest_%A_%a.out
#SBATCH --error=TF_exp9c_lrtest_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-2

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env
cd /home/xucabm80/ParamRegression

# exp9c: tests whether exp9b's d256 plateau (worse val_loss than d192,
# despite being bigger) was caused by an LR that was too conservative
# for the epoch budget -- not by depth instability, since exp9b's own
# loss curves show smooth, stable convergence with no spikes.
#
# Task 0 is the direct re-test: same d256/L12 architecture as exp9b,
# LR moved back up to 0.0002 (matching d192's LR) instead of 0.0001.
# Tasks 1-2 extend the same corrected LR logic to two bigger configs,
# continuing the d192->d256 step size (+64 width, +2 layers), to see
# whether the earlier "bigger benefits once LR is un-derated" trend
# keeps holding at larger scale, or where it eventually breaks.
#
# LR is kept close to d192/d256's corrected value rather than cut
# further per step -- the previous, more aggressive LR-cut schedule
# (exp9b) is what produced the underpowered d256 result in the first
# place, so this run deliberately does not repeat that pattern.

# label            d_model  num_layers  lr      warmup_frac
TASKS=(
  "model_d256_L12_lr2e4   256      12          2e-4    0.15"
  "model_d320_L14         320      14          1.5e-4  0.15"
  "model_d384_L16         384      16          1e-4    0.15"
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
    --out-dir runs/exp9c_scale_${LABEL}_noposenc

python analyze_results.py --run-dir runs/exp9c_scale_${LABEL}_noposenc

# Replot with the full corrected scaling curve once these finish:
#   python scaling_plot.py \
#       --model-sweep-dirs runs/exp9_scale_model_d32_noposenc runs/exp9_scale_model_d64_noposenc \
#                           runs/exp9_scale_model_d96_noposenc runs/exp9_scale_model_d128_noposenc \
#                           runs/exp9b_scale_model_d192_L10_lrfix_noposenc \
#                           runs/exp9c_scale_model_d256_L12_lr2e4_noposenc \
#                           runs/exp9c_scale_model_d320_L14_noposenc \
#                           runs/exp9c_scale_model_d384_L16_noposenc \
#       --out-dir runs/exp9c_scaling_plots \
#       --acts-ref-dir runs/exp9_scale_model_d128_noposenc
