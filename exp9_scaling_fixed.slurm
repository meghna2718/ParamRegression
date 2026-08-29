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
#SBATCH --array=0-7

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env
cd /home/xucabm80/ParamRegression

# Full model-size scaling sweep. pos-encoding=none (sinusoidal collapsed
# under this recipe). Same 200ep/patience-30 schedule for all points.

# label          d_model  num_layers  lr      warmup_frac
TASKS=(
  "model_d32       32       2           1e-3    0.05"
  "model_d64       64       4           1e-3    0.05"
  "model_d96       96       6           1e-3    0.05"
  "model_d128      128      8           1e-3    0.05"
  "model_d192      192      10          7e-4    0.07"
  "model_d256      256      12          5e-4    0.10"
  "model_d320      320      14          4e-4    0.10"
  "model_d384      384      16          4e-4    0.10"
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
    --out-dir runs/exp9_scale_${LABEL}_noposenc

python analyze_results.py --run-dir runs/exp9_scale_${LABEL}_noposenc

# Replot:
#   python scaling_plot.py \
#       --model-sweep-dirs runs/exp9_scale_model_d32_noposenc runs/exp9_scale_model_d64_noposenc \
#                           runs/exp9_scale_model_d96_noposenc runs/exp9_scale_model_d128_noposenc \
#                           runs/exp9_scale_model_d192_noposenc runs/exp9_scale_model_d256_noposenc \
#                           runs/exp9_scale_model_d320_noposenc runs/exp9_scale_model_d384_noposenc \
#       --data-sweep-dirs  runs/exp9_scale_model_d128_noposenc \
#       --out-dir runs/exp9_scaling_plots_noposenc \
#       --acts-ref-dir runs/exp9_scale_model_d128_noposenc
