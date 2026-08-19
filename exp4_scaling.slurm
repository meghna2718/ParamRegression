#!/bin/bash
#SBATCH --job-name=TF_exp4_scale
#SBATCH --output=TF_exp4_scale_%A_%a.out
#SBATCH --error=TF_exp4_scale_%A_%a.err
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-5            # 3 new model-size points + 3 new data-size points

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env

cd /home/xucabm80/ParamRegression
pip install -q -r requirements.txt   # one-time; harmless if already installed

# EXPERIMENT 4: scaling-law sweep. Two axes, each swept independently at 4
# points (3 new here + the exp3 run as the shared anchor at 128-dim/8-layer,
# 800 events -- do NOT rerun that one). Everything else held at exp3's recipe
# (norm_loss none, weight_decay 0.01, early stopping) so only model size or
# data size differs between points on the same axis.
# Metric: best_val_loss (run_summary.json) + per-param resolution
# (test_resolutions.json), both saved by every run automatically.
RUN_TAG="v1"

# task: label            d_model  num_layers  train_events  test_events  max_events
TASKS=(
  "model_small           32       2           800           100          950"
  "model_medium           64       4           800           100          950"
  "model_large           256      12          800           100          950"
  "data_small            128      8           200           100          350"
  "data_medium           128      8           400           100          550"
  "data_large            128      8           1500          100          1650"
)

read -r LABEL D_MODEL NUM_LAYERS TRAIN_EVENTS TEST_EVENTS MAX_EVENTS <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "Task $SLURM_ARRAY_TASK_ID: $LABEL  d_model=$D_MODEL  num_layers=$NUM_LAYERS  train_events=$TRAIN_EVENTS  max_events=$MAX_EVENTS"

python -u train_trackformer.py \
    --channel ttbar --pileup pu200 \
    --max-events ${MAX_EVENTS} --train-events ${TRAIN_EVENTS} --test-events ${TEST_EVENTS} \
    --features r dphi z \
    --d-model ${D_MODEL} --nhead 4 --num-layers ${NUM_LAYERS} \
    --pos-encoding sinusoidal \
    --head-mode shared \
    --norm-loss none \
    --weight-decay 0.01 \
    --seed 0 \
    --batch-size 128 --epochs 300 --lr 1e-3 \
    --early-stop-patience 20 \
    --eval-acts \
    --out-dir runs/exp4_scale_${LABEL}_${RUN_TAG}

python analyze_results.py --run-dir runs/exp4_scale_${LABEL}_${RUN_TAG}

# After all 6 tasks finish, combine with the exp3 anchor run and plot:
#   python scaling_plot.py \
#       --model-sweep-dirs runs/exp4_scale_model_small_v1 runs/exp4_scale_model_medium_v1 runs/exp3_fix_v3_stability_fix_seed0 runs/exp4_scale_model_large_v1 \
#       --data-sweep-dirs  runs/exp4_scale_data_small_v1  runs/exp4_scale_data_medium_v1  runs/exp3_fix_v3_stability_fix_seed0 runs/exp4_scale_data_large_v1 \
#       --out-dir runs/exp4_scaling_plots
