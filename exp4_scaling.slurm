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
#SBATCH --array=0-5            # tasks 0-5 done; task 6 (model_gap) is new -- submit with --array=6

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env

cd /home/xucabm80/ParamRegression
pip install -q -r requirements.txt

# Scaling-law sweep: model-size and data-size axes, anchor = exp3 run.
RUN_TAG="v1"

# label                  d_model  num_layers  train_events  test_events  max_events
TASKS=(
  "model_small           32       2           800           100          950"
  "model_medium           64       4           800           100          950"
  "model_large           256      12          800           100          950"
  "data_small            128      8           200           100          350"
  "data_medium           128      8           400           100          550"
  "data_large            128      8           1500          100          1650"
  "model_gap             96       6           800           100          950"
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

# Full sweep plot (model_large -> _v2 retry, data_large -> exp5's retry):
#   python scaling_plot.py \
#       --model-sweep-dirs runs/exp4_scale_model_small_v1 runs/exp4_scale_model_medium_v1 \
#                           runs/exp4_scale_model_gap_v1 \
#                           runs/exp3_fix_v3_stability_fix_seed0 runs/exp4_scale_model_large_v2 \
#       --data-sweep-dirs  runs/exp4_scale_data_small_v1 runs/exp4_scale_data_medium_v1 \
#                           runs/exp3_fix_v3_stability_fix_seed0 runs/exp5_data_1500_v2 \
#                           runs/exp5_data_3000 runs/exp5_data_6000 \
#       --out-dir runs/exp4_scaling_plots
