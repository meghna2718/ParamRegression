#!/bin/bash
#SBATCH --job-name=TF_exp7_recover
#SBATCH --output=TF_exp7_recover_%j.out
#SBATCH --error=TF_exp7_recover_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env
cd /home/xucabm80/ParamRegression

ANCHOR=runs/exp3_fix_v3_stability_fix_seed0

echo "=== $(date) : regenerating anchor's comparison_light.parquet (no retrain) ==="
python evaluate_checkpoint.py --run-dir ${ANCHOR}

echo "=== $(date) : calibration test on anchor ==="
python calibration_test.py --run-dir ${ANCHOR} --n-mc 30

echo "=== $(date) : attention maps on anchor ==="
python visualize_attention.py --run-dir ${ANCHOR} --n-tracks 4

echo "=== $(date) : rerunning scaling_plot.py with the CORRECT (v2, fixed) dirs ==="
python scaling_plot.py \
    --model-sweep-dirs runs/exp4_scale_model_small_v1 runs/exp4_scale_model_medium_v1 \
                        runs/exp4_scale_model_gap_v1 \
                        ${ANCHOR} runs/exp4_scale_model_large_v2 \
    --data-sweep-dirs  runs/exp4_scale_data_small_v1 runs/exp4_scale_data_medium_v1 \
                        ${ANCHOR} runs/exp5_data_1500_v2 \
    --out-dir runs/exp4_scaling_plots

echo "=== $(date) : done ==="
