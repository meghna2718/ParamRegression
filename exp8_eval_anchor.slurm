#!/bin/bash
#SBATCH --job-name=TF_exp8_eval
#SBATCH --output=TF_exp8_eval_%j.out
#SBATCH --error=TF_exp8_eval_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env
cd /home/xucabm80/ParamRegression

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export POLARS_MAX_THREADS=8

python evaluate_checkpoint.py --run-dir runs/exp3_fix_v3_stability_fix_seed0
