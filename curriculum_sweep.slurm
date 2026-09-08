#!/bin/bash
#SBATCH --job-name=curriculum_sweep
#SBATCH --output=curriculum_sweep_%j.out
#SBATCH --error=curriculum_sweep_%j.err
#SBATCH --time=09:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env
cd /home/xucabm80/ParamRegression   # adjust to wherever train_curriculum.py lives

python -u train_curriculum.py \
    --out-dir runs/curriculum_sweep_seeds012 \
    --seeds 0 1 2 \
    --precision float32 \
    --make-plots
