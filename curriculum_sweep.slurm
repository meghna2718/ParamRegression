#!/bin/bash
#SBATCH --job-name=curriculum_sweep
#SBATCH --output=curriculum_sweep_%A_%a.out
#SBATCH --error=curriculum_sweep_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-2

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env
cd /home/xucabm80/ParamRegression

# One array task per seed, each writing to its OWN out-dir. run_sweep()
# does a read-modify-write on sweep_results.csv inside --out-dir; three
# tasks sharing one out-dir would race on that file and could silently
# drop results. Merge the three out-dirs back together once all tasks
# finish (see merge_curriculum_seeds.py).
SEEDS=(0 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python -u train_curriculum.py \
    --out-dir runs/curriculum_sweep_seed${SEED} \
    --seeds ${SEED} \
    --precision float32 \
    --make-plots
