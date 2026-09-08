#!/bin/bash
#SBATCH --job-name=TF_exp2_shared_s3
#SBATCH --output=TF_exp2_shared_s3_%j.out
#SBATCH --error=TF_exp2_shared_s3_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env
cd /home/xucabm80/ParamRegression

# Replacement seed for exp2 (head-mode comparison, Sec. exp1_headmode):
# shared_s0 collapsed and was excluded by usable_seeds()'s factor-of-2
# outlier check, leaving shared with only 2 usable points against
# separate's 3. Seed 3 (not seed 0) fills that third slot instead of
# retrying the seed that already collapsed. Same v3_150ep_fixed recipe as
# exp2_headmode.slurm -- only --seed and --out-dir differ.
RUN_TAG="v3_150ep_fixed"
HEAD_MODE="shared"
SEED=3

python -u train_trackformer.py \
    --channel ttbar --pileup pu200 \
    --max-events 950 --train-events 800 --test-events 100 \
    --features r dphi z \
    --d-model 128 --nhead 4 --num-layers 8 \
    --pos-encoding sinusoidal \
    --head-mode ${HEAD_MODE} \
    --seed ${SEED} \
    --batch-size 128 --epochs 150 --lr 1e-3 \
    --early-stop-patience 20 \
    --eval-acts \
    --out-dir runs/exp2_head_${HEAD_MODE}_${RUN_TAG}_seed${SEED}

python analyze_results.py --run-dir runs/exp2_head_${HEAD_MODE}_${RUN_TAG}_seed${SEED}
