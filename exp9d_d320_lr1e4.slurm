#!/bin/bash
#SBATCH --job-name=TF_exp9d_d320
#SBATCH --output=TF_exp9d_d320_%j.out
#SBATCH --error=TF_exp9d_d320_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env
cd /home/xucabm80/ParamRegression

# exp9d: retest d320/L14 at lr=1e-4 (matching d384, instead of exp9c's
# 1.5e-4) to check whether d320's weak result in the exp9c sweep was an
# LR artifact -- d384 with a smaller LR trained fine, while d320 with a
# larger LR underperformed, which is the opposite of what you'd expect if
# LR just needed to keep decreasing with depth. This isolates that one
# variable.

python -u train_trackformer.py \
    --channel ttbar --pileup pu200 \
    --max-events 950 --train-events 800 --test-events 100 \
    --features r dphi z \
    --d-model 320 --nhead 4 --num-layers 14 \
    --pos-encoding none --head-mode shared \
    --norm-loss none --weight-decay 0.01 \
    --seed 0 --batch-size 128 --epochs 200 --lr 1e-4 \
    --warmup-frac 0.15 \
    --early-stop-patience 30 \
    --eval-acts \
    --out-dir runs/exp9d_scale_model_d320_L14_lr1e4_noposenc

python analyze_results.py --run-dir runs/exp9d_scale_model_d320_L14_lr1e4_noposenc
