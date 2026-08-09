#!/bin/bash
#SBATCH --job-name=TF_exp4_full_baseline
#SBATCH --output=TF_exp4_full_baseline_%j.out
#SBATCH --error=TF_exp4_full_baseline_%j.err
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1

# EXPERIMENT 4: full baseline run with the phi fix + z-symmetry fix + deeper model (8 layers).

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env

cd /home/xucabm80/ParamRegression
pip install -q -r requirements.txt   # one-time; harmless if already installed

python -u train_trackformer.py \
    --channel ttbar --pileup pu200 \
    --max-events 950 --train-events 800 --test-events 100 \
    --features r dphi z \
    --d-model 128 --nhead 4 --num-layers 8 \
    --pos-encoding none \
    --head-mode shared \
    --seed 0 \
    --batch-size 128 --epochs 400 --lr 1e-3 \
    --eval-acts \
    --out-dir runs/exp4_full_baseline

python analyze_results.py --run-dir runs/exp4_full_baseline
