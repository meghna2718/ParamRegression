#!/bin/bash
#SBATCH --job-name=TF_exp1_fixed
#SBATCH --output=TF_exp1_fixed_%A_%a.out
#SBATCH --error=TF_exp1_fixed_%A_%a.err
#SBATCH --time=09:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-2

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env
cd /home/xucabm80/ParamRegression

# Rerun of exp1 (pos-encoding: none/sinusoidal/learnable) with the current
# stable recipe -- the original exp1 predates all these fixes and its
# loss-collapse plots aren't trustworthy. One seed each for speed.
# epochs=150, matching v3_150ep_fixed (NOT 300/anchor's recipe) -- the
# schedule's decay length is tied to --epochs, and 300 was shown to stop
# ~10x short of the loss v3_150ep_fixed reached at 150. --time bumped to 9h
# since v3_150ep_fixed's shared runs took 6h40m+ at this same config.

POS_ENCODINGS=(none sinusoidal learnable)
POS_ENC=${POS_ENCODINGS[$SLURM_ARRAY_TASK_ID]}

python -u train_trackformer.py \
    --channel ttbar --pileup pu200 \
    --max-events 950 --train-events 800 --test-events 100 \
    --features r dphi z \
    --d-model 128 --nhead 4 --num-layers 8 \
    --pos-encoding ${POS_ENC} \
    --head-mode shared \
    --norm-loss none --weight-decay 0.01 \
    --seed 0 --batch-size 128 --epochs 150 --lr 1e-3 \
    --early-stop-patience 20 \
    --eval-acts \
    --out-dir runs/exp1_fixed_posenc_${POS_ENC}_seed0

python analyze_results.py --run-dir runs/exp1_fixed_posenc_${POS_ENC}_seed0

# After all 3 finish:
#   python compare_runs.py --run-dirs runs/exp1_fixed_posenc_none_seed0 runs/exp1_fixed_posenc_sinusoidal_seed0 runs/exp1_fixed_posenc_learnable_seed0 --out-dir runs/exp1_fixed_comparison
