#!/bin/bash
#SBATCH --job-name=TF_exp1_fixed
#SBATCH --output=TF_exp1_fixed_%A_%a.out
#SBATCH --error=TF_exp1_fixed_%A_%a.err
#SBATCH --time=06:00:00
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
# stable recipe (norm-loss=none, weight-decay, auto warmup-frac,
# early-stop-patience) -- the original exp1 predates all of these and its
# loss-collapse plots are not trustworthy. One seed each for speed.
# epochs=300 (not 150) -- empirically converges faster for this architecture,
# see anchor's ~2h runtime vs v3_150ep_fixed's 6h+ runs at epochs=150.

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
    --seed 0 --batch-size 128 --epochs 300 --lr 1e-3 \
    --early-stop-patience 20 \
    --eval-acts \
    --out-dir runs/exp1_fixed_posenc_${POS_ENC}_seed0

python analyze_results.py --run-dir runs/exp1_fixed_posenc_${POS_ENC}_seed0

# After all 3 finish:
#   python compare_runs.py --run-dirs runs/exp1_fixed_posenc_none_seed0 runs/exp1_fixed_posenc_sinusoidal_seed0 runs/exp1_fixed_posenc_learnable_seed0 --out-dir runs/exp1_fixed_comparison
