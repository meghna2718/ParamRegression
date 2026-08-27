#!/bin/bash
#SBATCH --job-name=TF_exp10_norm
#SBATCH --output=TF_exp10_norm_%j.out
#SBATCH --error=TF_exp10_norm_%j.err
#SBATCH --time=09:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env
cd /home/xucabm80/ParamRegression

# Feature-normalization ablation. Toggle NORMALIZE below and resubmit to
# compare against the un-normalized baseline (none/learnable, qop=0.0047).
# POS_ENC: set to whichever won the exp1 rerun (none/sinusoidal/learnable).
NORMALIZE=true
POS_ENC=none   # <-- update once you've checked the sinusoidal rerun

NORM_FLAG=""
TAG="unnorm"
if [ "$NORMALIZE" = true ]; then
    NORM_FLAG="--normalize-features"
    TAG="norm"
fi

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
    ${NORM_FLAG} \
    --out-dir runs/exp10_${TAG}_posenc_${POS_ENC}_seed0

python analyze_results.py --run-dir runs/exp10_${TAG}_posenc_${POS_ENC}_seed0
