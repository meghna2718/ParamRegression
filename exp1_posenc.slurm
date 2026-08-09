#!/bin/bash
#SBATCH --job-name=TF_exp1_posenc
#SBATCH --output=TF_exp1_posenc_%A_%a.out
#SBATCH --error=TF_exp1_posenc_%A_%a.err
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-2            # 3 encodings x 1 seed

eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate paramreg_env

cd /home/xucabm80/ParamRegression
pip install -q -r requirements.txt   # one-time; harmless if already installed

# EXPERIMENT 1: positional encoding comparison. Single seed (provisional --
# rerun with a 2nd seed before trusting small differences). "none" doubles
# as the full baseline run (phi fix + z-symmetry fix + 8 layers).
ENCODINGS=(none sinusoidal learnable)
ENC=${ENCODINGS[$SLURM_ARRAY_TASK_ID]}
SEED=0

# --particle-types 13 to restrict to muons only; empty = all primary species.
PARTICLE_TYPE_FLAG=""

python -u train_trackformer.py \
    --channel ttbar --pileup pu200 \
    --max-events 950 --train-events 800 --test-events 100 \
    --features r dphi z \
    --d-model 128 --nhead 4 --num-layers 8 \
    --head-mode shared \
    --pos-encoding ${ENC} \
    --seed ${SEED} \
    --batch-size 128 --epochs 400 --lr 1e-3 \
    --eval-acts \
    ${PARTICLE_TYPE_FLAG} \
    --out-dir runs/exp1_posenc_${ENC}_seed${SEED}

python analyze_results.py --run-dir runs/exp1_posenc_${ENC}_seed${SEED}

# After all 3 array tasks finish, compare (run manually, doesn't need a GPU):
#   python compare_runs.py \
#       --run-dirs runs/exp1_posenc_none_seed0 runs/exp1_posenc_sinusoidal_seed0 runs/exp1_posenc_learnable_seed0 \
#       --labels   none                        sinusoidal                       learnable \
#       --out-dir runs/exp1_posenc_comparison
