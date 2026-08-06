#!/bin/bash
#SBATCH --job-name=TF_exp1_posenc
#SBATCH --output=TF_exp1_posenc_%A_%a.out
#SBATCH --error=TF_exp1_posenc_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-5            # 3 encodings x 2 seeds

module load Python/3.9.6-GCCcore-11.2.0
source /home/xucabm80/ParamRegression/trackformer_env/bin/activate

cd /home/xucabm80/ParamRegression
pip install -q -r requirements.txt   # one-time; harmless if already installed

# EXPERIMENT 1: Does positional encoding matter, and which kind?
# Base model = Jeremy's architecture (shared regression head), everything else
# held fixed. --pos-encoding varies across {none, sinusoidal, learnable}, and
# each is trained with 2 seeds -- a single-seed comparison can't distinguish
# "this encoding is better" from "this training run got lucky". Extend
# SEEDS below (and --array accordingly) for more replicates if time allows.
ENCODINGS=(none sinusoidal learnable)
SEEDS=(0 1)
N_SEEDS=${#SEEDS[@]}
ENC=${ENCODINGS[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}

python train_trackformer.py \
    --channel ttbar --pileup pu200 \
    --max-events 1700 --train-events 1500 --test-events 150 \
    --features r dphi z \
    --d-model 128 --nhead 4 --num-layers 4 \
    --head-mode shared \
    --pos-encoding ${ENC} \
    --seed ${SEED} \
    --batch-size 128 --epochs 400 --lr 1e-3 \
    --eval-acts \
    --out-dir runs/exp1_posenc_${ENC}_seed${SEED}

python analyze_results.py --run-dir runs/exp1_posenc_${ENC}_seed${SEED}

# After all 6 array tasks finish, aggregate with (repeated labels are
# automatically treated as seed replicates -> mean +/- std per encoding):
#   python compare_runs.py \
#       --run-dirs runs/exp1_posenc_none_seed0       runs/exp1_posenc_none_seed1 \
#                  runs/exp1_posenc_sinusoidal_seed0 runs/exp1_posenc_sinusoidal_seed1 \
#                  runs/exp1_posenc_learnable_seed0  runs/exp1_posenc_learnable_seed1 \
#       --labels   none none sinusoidal sinusoidal learnable learnable \
#       --out-dir runs/exp1_posenc_comparison
