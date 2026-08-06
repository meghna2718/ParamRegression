#!/bin/bash
#SBATCH --job-name=TF_exp3_datascale
#SBATCH --output=TF_exp3_datascale_%A_%a.out
#SBATCH --error=TF_exp3_datascale_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-5            # 6 dataset sizes, 1 seed each (add seeds if time allows)

module load Python/3.9.6-GCCcore-11.2.0
source /home/xucabm80/ParamRegression/trackformer_env/bin/activate

cd /home/xucabm80/ParamRegression
pip install -q -r requirements.txt   # one-time; harmless if already installed

# EXPERIMENT 3: How much does resolution improve with more training data?
# Justifies your chosen --train-events empirically instead of by assertion,
# and tells you whether it's worth spending more cluster time on a bigger
# dataset for the final numbers, or whether resolution has already plateaued.
# Base model = same as exp1_posenc_none (shared head, pos-encoding none).
# Range extended up to 5000 events (still ~10-20x smaller than Jeremy's
# actual muon training set, which used 50,000-event splits) -- this is
# the tool for deciding empirically whether that gap actually matters for
# your resolution, rather than guessing.
TRAIN_EVENTS=(100 250 500 1000 2500 5000)
TEST_EVENTS=150
N=${TRAIN_EVENTS[$SLURM_ARRAY_TASK_ID]}
MAX_EVENTS=$(( N + TEST_EVENTS + 20 ))   # small buffer

python train_trackformer.py \
    --channel ttbar --pileup pu200 \
    --max-events ${MAX_EVENTS} --train-events ${N} --test-events ${TEST_EVENTS} \
    --features r dphi z \
    --d-model 128 --nhead 4 --num-layers 4 \
    --pos-encoding none --head-mode shared \
    --seed 0 \
    --batch-size 128 --epochs 400 --lr 1e-3 \
    --eval-acts \
    --out-dir runs/exp3_datascale_n${N}

python analyze_results.py --run-dir runs/exp3_datascale_n${N}

# After all 6 array tasks finish, aggregate with (label order controls plot
# x-axis order, so list them ascending):
#   python compare_runs.py \
#       --run-dirs runs/exp3_datascale_n100 runs/exp3_datascale_n250 runs/exp3_datascale_n500 \
#                  runs/exp3_datascale_n1000 runs/exp3_datascale_n2500 runs/exp3_datascale_n5000 \
#       --labels   100 250 500 1000 2500 5000 \
#       --out-dir runs/exp3_datascale_comparison
