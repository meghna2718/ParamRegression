#!/bin/bash
#SBATCH --job-name=TF_test
#SBATCH --output=TF_test_%j.out
#SBATCH --error=TF_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:1

module load Python/3.9.6-GCCcore-11.2.0
source /home/xucabm80/ParamRegression/trackformer_env/bin/activate
cd /home/xucabm80/ParamRegression   # or wherever you put train_trackformer.py
pip install -q -r requirements.txt   # one-time; harmless if already installed

# Small/fast smoke test: confirms data download, feature engineering, model,
# training loop and checkpointing all work end to end before you commit to a
# real run or a sweep.
python train_trackformer.py \
    --channel ttbar --pileup pu200 \
    --max-events 60 --train-events 50 --test-events 10 \
    --features r dphi z \
    --d-model 32 --nhead 4 --num-layers 2 \
    --pos-encoding none --head-mode shared \
    --batch-size 64 --epochs 10 \
    --eval-acts \
    --out-dir runs/test_run

# After this finishes, generate the loss-curve / feature-importance / ACTS
# comparison plots:
python analyze_results.py --run-dir runs/test_run
