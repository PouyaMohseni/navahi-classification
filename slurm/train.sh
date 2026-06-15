#!/bin/bash
#SBATCH --job-name=navahi-train
#SBATCH --account=def-ichiro          # change to your allocation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -e
mkdir -p logs

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH=/lustre07/scratch/pmohseni
FEATURES_DIR=$SCRATCH/navahi-features
CHECKPOINTS_DIR=$SCRATCH/navahi-checkpoints
CODE_DIR=$SCRATCH/navahi-classification
# ─────────────────────────────────────────────────────────────────────────────

export NAVAHI_FEATURES_DIR=$FEATURES_DIR
export NAVAHI_CHECKPOINTS_DIR=$CHECKPOINTS_DIR

module load python/3.11
module load cuda/12.2

source $SCRATCH/navahi-venv/bin/activate

cd $CODE_DIR
python src/train.py \
    --epochs 10 \
    --batch_size 32 \
    --lr 2e-5 \
    --output $CHECKPOINTS_DIR

echo "Training done. Best checkpoint in: $CHECKPOINTS_DIR"
