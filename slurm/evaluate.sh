#!/bin/bash
#SBATCH --job-name=navahi-eval
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -e
mkdir -p logs

SCRATCH=/lustre07/scratch/pmohseni
FEATURES_DIR=$SCRATCH/navahi-features
CHECKPOINTS_DIR=$SCRATCH/navahi-checkpoints
CODE_DIR=$SCRATCH/navahi-classification

export NAVAHI_FEATURES_DIR=$FEATURES_DIR
export NAVAHI_CHECKPOINTS_DIR=$CHECKPOINTS_DIR

module load python/3.11
module load cuda/12.2

source ~/navahi-venv/bin/activate

cd $CODE_DIR
python src/evaluate.py \
    --checkpoint $CHECKPOINTS_DIR/best_model.pt \
    --split test
