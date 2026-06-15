#!/bin/bash
#SBATCH --job-name=navahi-dual-extract
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/dual_extract_%j.out
#SBATCH --error=logs/dual_extract_%j.err

set -e
mkdir -p logs

SCRATCH=/lustre07/scratch/pmohseni
NAVAHI_ROOT=$SCRATCH/datasets/Navahi
FEATURES_DUAL_DIR=$SCRATCH/navahi-features-dual
CHECKPOINTS_DUAL_DIR=$SCRATCH/navahi-checkpoints-dual
CODE_DIR=$SCRATCH/navahi-classification

export NAVAHI_ROOT=$NAVAHI_ROOT
export NAVAHI_FEATURES_DUAL_DIR=$FEATURES_DUAL_DIR
export NAVAHI_CHECKPOINTS_DUAL_DIR=$CHECKPOINTS_DUAL_DIR

module load python/3.11
module load cuda/12.2

source $SCRATCH/navahi-venv/bin/activate
pip install -q demucs

cd $CODE_DIR
python src/extract_features_dual.py --split all

echo "Dual extraction done. Features: $FEATURES_DUAL_DIR"
