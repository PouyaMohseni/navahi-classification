#!/bin/bash
#SBATCH --job-name=navahi-dual-train
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/dual_train_%j.out
#SBATCH --error=logs/dual_train_%j.err

set -e
mkdir -p logs

SCRATCH=/lustre07/scratch/pmohseni
export NAVAHI_ROOT=$SCRATCH/datasets/Navahi
export NAVAHI_FEATURES_DUAL_DIR=$SCRATCH/navahi-features-dual
export NAVAHI_CHECKPOINTS_DUAL_DIR=$SCRATCH/navahi-checkpoints-dual

module load python/3.11
module load cuda/12.2
source ~/navahi-venv/bin/activate

cd $SCRATCH/navahi-classification
python src/train_dual.py --epochs 10 --batch_size 32 --lr 2e-5
