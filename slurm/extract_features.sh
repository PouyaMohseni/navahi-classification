#!/bin/bash
#SBATCH --job-name=navahi-extract
#SBATCH --account=def-ichiro          # change to your allocation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err

set -e
mkdir -p logs

# ── Paths (adjust if your scratch layout differs) ────────────────────────────
SCRATCH=/lustre07/scratch/pmohseni
DATASET_DIR=$SCRATCH/datasets/Navahi/Navahi-Dataset
FEATURES_DIR=$SCRATCH/navahi-features
CHECKPOINTS_DIR=$SCRATCH/navahi-checkpoints
CODE_DIR=$SCRATCH/navahi-classification   # where you cloned the repo
# ─────────────────────────────────────────────────────────────────────────────

export NAVAHI_AUDIO_ROOT=$DATASET_DIR
export NAVAHI_FEATURES_DIR=$FEATURES_DIR
export NAVAHI_CHECKPOINTS_DIR=$CHECKPOINTS_DIR

module load python/3.11
module load cuda/12.2

# Create virtualenv on first run
VENV=$SCRATCH/navahi-venv
if [ ! -d "$VENV" ]; then
    python -m venv $VENV
fi
source $VENV/bin/activate
pip install -q -r $CODE_DIR/requirements.txt

cd $CODE_DIR
python src/extract_features.py --split all --device auto

echo "Feature extraction done. Features saved to: $FEATURES_DIR"
