#!/bin/bash
#
# Train 9 vocal-only (wav2vec2-xlsr-53) classifier configs.
# Grid: 3 lr × 3 layer_sets (low/mid/high) = 9 array tasks.
#
# Submit from the LOGIN NODE:
#   cd $SCRATCH/navahi-classification
#   sbatch slurm/train_vocal_grid.sh
#
# After all jobs finish, run late fusion evaluation:
#   python src/late_fusion_eval.py \
#       --mert_ckpt $SCRATCH/navahi-checkpoints-v5/best_model.pt \
#       --vocal_dir $SCRATCH/navahi-vocal-gs \
#       --split test_simplified
#
#SBATCH --job-name=navahi-vocal
#SBATCH --account=def-ichiro
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH --output=logs/vocal_%A_%a.out
#SBATCH --error=logs/vocal_%A_%a.err

set -e
mkdir -p logs

SCRATCH=/lustre07/scratch/pmohseni
VOCAL_GS_DIR=$SCRATCH/navahi-vocal-gs

export NAVAHI_ROOT=$SCRATCH/datasets/Navahi
export NAVAHI_FEATURES_DIR=$SCRATCH/navahi-features
export NAVAHI_FEATURES_DUAL_DIR=$SCRATCH/navahi-features-dual

export HF_HOME=$SCRATCH/hf-cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

module load python/3.11
module load cuda/12.2
source ~/navahi-venv/bin/activate

cd $SCRATCH/navahi-classification
mkdir -p "$VOCAL_GS_DIR"

echo "Starting vocal run $SLURM_ARRAY_TASK_ID / 8"
python src/train_vocal_grid.py \
    --run_idx    "$SLURM_ARRAY_TASK_ID" \
    --output_dir "$VOCAL_GS_DIR"

echo "Vocal run $SLURM_ARRAY_TASK_ID complete."
