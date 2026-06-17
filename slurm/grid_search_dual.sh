#!/bin/bash
#
# Grid search over dual-stream configs: 3 lr × 3 lambda × 3 layer_sets = 27 runs.
# Each array task trains one config and evaluates on test + test_simplified.
#
# Submit from the LOGIN NODE:
#   cd $SCRATCH/navahi-classification
#   sbatch slurm/grid_search_dual.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/gs_<arrayID>_<taskID>.out
#
# Compare results after all jobs finish:
#   python src/grid_compare.py --results_dir $SCRATCH/navahi-gs-dual
#
#SBATCH --job-name=navahi-gs
#SBATCH --account=def-ichiro
#SBATCH --array=0-26
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/gs_%A_%a.out
#SBATCH --error=logs/gs_%A_%a.err

set -e
mkdir -p logs

SCRATCH=/lustre07/scratch/pmohseni
GS_DIR=$SCRATCH/navahi-gs-dual

export NAVAHI_ROOT=$SCRATCH/datasets/Navahi
export NAVAHI_FEATURES_DUAL_DIR=$SCRATCH/navahi-features-dual
export NAVAHI_CHECKPOINTS_DUAL_DIR=$SCRATCH/navahi-checkpoints-dual

export HF_HOME=$SCRATCH/hf-cache
export TORCH_HOME=$SCRATCH/hf-cache/torch
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

module load python/3.11
module load cuda/12.2
source ~/navahi-venv/bin/activate

cd $SCRATCH/navahi-classification
mkdir -p "$GS_DIR"

echo "Starting run $SLURM_ARRAY_TASK_ID / 26"
python src/grid_train_eval.py \
    --run_idx    "$SLURM_ARRAY_TASK_ID" \
    --output_dir "$GS_DIR"

echo "Run $SLURM_ARRAY_TASK_ID complete."
