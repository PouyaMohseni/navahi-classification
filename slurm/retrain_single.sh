#!/bin/bash
# ============================================================
# retrain_single.sh — retrains single-stream with fixed hyperparams
#
#   train_single  →  evaluate (single-stream only)
#
# Features already exist; this only re-runs training + eval.
# Run from the code dir on the Narval login node:
#
#   cd /lustre07/scratch/pmohseni/navahi-classification
#   bash slurm/retrain_single.sh
# ============================================================

set -e

SCRATCH=/lustre07/scratch/pmohseni
CODE_DIR=$SCRATCH/navahi-classification
ACCOUNT=${SLURM_ACCOUNT:-def-ichiro}

mkdir -p $CODE_DIR/logs

NAVAHI_ROOT=$SCRATCH/datasets/Navahi
HF_CACHE=$SCRATCH/hf-cache
FEATURES_DIR=$SCRATCH/navahi-features
CHECKPOINTS_DIR=$SCRATCH/navahi-checkpoints

ENV_BLOCK="
module load python/3.11
module load cuda/12.2
source ~/navahi-venv/bin/activate
export HF_HOME=$HF_CACHE
export TORCH_HOME=$HF_CACHE/torch
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NAVAHI_ROOT=$NAVAHI_ROOT
export NAVAHI_FEATURES_DIR=$FEATURES_DIR
export NAVAHI_CHECKPOINTS_DIR=$CHECKPOINTS_DIR
cd $CODE_DIR
"

echo "======================================================"
echo " Navahi — retraining single-stream (lr=1e-3, 50 epochs, weighted loss)"
echo " Account : $ACCOUNT"
echo "======================================================"

# ── 1. Train single ───────────────────────────────────────
JOB1=$(sbatch --parsable \
  --job-name=navahi-train \
  --account=$ACCOUNT \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --time=3:00:00 \
  --output=$CODE_DIR/logs/train_%j.out \
  --error=$CODE_DIR/logs/train_%j.err \
  --wrap="$ENV_BLOCK
python src/train.py --epochs 50 --batch_size 32 --lr 1e-3
echo 'Training done.'")

echo "Submitted train_single  -> job $JOB1"

# ── 2. Evaluate (waits for training) ─────────────────────
JOB2=$(sbatch --parsable \
  --job-name=navahi-eval \
  --account=$ACCOUNT \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=8G \
  --time=0:30:00 \
  --dependency=afterok:$JOB1 \
  --output=$CODE_DIR/logs/eval_%j.out \
  --error=$CODE_DIR/logs/eval_%j.err \
  --wrap="$ENV_BLOCK
echo '=== Single-stream ==='
python src/evaluate.py --checkpoint $CHECKPOINTS_DIR/best_model.pt --split test
echo 'Evaluation done.'")

echo "Submitted evaluate      -> job $JOB2  (after $JOB1)"

echo ""
echo "======================================================"
echo " Monitor with:"
echo "   squeue -u \$USER"
echo "   tail -f $CODE_DIR/logs/train_${JOB1}.out"
echo "======================================================"
