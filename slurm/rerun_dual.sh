#!/bin/bash
# ============================================================
# rerun_dual.sh — re-runs only the dual pipeline
#
#   extract_dual  →  train_dual  →  evaluate (both streams)
#
# Single-stream features and checkpoint are left intact.
# Run from the code dir on the Narval login node:
#
#   cd /lustre07/scratch/pmohseni/navahi-classification
#   bash slurm/rerun_dual.sh
# ============================================================

set -e

SCRATCH=/lustre07/scratch/pmohseni
CODE_DIR=$SCRATCH/navahi-classification
ACCOUNT=${SLURM_ACCOUNT:-def-ichiro}

mkdir -p $CODE_DIR/logs

NAVAHI_ROOT=$SCRATCH/datasets/Navahi
HF_CACHE=$SCRATCH/hf-cache
FEATURES_DIR=$SCRATCH/navahi-features
FEATURES_DUAL_DIR=$SCRATCH/navahi-features-dual
CHECKPOINTS_DIR=$SCRATCH/navahi-checkpoints
CHECKPOINTS_DUAL_DIR=$SCRATCH/navahi-checkpoints-dual

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
export NAVAHI_FEATURES_DUAL_DIR=$FEATURES_DUAL_DIR
export NAVAHI_CHECKPOINTS_DIR=$CHECKPOINTS_DIR
export NAVAHI_CHECKPOINTS_DUAL_DIR=$CHECKPOINTS_DUAL_DIR
cd $CODE_DIR
"

echo "======================================================"
echo " Navahi — re-running dual pipeline"
echo " Account : $ACCOUNT"
echo "======================================================"

# ── 1. Dual extraction ────────────────────────────────────
JOB1=$(sbatch --parsable \
  --job-name=navahi-dual-extract \
  --account=$ACCOUNT \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=48G \
  --time=10:00:00 \
  --output=$CODE_DIR/logs/dual_extract_%j.out \
  --error=$CODE_DIR/logs/dual_extract_%j.err \
  --wrap="$ENV_BLOCK
python src/extract_features_dual.py --split all
echo 'Dual extraction done.'")

echo "Submitted dual_extract  -> job $JOB1"

# ── 2. Train dual (waits for extraction) ─────────────────
JOB2=$(sbatch --parsable \
  --job-name=navahi-dual-train \
  --account=$ACCOUNT \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --time=3:00:00 \
  --dependency=afterok:$JOB1 \
  --output=$CODE_DIR/logs/dual_train_%j.out \
  --error=$CODE_DIR/logs/dual_train_%j.err \
  --wrap="$ENV_BLOCK
python src/train_dual.py --epochs 10 --batch_size 32 --lr 2e-5
echo 'Dual training done.'")

echo "Submitted dual_train    -> job $JOB2  (after $JOB1)"

# ── 3. Evaluate both streams (waits for dual training) ───
JOB3=$(sbatch --parsable \
  --job-name=navahi-eval \
  --account=$ACCOUNT \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=8G \
  --time=0:30:00 \
  --dependency=afterok:$JOB2 \
  --output=$CODE_DIR/logs/eval_%j.out \
  --error=$CODE_DIR/logs/eval_%j.err \
  --wrap="$ENV_BLOCK
echo '=== Single-stream ==='
python src/evaluate.py --checkpoint $CHECKPOINTS_DIR/best_model.pt --split test

echo ''
echo '=== Dual-stream ==='
python src/evaluate.py --checkpoint $CHECKPOINTS_DUAL_DIR/best_model.pt --split test --dual
echo 'Evaluation done.'")

echo "Submitted evaluate      -> job $JOB3  (after $JOB2)"

echo ""
echo "======================================================"
echo " Job chain:"
echo "   $JOB1 (dual_extract)"
echo "   $JOB2 (dual_train)   (after $JOB1)"
echo "   $JOB3 (evaluate)     (after $JOB2)"
echo ""
echo " Monitor with:"
echo "   squeue -u \$USER"
echo "   tail -f $CODE_DIR/logs/dual_extract_${JOB1}.out"
echo "======================================================"
