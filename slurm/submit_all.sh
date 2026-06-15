#!/bin/bash
# ============================================================
# submit_all.sh — submits the full pipeline as chained SLURM jobs
#
# Job graph:
#
#   [extract_single] ──────────────────┐
#                                      ├──> [train_single] ──┐
#   [extract_dual]  ──> [train_dual] ──┘                     ├──> [evaluate]
#                                                             │
#                   (extract_single and extract_dual          │
#                    run in parallel)                         │
#                                                            done
#
# Usage:
#   cd /lustre07/scratch/pmohseni/navahi-classification
#   bash slurm/submit_all.sh
# ============================================================

set -e

SCRATCH=/lustre07/scratch/pmohseni
CODE_DIR=$SCRATCH/navahi-classification
ACCOUNT=${SLURM_ACCOUNT:-def-ichiro}    # override: SLURM_ACCOUNT=def-foo bash submit_all.sh

mkdir -p $CODE_DIR/logs

echo "======================================================"
echo " Navahi — submitting pipeline"
echo " Account : $ACCOUNT"
echo " Code dir: $CODE_DIR"
echo "======================================================"

# ── Shared environment ─────────────────────────────────────
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
mkdir -p logs
"

# ── 1. Extract single-stream features ─────────────────────
JOB1=$(sbatch --parsable \
  --job-name=navahi-extract \
  --account=$ACCOUNT \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=6:00:00 \
  --output=$CODE_DIR/logs/extract_%j.out \
  --error=$CODE_DIR/logs/extract_%j.err \
  --wrap="$ENV_BLOCK
python src/extract_features.py --split all
echo 'Single-stream extraction done.'")

echo "Submitted extract_single   -> job $JOB1"

# ── 2. Extract dual-stream features (runs in parallel) ────
JOB2=$(sbatch --parsable \
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
echo 'Dual-stream extraction done.'")

echo "Submitted extract_dual     -> job $JOB2  (parallel with $JOB1)"

# ── 3. Train single-stream (waits for job 1) ──────────────
JOB3=$(sbatch --parsable \
  --job-name=navahi-train \
  --account=$ACCOUNT \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --time=1:30:00 \
  --dependency=afterok:$JOB1 \
  --output=$CODE_DIR/logs/train_%j.out \
  --error=$CODE_DIR/logs/train_%j.err \
  --wrap="$ENV_BLOCK
python src/train.py --epochs 10 --batch_size 32 --lr 2e-5
echo 'Single-stream training done.'")

echo "Submitted train_single     -> job $JOB3  (after $JOB1)"

# ── 4. Train dual-stream (waits for job 2) ────────────────
JOB4=$(sbatch --parsable \
  --job-name=navahi-dual-train \
  --account=$ACCOUNT \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --time=1:30:00 \
  --dependency=afterok:$JOB2 \
  --output=$CODE_DIR/logs/dual_train_%j.out \
  --error=$CODE_DIR/logs/dual_train_%j.err \
  --wrap="$ENV_BLOCK
python src/train_dual.py --epochs 10 --batch_size 32 --lr 2e-5
echo 'Dual-stream training done.'")

echo "Submitted train_dual       -> job $JOB4  (after $JOB2)"

# ── 5. Evaluate both (waits for jobs 3 and 4) ─────────────
JOB5=$(sbatch --parsable \
  --job-name=navahi-eval \
  --account=$ACCOUNT \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=8G \
  --time=0:30:00 \
  --dependency=afterok:$JOB3:$JOB4 \
  --output=$CODE_DIR/logs/eval_%j.out \
  --error=$CODE_DIR/logs/eval_%j.err \
  --wrap="$ENV_BLOCK
echo '=== Single-stream ==='
python src/evaluate.py --checkpoint $CHECKPOINTS_DIR/best_model.pt --split test

echo ''
echo '=== Dual-stream ==='
python src/evaluate.py --checkpoint $CHECKPOINTS_DUAL_DIR/best_model.pt --split test --dual
echo 'Evaluation done.'")

echo "Submitted evaluate         -> job $JOB5  (after $JOB3 and $JOB4)"

echo ""
echo "======================================================"
echo " Pipeline submitted. Monitor with:"
echo "   squeue -u \$USER"
echo "   tail -f $CODE_DIR/logs/extract_${JOB1}.out"
echo ""
echo " Job chain:"
echo "   $JOB1 (extract_single)"
echo "   $JOB2 (extract_dual)   --> $JOB4 (train_dual)"
echo "   $JOB1                  --> $JOB3 (train_single)"
echo "   $JOB3 + $JOB4          --> $JOB5 (evaluate)"
echo "======================================================"
