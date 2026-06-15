#!/usr/bin/env bash
# Navahi pipeline: extract → train → evaluate
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Step 1: Extract MERT features ==="
python src/extract_features.py --split all

echo ""
echo "=== Step 2: Train classifier ==="
python src/train.py --epochs 10 --batch_size 32 --lr 2e-5

echo ""
echo "=== Step 3: Evaluate on test set ==="
python src/evaluate.py --split test
