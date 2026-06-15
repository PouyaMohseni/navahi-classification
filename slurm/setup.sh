#!/bin/bash
# ============================================================
# setup.sh — run ONCE on the login node (no GPU needed)
#
# Does:
#   1. Creates a Python virtualenv in scratch
#   2. Installs all dependencies
#   3. Pre-downloads all HuggingFace models to scratch cache
#      so compute nodes can run fully offline
#
# Usage:
#   cd /lustre07/scratch/pmohseni/navahi-classification
#   bash slurm/setup.sh
# ============================================================

set -e

SCRATCH=/lustre07/scratch/pmohseni
CODE_DIR=$SCRATCH/navahi-classification
VENV=$SCRATCH/navahi-venv
HF_CACHE=$SCRATCH/hf-cache    # all model weights land here

echo "======================================================"
echo " Navahi setup — login node"
echo " SCRATCH : $SCRATCH"
echo " VENV    : $VENV"
echo " HF cache: $HF_CACHE"
echo "======================================================"

# ── 1. Python + venv ──────────────────────────────────────
module load python/3.11
module load cuda/12.2

if [ ! -d "$VENV" ]; then
    echo "[1/3] Creating virtualenv..."
    python -m venv $VENV
else
    echo "[1/3] Virtualenv already exists — skipping"
fi

source $VENV/bin/activate
pip install -q --upgrade pip
pip install -q -r $CODE_DIR/requirements.txt
pip install -q demucs
echo "      Dependencies installed."

# ── 2. Pre-download HuggingFace models ────────────────────
export HF_HOME=$HF_CACHE
mkdir -p $HF_CACHE

echo "[2/3] Downloading models to $HF_CACHE ..."
python - <<'PYEOF'
import os
os.environ["HF_HOME"] = os.environ["HF_HOME"]   # already set above

from transformers import AutoModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model

print("  -> MERT-v1-95M ...")
AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

print("  -> wav2vec2-large-xlsr-53 ...")
Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

print("  -> Demucs htdemucs weights ...")
from demucs.pretrained import get_model
get_model("htdemucs")

print("  All models cached.")
PYEOF

# ── 3. Sanity-check dataset layout ────────────────────────
echo "[3/3] Checking dataset layout..."
NAVAHI=$SCRATCH/datasets/Navahi

MISSING=0
for subdir in "Navahi-Dataset/train" "Navahi-Dataset/test" "Split9" "Data"; do
    if [ ! -d "$NAVAHI/$subdir" ]; then
        echo "  MISSING: $NAVAHI/$subdir"
        MISSING=1
    else
        COUNT=$(find "$NAVAHI/$subdir" -name "*.mp3" 2>/dev/null | wc -l)
        echo "  OK: $NAVAHI/$subdir  ($COUNT mp3s)"
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "  Dataset incomplete. Unzip Navahi.zip first:"
    echo "    cd $SCRATCH/datasets && unzip Navahi.zip"
    exit 1
fi

echo ""
echo "======================================================"
echo " Setup complete. Submit jobs with:"
echo "   bash slurm/submit_all.sh"
echo "======================================================"
