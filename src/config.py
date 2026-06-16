import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Paths (override with env vars on cluster) ──────────────────────────────────
NAVAHI_ROOT       = os.environ.get("NAVAHI_ROOT",
                                    os.path.join(PROJECT_ROOT, "Navahi_data", "Navahi"))
SPLIT9_DIR        = os.path.join(NAVAHI_ROOT, "Split9")
FEATURES_DIR      = os.environ.get("NAVAHI_FEATURES_DIR",
                                    os.path.join(PROJECT_ROOT, "features"))
FEATURES_DUAL_DIR = os.environ.get("NAVAHI_FEATURES_DUAL_DIR",
                                    os.path.join(PROJECT_ROOT, "features_dual"))
CHECKPOINTS_DIR   = os.environ.get("NAVAHI_CHECKPOINTS_DIR",
                                    os.path.join(PROJECT_ROOT, "checkpoints"))
CHECKPOINTS_DUAL_DIR = os.environ.get("NAVAHI_CHECKPOINTS_DUAL_DIR",
                                       os.path.join(PROJECT_ROOT, "checkpoints_dual"))

# ── Classes ────────────────────────────────────────────────────────────────────
CLASS_MAP = {
    "Gilan&Mazandaran":     0,
    "Lorestan":             1,
    "Khorasan":             2,
    "Kordestan&Kermanshah": 3,
    "Azerbaijan":           4,
    "Sistan&Baluchestan":   5,
    "Turkaman":             6,
    "Bushehr":              7,
}
CLASS_NAMES = [
    "Gilan, Talesh & Mazandaran",
    "Lorestan, Bakhtiyari & Fars",
    "Khorasan",
    "Kordestan & Kermanshah",
    "Azerbaijan",
    "Sistan & Baluchestan",
    "Golestan & Turkaman-Sahra",
    "Southern Coasts of Iran",
]
NUM_CLASSES = 8

CLASS_COORDS = {
    0: (37.10, 50.48),
    1: (32.04, 50.56),
    2: (36.36, 58.21),
    3: (35.11, 47.01),
    4: (37.83, 46.12),
    5: (27.53, 60.52),
    6: (37.23, 55.13),
    7: (28.65, 51.75),
}

LAT_MIN, LAT_MAX = 25.064, 39.780
LON_MIN, LON_MAX = 44.039, 63.333

# ── MERT ───────────────────────────────────────────────────────────────────────
MERT_MODEL        = "m-a-p/MERT-v1-95M"
MERT_SAMPLE_RATE  = 24000
SEGMENT_SEC       = 5          # each audio segment fed to MERT
NUM_HIDDEN_STATES = 13         # 1 embedding layer + 12 transformer layers
EMBED_DIM         = 768
MERT_LAYERS       = [6, 7, 8]  # selected at dataset time (not extraction time)

# Extracted .npy shape per file: (N_segs, NUM_HIDDEN_STATES, EMBED_DIM) = (N_segs, 13, 768)

# ── Window (segment stacking) ─────────────────────────────────────────────────
EVAL_WINDOW_SIZE = 12    # 12 × 5s = 60s default eval context

# Input to MLP = len(MERT_LAYERS) × EMBED_DIM × EVAL_WINDOW_SIZE
FEATURE_DIM = len(MERT_LAYERS) * EMBED_DIM * EVAL_WINDOW_SIZE  # 3*768*12 = 27648

# ── Dual-stream (vocal via wav2vec2-xlsr-53) ───────────────────────────────────
VOCAL_MODEL             = "facebook/wav2vec2-large-xlsr-53"
VOCAL_LAYERS            = [6, 7, 8]
VOCAL_EMBED_DIM         = 1024
VOCAL_NUM_HIDDEN_STATES = 25   # 1 CNN + 24 transformer

# Dual: two separate files per song
#   <stem>_instru.npy: (N_segs, 13, 768)
#   <stem>_vocal.npy:  (N_segs, 25, 1024)
# Layer selection at dataset time (same as single stream)
DUAL_FEATURE_DIM = (len(MERT_LAYERS) * EMBED_DIM + len(VOCAL_LAYERS) * VOCAL_EMBED_DIM) * EVAL_WINDOW_SIZE
# (3*768 + 3*1024) * 12 = 64512

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS    = 10
LAMBDA_REG    = 1.0
SEED          = 42
