import os

# Paths — override with env vars for cluster runs
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_default_navahi = os.path.join(PROJECT_ROOT, "Navahi_data", "Navahi")
_default_features = os.path.join(PROJECT_ROOT, "features")
_default_checkpoints = os.path.join(PROJECT_ROOT, "checkpoints")

NAVAHI_ROOT = os.environ.get("NAVAHI_ROOT", _default_navahi)
AUDIO_ROOT = os.path.join(NAVAHI_ROOT, "Navahi-Dataset")   # fallback: folder-based
DATA_ROOT = os.path.join(NAVAHI_ROOT, "Data")               # Mahoor/Spotify/Cassette/AppleMusic
SPLIT9_DIR = os.path.join(NAVAHI_ROOT, "Split9")            # official split xlsx files
FEATURES_DIR = os.environ.get("NAVAHI_FEATURES_DIR", _default_features)
CHECKPOINTS_DIR = os.environ.get("NAVAHI_CHECKPOINTS_DIR", _default_checkpoints)

# Classes — folder name → integer label
CLASS_MAP = {
    "Gilan&Mazandaran": 0,
    "Lorestan": 1,
    "Khorasan": 2,
    "Kordestan&Kermanshah": 3,
    "Azerbaijan": 4,
    "Sistan&Baluchestan": 5,
    "Turkaman": 6,
    "Bushehr": 7,
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

# Class-level mean geo-coordinates (lat, lon) derived from all.xlsx
# Used as regression targets when per-file coordinates are unavailable
CLASS_COORDS = {
    0: (37.10, 50.48),   # Gilan & Mazandaran
    1: (32.04, 50.56),   # Lorestan
    2: (36.36, 58.21),   # Khorasan
    3: (35.11, 47.01),   # Kordestan & Kermanshah
    4: (37.83, 46.12),   # Azerbaijan
    5: (27.53, 60.52),   # Sistan & Baluchestan
    6: (37.23, 55.13),   # Turkaman
    7: (28.65, 51.75),   # Bushehr
}

# Iran bounding box for coordinate normalization
LAT_MIN, LAT_MAX = 25.064, 39.780
LON_MIN, LON_MAX = 44.039, 63.333

# MERT config
MERT_MODEL = "m-a-p/MERT-v1-95M"
MERT_LAYERS = [6, 7, 8]       # middle transformer layers
MERT_SAMPLE_RATE = 24000
SEGMENT_SEC = 5                # sub-segment length
CHUNK_SEC = 60                 # full chunk fed as context
EMBED_DIM = 768                # hidden size per layer
FEATURE_DIM = len(MERT_LAYERS) * EMBED_DIM  # 2304

# Dual-stream config (vocals + instruments)
FEATURES_DUAL_DIR = os.environ.get("NAVAHI_FEATURES_DUAL_DIR",
                                   os.path.join(PROJECT_ROOT, "features_dual"))
CHECKPOINTS_DUAL_DIR = os.environ.get("NAVAHI_CHECKPOINTS_DUAL_DIR",
                                      os.path.join(PROJECT_ROOT, "checkpoints_dual"))

VOCAL_MODEL = "facebook/wav2vec2-large-xlsr-53"
VOCAL_LAYERS = [6, 7, 8]       # middle transformer layers (hidden size 1024)
VOCAL_EMBED_DIM = 1024
VOCAL_FEATURE_DIM = len(VOCAL_LAYERS) * VOCAL_EMBED_DIM   # 3072

DUAL_FEATURE_DIM = FEATURE_DIM + VOCAL_FEATURE_DIM        # 2304 + 3072 = 5376

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
LAMBDA_REG = 1.0               # weight on regression loss
VAL_RATIO = 0.1                # fraction of train used for validation
SEED = 42
