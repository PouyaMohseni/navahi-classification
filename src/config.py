import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "Navahi_data", "Navahi")
AUDIO_ROOT = os.path.join(DATA_ROOT, "Navahi-Dataset")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")

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

# Training
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
LAMBDA_REG = 1.0               # weight on regression loss
VAL_RATIO = 0.1                # fraction of train used for validation
SEED = 42
