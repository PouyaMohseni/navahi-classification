"""
NavahiDataset: loads pre-extracted MERT features using the official Split9 split.

Each sample is one 60-second chunk from an audio file:
  - x:      float32 tensor (FEATURE_DIM,)  [2304-dim MERT embedding]
  - label:  int in [0, 7]
  - coords: float32 tensor (2,)  [normalized lat, lon in [0, 1]]

Lat/lon come from Split9 per-file annotations (State_x, State_y columns).
"""

import os
import sys

import numpy as np
import openpyxl
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FEATURES_DIR, SPLIT9_DIR, CLASS_MAP,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
)

# Map Genre column values → integer labels
_GENRE_TO_LABEL = {
    "Gilan":             0,
    "Lorestan":          1,
    "Khorasan":          2,
    "Kordestan":         3,
    "Azerbaijan":        4,
    "Sistan&Baluchestan":5,
    "Turkaman":          6,
    "Bushehr":           7,
}


def _normalize_coords(lat: float, lon: float) -> np.ndarray:
    lat_n = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    lon_n = (lon - LON_MIN) / (LON_MAX - LON_MIN)
    return np.clip(np.array([lat_n, lon_n], dtype=np.float32), 0.0, 1.0)


def _load_split_metadata(split: str) -> list[tuple[str, int, np.ndarray]]:
    """
    Returns list of (filename_stem, label, coords_normalized) from Split9/<split>.xlsx.
    Rows with missing genre or coordinates are skipped.
    """
    path = os.path.join(SPLIT9_DIR, f"{split}.xlsx")
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    headers = list(rows[0])

    fn_col    = headers.index("File Name")
    genre_col = headers.index("Genre")
    lat_col   = headers.index("State_x")
    lon_col   = headers.index("State_y")

    records = []
    for r in rows[1:]:
        fname, genre, lat, lon = r[fn_col], r[genre_col], r[lat_col], r[lon_col]
        if not fname or genre not in _GENRE_TO_LABEL:
            continue
        label = _GENRE_TO_LABEL[genre]
        if lat and lon:
            coords = _normalize_coords(float(lat), float(lon))
        else:
            # Fall back to class-level mean from config
            from config import CLASS_COORDS
            coords = _normalize_coords(*CLASS_COORDS[label])
        stem = os.path.splitext(fname)[0]
        records.append((stem, label, coords))
    return records


class NavahiDataset(Dataset):
    def __init__(self, split: str = "train"):
        assert split in ("train", "val", "test")
        self.samples = []  # (feat_path, chunk_idx, label, coords)

        feat_split_dir = os.path.join(FEATURES_DIR, split)
        metadata = _load_split_metadata(split)

        missing = 0
        for stem, label, coords in metadata:
            feat_path = os.path.join(feat_split_dir, stem + ".npy")
            if not os.path.exists(feat_path):
                missing += 1
                continue
            n_chunks = np.load(feat_path, mmap_mode="r").shape[0]
            for chunk_idx in range(n_chunks):
                self.samples.append((feat_path, chunk_idx, label, coords))

        if missing:
            print(f"[NavahiDataset/{split}] {missing}/{len(metadata)} files missing features — run extract_features.py first")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, chunk_idx, label, coords = self.samples[idx]
        feats = np.load(feat_path)
        x = torch.from_numpy(feats[chunk_idx].astype(np.float32))
        return x, label, torch.from_numpy(coords)
