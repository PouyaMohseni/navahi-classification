"""
NavahiDataset — loads (N_segs, 13, 768) per-file features and builds
windows by CONCATENATING consecutive segments (not mean-pooling).

Window construction (matches mlptest/mlptest-reg2.py StackedAudioDataset):
  1. Take stack_size consecutive segments from the same file
     → array (stack_size, 13, 768)
  2. Concatenate along the last axis:
     np.concatenate(segs, axis=-1) → (13, 768*stack_size)
  3. Select time_indices rows → (len(time_indices), 768*stack_size)
  4. Flatten → len(time_indices)*768*stack_size  (e.g. 3*768*12 = 27648)

Stride:
  overlap=False  (training)   stride = stack_size  (non-overlapping)
  overlap=True   (val/test)   stride = 1           (sliding window)

Each __getitem__ returns:
  x:      float32 tensor (FEATURE_DIM,)
  label:  int in [0, 7]
  coords: float32 tensor (2,) — normalised [lat, lon] ∈ [0, 1]
"""

import os
import sys

import numpy as np
import openpyxl
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FEATURES_DIR, SPLIT9_DIR,
    MERT_LAYERS,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, CLASS_COORDS,
)

_GENRE_TO_LABEL = {
    "Gilan":              0,
    "Lorestan":           1,
    "Khorasan":           2,
    "Kordestan":          3,
    "Azerbaijan":         4,
    "Sistan&Baluchestan": 5,
    "Turkaman":           6,
    "Bushehr":            7,
}


def _normalize_coords(lat: float, lon: float) -> np.ndarray:
    return np.clip(
        np.array([(lat - LAT_MIN) / (LAT_MAX - LAT_MIN),
                  (lon - LON_MIN) / (LON_MAX - LON_MIN)], dtype=np.float32),
        0.0, 1.0,
    )


def _load_split_metadata(split: str) -> list:
    wb = openpyxl.load_workbook(os.path.join(SPLIT9_DIR, f"{split}.xlsx"))
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    h = list(rows[0])
    fn_col, genre_col = h.index("File Name"), h.index("Genre")
    lat_col, lon_col   = h.index("State_x"),   h.index("State_y")

    records = []
    for r in rows[1:]:
        fname, genre = r[fn_col], r[genre_col]
        if not fname or genre not in _GENRE_TO_LABEL:
            continue
        label = _GENRE_TO_LABEL[genre]
        lat, lon = r[lat_col], r[lon_col]
        coords = _normalize_coords(float(lat), float(lon)) if (lat and lon) \
                 else _normalize_coords(*CLASS_COORDS[label])
        records.append((os.path.splitext(fname)[0], label, coords))
    return records


class NavahiDataset(Dataset):
    def __init__(
        self,
        split:        str  = "train",
        stack_size:   int  = 12,
        overlap:      bool = False,
        time_indices: list = None,
    ):
        assert split in ("train", "val", "test")
        self.stack_size   = stack_size
        self.time_indices = time_indices if time_indices is not None else MERT_LAYERS
        stride = 1 if overlap else stack_size

        self.samples = []  # (feat_path, start, label, coords)
        feat_dir  = os.path.join(FEATURES_DIR, split)
        metadata  = _load_split_metadata(split)
        missing   = 0

        for stem, label, coords in metadata:
            feat_path = os.path.join(feat_dir, stem + ".npy")
            if not os.path.exists(feat_path):
                missing += 1
                continue
            n_segs = np.load(feat_path, mmap_mode="r").shape[0]
            start = 0
            while start + stack_size <= n_segs:
                self.samples.append((feat_path, start, label, coords))
                start += stride

        if missing:
            print(f"[NavahiDataset/{split}] {missing}/{len(metadata)} files missing "
                  "— run extract_features.py first")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, start, label, coords = self.samples[idx]
        feats  = np.load(feat_path)                      # (N_segs, 13, 768)
        window = feats[start : start + self.stack_size]  # (stack_size, 13, 768)

        # Concatenate segments along last axis → (13, 768*stack_size)
        stacked = np.concatenate(window, axis=-1)

        # Select layers → (len(time_indices), 768*stack_size)
        selected = stacked[self.time_indices, :]

        x = torch.from_numpy(selected.reshape(-1).astype(np.float32))
        return x, label, torch.from_numpy(coords)
