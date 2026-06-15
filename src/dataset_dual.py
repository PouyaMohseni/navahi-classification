"""
NavahiDualDataset: loads dual-stream (instruments + vocals) pre-extracted features.

Each sample is one 60-second chunk:
  x:      float32 tensor (DUAL_FEATURE_DIM,)  [instruments(2304) | vocals(3072)]
  label:  int in [0, 7]
  coords: float32 tensor (2,)

Identical interface to NavahiDataset — swap dataset class to switch streams.
"""

import os
import sys

import numpy as np
import openpyxl
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FEATURES_DUAL_DIR, SPLIT9_DIR,
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


def _normalize_coords(lat, lon):
    return np.clip(
        np.array([(lat - LAT_MIN) / (LAT_MAX - LAT_MIN),
                  (lon - LON_MIN) / (LON_MAX - LON_MIN)], dtype=np.float32),
        0.0, 1.0,
    )


def _load_split_metadata(split):
    path = os.path.join(SPLIT9_DIR, f"{split}.xlsx")
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    h = list(rows[0])
    fn_col, genre_col = h.index("File Name"), h.index("Genre")
    lat_col, lon_col  = h.index("State_x"),   h.index("State_y")

    records = []
    for r in rows[1:]:
        fname, genre = r[fn_col], r[genre_col]
        if not fname or genre not in _GENRE_TO_LABEL:
            continue
        label = _GENRE_TO_LABEL[genre]
        lat, lon = r[lat_col], r[lon_col]
        if lat and lon:
            coords = _normalize_coords(float(lat), float(lon))
        else:
            coords = _normalize_coords(*CLASS_COORDS[label])
        stem = os.path.splitext(fname)[0]
        records.append((stem, label, coords))
    return records


class NavahiDualDataset(Dataset):
    def __init__(self, split: str = "train"):
        assert split in ("train", "val", "test")
        self.samples = []

        feat_dir = os.path.join(FEATURES_DUAL_DIR, split)
        metadata = _load_split_metadata(split)

        missing = 0
        for stem, label, coords in metadata:
            feat_path = os.path.join(feat_dir, stem + ".npy")
            if not os.path.exists(feat_path):
                missing += 1
                continue
            n_chunks = np.load(feat_path, mmap_mode="r").shape[0]
            for ci in range(n_chunks):
                self.samples.append((feat_path, ci, label, coords))

        if missing:
            print(f"[NavahiDualDataset/{split}] {missing}/{len(metadata)} files "
                  "missing — run extract_features_dual.py first")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, ci, label, coords = self.samples[idx]
        x = torch.from_numpy(np.load(feat_path)[ci].astype(np.float32))
        return x, label, torch.from_numpy(coords)
