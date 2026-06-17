"""
NavahiDualDataset — loads separate instrument and vocal feature files and
builds windows by concatenating consecutive segments (same as single stream).

Features on disk per song:
  <stem>_instru.npy  (N_segs, 13, 768)   — all MERT hidden states
  <stem>_vocal.npy   (N_segs, 25, 1024)  — all wav2vec2-xlsr-53 hidden states

Window construction:
  1. Take stack_size consecutive segments from same song
  2. For instruments: np.concatenate → (13, 768*stack_size),  select instru_indices
  3. For vocals:      np.concatenate → (25, 1024*stack_size), select vocal_indices
  4. Flatten each and concatenate → final feature vector

Default indices [6,7,8] for both streams (middle layers).

Stride:
  overlap=False  (training)   stride = stack_size
  overlap=True   (val/test)   stride = 1
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
    MERT_LAYERS, VOCAL_LAYERS,
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
    return np.array([(lat - LAT_MIN) / (LAT_MAX - LAT_MIN),
                     (lon - LON_MIN) / (LON_MAX - LON_MIN)], dtype=np.float32)


def _load_split_metadata(split: str) -> list:
    wb = openpyxl.load_workbook(os.path.join(SPLIT9_DIR, f"{split}.xlsx"))
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    h = list(rows[0])
    fn_col, genre_col = h.index("File Name"), h.index("Genre")
    if split == "test_simplified":
        lat_col = h.index("Genre_x")
        lon_col = h.index("Genre_y")
    else:
        lat_col = h.index("State_x")
        lon_col = h.index("State_y")
    sel_col  = h.index("Select")  if "Select"  in h else None
    sel2_col = h.index("Select2") if "Select2" in h else None

    records = []
    for r in rows[1:]:
        if sel_col  is not None and r[sel_col]  == 0:
            continue
        if sel2_col is not None and r[sel2_col] == 0:
            continue
        fname, genre = r[fn_col], r[genre_col]
        if not fname or genre not in _GENRE_TO_LABEL:
            continue
        label = _GENRE_TO_LABEL[genre]
        lat, lon = r[lat_col], r[lon_col]
        coords = _normalize_coords(float(lat), float(lon)) if (lat and lon) \
                 else _normalize_coords(*CLASS_COORDS[label])
        records.append((os.path.splitext(fname)[0], label, coords))
    return records


class NavahiDualDataset(Dataset):
    def __init__(
        self,
        split:          str  = "train",
        stack_size:     int  = 12,
        overlap:        bool = False,
        instru_indices: list = None,
        vocal_indices:  list = None,
    ):
        assert split in ("train", "val", "test", "test_simplified")
        self.stack_size     = stack_size
        self.instru_indices = instru_indices if instru_indices is not None else MERT_LAYERS
        self.vocal_indices  = vocal_indices  if vocal_indices  is not None else VOCAL_LAYERS
        stride = 1 if overlap else stack_size

        self.samples = []  # (instru_path, vocal_path, start, label, coords, file_idx)
        feat_split = "test" if split == "test_simplified" else split
        feat_dir  = os.path.join(FEATURES_DUAL_DIR, feat_split)
        metadata  = _load_split_metadata(split)
        missing   = 0
        file_idx  = 0

        for stem, label, coords in metadata:
            instru_path = os.path.join(feat_dir, stem + "_instru.npy")
            vocal_path  = os.path.join(feat_dir, stem + "_vocal.npy")
            if not os.path.exists(instru_path) or not os.path.exists(vocal_path):
                missing += 1
                continue
            n_segs = np.load(instru_path, mmap_mode="r").shape[0]
            start = 0
            while start + stack_size <= n_segs:
                self.samples.append((instru_path, vocal_path, start, label, coords, file_idx))
                start += stride
            file_idx += 1

        if missing:
            print(f"[NavahiDualDataset/{split}] {missing}/{len(metadata)} files missing "
                  "— run extract_features_dual.py first")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        instru_path, vocal_path, start, label, coords, file_idx = self.samples[idx]

        instru = np.load(instru_path)                        # (N_segs, 13, 768)
        vocal  = np.load(vocal_path)                         # (N_segs, 25, 1024)

        iw = instru[start : start + self.stack_size]         # (W, 13, 768)
        vw = vocal[start  : start + self.stack_size]         # (W, 25, 1024)

        # Concatenate segments along last axis
        istacked = np.concatenate(iw, axis=-1)               # (13, 768*W)
        vstacked = np.concatenate(vw, axis=-1)               # (25, 1024*W)

        # Select layers
        isel = istacked[self.instru_indices, :]              # (3, 768*W)
        vsel = vstacked[self.vocal_indices,  :]              # (3, 1024*W)

        # Flatten and concatenate both streams
        x = np.concatenate([isel.reshape(-1), vsel.reshape(-1)])  # (3*768*W + 3*1024*W,)

        return torch.from_numpy(x.astype(np.float32)), label, torch.from_numpy(coords), file_idx
