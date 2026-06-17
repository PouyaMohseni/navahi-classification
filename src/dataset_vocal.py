"""
NavahiVocalDataset — loads wav2vec2-xlsr-53 vocal features only.
Feature files: <stem>_vocal.npy  (N_segs, 25, 1024)
Located in FEATURES_DUAL_DIR/<split>/  (same as dual dataset).
"""

import os
import sys

import numpy as np
import openpyxl
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FEATURES_DUAL_DIR, SPLIT9_DIR, VOCAL_LAYERS,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, CLASS_COORDS,
)

_GENRE_TO_LABEL = {
    "Gilan": 0, "Lorestan": 1, "Khorasan": 2, "Kordestan": 3,
    "Azerbaijan": 4, "Sistan&Baluchestan": 5, "Turkaman": 6, "Bushehr": 7,
}


def _normalize_coords(lat, lon):
    return np.array([(lat - LAT_MIN) / (LAT_MAX - LAT_MIN),
                     (lon - LON_MIN) / (LON_MAX - LON_MIN)], dtype=np.float32)


def _load_split_metadata(split):
    wb = openpyxl.load_workbook(os.path.join(SPLIT9_DIR, f"{split}.xlsx"))
    rows = list(wb.active.iter_rows(values_only=True))
    h = list(rows[0])
    fn_col, genre_col = h.index("File Name"), h.index("Genre")
    lat_col = h.index("Genre_x" if split == "test_simplified" else "State_x")
    lon_col = h.index("Genre_y" if split == "test_simplified" else "State_y")
    sel_col  = h.index("Select")  if "Select"  in h else None
    sel2_col = h.index("Select2") if "Select2" in h else None
    records = []
    for r in rows[1:]:
        if sel_col  is not None and r[sel_col]  == 0: continue
        if sel2_col is not None and r[sel2_col] == 0: continue
        fname, genre = r[fn_col], r[genre_col]
        if not fname or genre not in _GENRE_TO_LABEL: continue
        label = _GENRE_TO_LABEL[genre]
        lat, lon = r[lat_col], r[lon_col]
        coords = _normalize_coords(float(lat), float(lon)) if (lat and lon) \
                 else _normalize_coords(*CLASS_COORDS[label])
        records.append((os.path.splitext(fname)[0], label, coords))
    return records


class NavahiVocalDataset(Dataset):
    def __init__(self, split="train", stack_size=12, overlap=False, vocal_indices=None):
        assert split in ("train", "val", "test", "test_simplified")
        self.stack_size    = stack_size
        self.vocal_indices = vocal_indices if vocal_indices is not None else VOCAL_LAYERS
        stride     = 1 if overlap else stack_size
        feat_split = "test" if split == "test_simplified" else split
        feat_dir   = os.path.join(FEATURES_DUAL_DIR, feat_split)

        self.samples = []
        missing, file_idx = 0, 0
        for stem, label, coords in _load_split_metadata(split):
            path = os.path.join(feat_dir, stem + "_vocal.npy")
            if not os.path.exists(path):
                missing += 1
                continue
            n_segs = np.load(path, mmap_mode="r").shape[0]
            start = 0
            while start + stack_size <= n_segs:
                self.samples.append((path, start, label, coords, file_idx))
                start += stride
            file_idx += 1
        if missing:
            print(f"[NavahiVocalDataset/{split}] {missing} files missing")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, start, label, coords, file_idx = self.samples[idx]
        vocal   = np.load(path)                              # (N_segs, 25, 1024)
        vw      = vocal[start : start + self.stack_size]    # (W, 25, 1024)
        stacked = np.concatenate(vw, axis=-1)               # (25, 1024*W)
        sel     = stacked[self.vocal_indices, :]            # (3, 1024*W)
        x = torch.from_numpy(sel.reshape(-1).astype(np.float32))
        return x, label, torch.from_numpy(coords), file_idx
