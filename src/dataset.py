"""
NavahiDataset: loads pre-extracted MERT features from disk.

Each sample is one 60-second chunk from an audio file:
  - x: float32 tensor of shape (FEATURE_DIM,)  [2304-dim MERT embedding]
  - label: int in [0, 7]
  - coords: float32 tensor of shape (2,)  [normalized lat, lon in [0, 1]]
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FEATURES_DIR, CLASS_MAP, CLASS_COORDS,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, FEATURE_DIM,
    VAL_RATIO, SEED,
)


def normalize_coords(lat: float, lon: float):
    lat_n = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    lon_n = (lon - LON_MIN) / (LON_MAX - LON_MIN)
    return np.array([lat_n, lon_n], dtype=np.float32)


class NavahiDataset(Dataset):
    def __init__(self, split: str = "train"):
        """
        split: "train" | "val" | "test"

        train/val are carved from the features/train directory (90/10 split).
        test uses features/test.
        """
        assert split in ("train", "val", "test")
        self.samples = []  # list of (feat_path, chunk_idx, label, coords_normalized)

        if split in ("train", "val"):
            raw_split = "train"
        else:
            raw_split = "test"

        feat_split_dir = os.path.join(FEATURES_DIR, raw_split)

        all_files = []
        for cls_folder, label in CLASS_MAP.items():
            cls_dir = os.path.join(feat_split_dir, cls_folder)
            if not os.path.isdir(cls_dir):
                continue
            lat, lon = CLASS_COORDS[label]
            coords = normalize_coords(lat, lon)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith(".npy"):
                    all_files.append((os.path.join(cls_dir, fname), label, coords))

        if raw_split == "train":
            rng = np.random.default_rng(SEED)
            idx = np.arange(len(all_files))
            rng.shuffle(idx)
            n_val = max(1, int(len(all_files) * VAL_RATIO))
            val_idx = set(idx[:n_val].tolist())
            train_idx = set(idx[n_val:].tolist())
            chosen = val_idx if split == "val" else train_idx
            all_files = [all_files[i] for i in sorted(chosen)]

        for feat_path, label, coords in all_files:
            feats = np.load(feat_path)  # (N_chunks, FEATURE_DIM)
            for chunk_idx in range(len(feats)):
                self.samples.append((feat_path, chunk_idx, label, coords))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, chunk_idx, label, coords = self.samples[idx]
        feats = np.load(feat_path)
        x = torch.from_numpy(feats[chunk_idx].astype(np.float32))
        return x, label, torch.from_numpy(coords)
