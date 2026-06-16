"""
Extract MERT embeddings and save ALL 13 hidden states per 5-second segment.

For each audio file:
  1. RMS-normalize to -20 dB
  2. Segment into non-overlapping 5-second windows (last segment zero-padded)
  3. Run MERT on each 5-second segment with output_hidden_states=True
  4. For each of the 13 hidden states: mean-pool over time → 768-dim vector
  5. Stack all 13 → (13, 768) per segment
  6. Save as features/<split>/<stem>.npy  shape (N_segs, 13, 768)

Layer selection (e.g. [6,7,8]) and window stacking happen at dataset/training time,
not here, so features can be reused with different configurations.

Usage:
    python src/extract_features.py [--split train|val|test|all]
"""

import argparse
import os
import sys

import numpy as np
import openpyxl
import torch
import librosa
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    NAVAHI_ROOT, SPLIT9_DIR, FEATURES_DIR,
    MERT_MODEL, MERT_SAMPLE_RATE,
    NUM_HIDDEN_STATES, SEGMENT_SEC,
)

TARGET_DB = -20.0
EPS = 1e-9


def build_file_index(navahi_root: str) -> dict:
    index = {}
    for src in ["Mahoor", "Spotify", "Cassette", "AppleMusic"]:
        d = os.path.join(navahi_root, "Data", src)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".mp3"):
                    index[f] = os.path.join(d, f)
    for split in ["train", "test"]:
        nav = os.path.join(navahi_root, "Navahi-Dataset", split)
        if not os.path.isdir(nav):
            continue
        for cls in os.listdir(nav):
            d = os.path.join(nav, cls)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith(".mp3"):
                        index[f] = os.path.join(d, f)
    return index


def load_split_filenames(split: str) -> list:
    wb = openpyxl.load_workbook(os.path.join(SPLIT9_DIR, f"{split}.xlsx"))
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    col = list(rows[0]).index("File Name")
    return [r[col] for r in rows[1:] if r[col]]


def rms_normalize(audio: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    return audio * (10 ** (TARGET_DB / 20)) / (rms + EPS)


def segment_audio(audio: np.ndarray) -> list:
    seg_len = SEGMENT_SEC * MERT_SAMPLE_RATE
    segs = []
    for start in range(0, len(audio), seg_len):
        s = audio[start:start + seg_len]
        if len(s) < seg_len:
            s = np.pad(s, (0, seg_len - len(s)))
        segs.append(s)
    return segs


@torch.no_grad()
def extract_file_features(audio_path: str, model, processor, device) -> np.ndarray:
    """Returns (N_segs, 13, 768) float32 — all hidden states, mean-pooled over time."""
    audio, _ = librosa.load(audio_path, sr=MERT_SAMPLE_RATE, mono=True)
    audio = rms_normalize(audio)
    segments = segment_audio(audio)

    all_segs = []
    for seg in segments:
        inputs = processor(seg, sampling_rate=MERT_SAMPLE_RATE,
                           return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of 13 tensors, each (1, T, 768)
        layer_vecs = []
        for h in out.hidden_states:            # 13 layers
            vec = h.mean(dim=1).squeeze(0).cpu().numpy()   # (768,)
            layer_vecs.append(vec)

        all_segs.append(np.stack(layer_vecs))  # (13, 768)

    return np.stack(all_segs).astype(np.float32)  # (N_segs, 13, 768)


def process_split(split: str, file_index: dict, model, processor, device):
    filenames = load_split_filenames(split)
    out_dir = os.path.join(FEATURES_DIR, split)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[{split}] {len(filenames)} files")
    errors = []

    for fname in tqdm(filenames, desc=split):
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, stem + ".npy")
        if os.path.exists(out_path):
            continue

        audio_path = file_index.get(fname)
        if audio_path is None:
            errors.append((fname, "not in file index"))
            continue

        try:
            feats = extract_file_features(audio_path, model, processor, device)
            np.save(out_path, feats)
        except Exception as e:
            errors.append((fname, str(e)))
            print(f"\n  ERROR {fname}: {e}")

    if errors:
        print(f"\n{len(errors)} errors in [{split}]:")
        for f, e in errors[:10]:
            print(f"  {f}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all",
                        choices=["train", "val", "test", "test_simplified", "all"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    ) if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    print("Building file index...")
    file_index = build_file_index(NAVAHI_ROOT)
    print(f"Indexed {len(file_index)} audio files")

    print(f"Loading MERT: {MERT_MODEL}")
    model = AutoModel.from_pretrained(MERT_MODEL, trust_remote_code=True).to(device).eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_MODEL, trust_remote_code=True)

    splits = ["train", "val", "test", "test_simplified"] if args.split == "all" else [args.split]
    for split in splits:
        process_split(split, file_index, model, processor, device)

    print("\nFeature extraction complete.")
    print(f"Saved to: {FEATURES_DIR}  (shape per file: N_segs × 13 × 768)")


if __name__ == "__main__":
    main()
