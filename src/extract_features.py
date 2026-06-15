"""
Extract MERT embeddings from all audio files referenced in Split9 and save to disk.

For each audio file:
  1. RMS-normalize to -20 dB
  2. Segment into non-overlapping 60-second chunks
  3. Each 60-second chunk → twelve 5-second sub-segments
  4. Run MERT on each 5-second sub-segment
  5. Mean-pool each layer over the time dimension
  6. Concatenate layers 6, 7, 8 → 2304-dim vector per sub-segment
  7. Mean-pool over 12 sub-segments → 2304-dim vector per 60-second chunk
  8. Save as features/<split>/<filename_stem>.npy  shape (N_chunks, 2304)

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
    NAVAHI_ROOT, DATA_ROOT, SPLIT9_DIR, AUDIO_ROOT,
    FEATURES_DIR,
    MERT_MODEL, MERT_LAYERS, MERT_SAMPLE_RATE,
    SEGMENT_SEC, CHUNK_SEC,
)

TARGET_DB = -20.0
EPS = 1e-9


def build_file_index(navahi_root: str) -> dict[str, str]:
    """Return {filename: absolute_path} for every mp3 under Data/ and Navahi-Dataset/."""
    index = {}
    data_root = os.path.join(navahi_root, "Data")
    for src in ["Mahoor", "Spotify", "Cassette", "AppleMusic"]:
        d = os.path.join(data_root, src)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith(".mp3"):
                index[f] = os.path.join(d, f)

    nav_root = os.path.join(navahi_root, "Navahi-Dataset")
    for split in ["train", "test"]:
        split_dir = os.path.join(nav_root, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in os.listdir(split_dir):
            d = os.path.join(split_dir, cls)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.lower().endswith(".mp3"):
                    index[f] = os.path.join(d, f)
    return index


def load_split_filenames(split: str) -> list[str]:
    """Return list of File Name values from Split9/<split>.xlsx."""
    path = os.path.join(SPLIT9_DIR, f"{split}.xlsx")
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    headers = list(rows[0])
    fn_col = headers.index("File Name")
    return [r[fn_col] for r in rows[1:] if r[fn_col]]


def rms_normalize(audio: np.ndarray, target_db: float = TARGET_DB) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    scalar = (10 ** (target_db / 20)) / (rms + EPS)
    return audio * scalar


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=MERT_SAMPLE_RATE, mono=True)
    return rms_normalize(audio)


def segment_audio(audio: np.ndarray, seg_sec: int) -> list[np.ndarray]:
    seg_len = seg_sec * MERT_SAMPLE_RATE
    segments = []
    for start in range(0, len(audio), seg_len):
        seg = audio[start:start + seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        segments.append(seg)
    return segments


@torch.no_grad()
def extract_file_features(audio_path, model, processor, device) -> np.ndarray:
    """Returns (N_chunks, FEATURE_DIM) float32 array."""
    audio = load_audio(audio_path)
    chunks = segment_audio(audio, CHUNK_SEC)
    chunk_embeds = []

    for chunk in chunks:
        sub_segments = segment_audio(chunk, SEGMENT_SEC)
        sub_embeds = []

        for sub in sub_segments:
            inputs = processor(sub, sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)

            layer_vecs = []
            for layer_idx in MERT_LAYERS:
                hidden = outputs.hidden_states[layer_idx]  # (1, T, 768)
                vec = hidden.mean(dim=1).squeeze(0).cpu().numpy()
                layer_vecs.append(vec)

            sub_embeds.append(np.concatenate(layer_vecs))  # (2304,)

        chunk_embeds.append(np.stack(sub_embeds).mean(axis=0))  # (2304,)

    return np.stack(chunk_embeds).astype(np.float32)  # (N_chunks, 2304)


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
            errors.append((fname, "not found in index"))
            continue

        try:
            feats = extract_file_features(audio_path, model, processor, device)
            np.save(out_path, feats)
        except Exception as e:
            errors.append((fname, str(e)))
            print(f"\n  ERROR: {fname}: {e}")

    if errors:
        print(f"\n{len(errors)} files failed in [{split}]")
        for f, e in errors[:5]:
            print(f"  {f}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    ) if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")

    print("Building audio file index...")
    file_index = build_file_index(NAVAHI_ROOT)
    print(f"Indexed {len(file_index)} audio files")

    print(f"Loading MERT model: {MERT_MODEL}")
    model = AutoModel.from_pretrained(MERT_MODEL, trust_remote_code=True).to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_MODEL, trust_remote_code=True)
    model.eval()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        process_split(split, file_index, model, processor, device)

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
