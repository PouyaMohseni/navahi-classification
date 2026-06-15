"""
Extract MERT embeddings from all audio files and save to disk.

For each audio file:
  1. RMS-normalize to -20 dB
  2. Segment into non-overlapping 60-second chunks
  3. Each 60-second chunk → twelve 5-second sub-segments
  4. Run MERT on each 5-second sub-segment
  5. Mean-pool each layer over the time dimension
  6. Concatenate layers 6, 7, 8 → 2304-dim vector per sub-segment
  7. Mean-pool over 12 sub-segments → 2304-dim vector per 60-second chunk
  8. Save all chunk embeddings for the file as a .npy array

Usage:
    python src/extract_features.py [--split train|test|all]
"""

import argparse
import os
import sys

import numpy as np
import torch
import librosa
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    AUDIO_ROOT, FEATURES_DIR, CLASS_MAP,
    MERT_MODEL, MERT_LAYERS, MERT_SAMPLE_RATE,
    SEGMENT_SEC, CHUNK_SEC, EMBED_DIM, FEATURE_DIM,
)

TARGET_DB = -20.0
EPS = 1e-9


def rms_normalize(audio: np.ndarray, target_db: float = TARGET_DB) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    scalar = (10 ** (target_db / 20)) / (rms + EPS)
    return audio * scalar


def load_audio(path: str, sr: int = MERT_SAMPLE_RATE) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return rms_normalize(audio)


def segment_audio(audio: np.ndarray, sr: int, seg_sec: int) -> list[np.ndarray]:
    """Split audio into non-overlapping fixed-length segments. Pad last if needed."""
    seg_len = seg_sec * sr
    segments = []
    for start in range(0, len(audio), seg_len):
        seg = audio[start:start + seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        segments.append(seg)
    return segments


@torch.no_grad()
def extract_file_features(
    audio_path: str,
    model: AutoModel,
    processor: Wav2Vec2FeatureExtractor,
    device: torch.device,
) -> np.ndarray:
    """
    Returns array of shape (N_chunks, FEATURE_DIM) where N_chunks = ceil(duration / 60).
    """
    audio = load_audio(audio_path)
    chunks = segment_audio(audio, MERT_SAMPLE_RATE, CHUNK_SEC)
    chunk_embeds = []

    for chunk in chunks:
        sub_segments = segment_audio(chunk, MERT_SAMPLE_RATE, SEGMENT_SEC)
        sub_embeds = []

        for sub in sub_segments:
            inputs = processor(
                sub,
                sampling_rate=MERT_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)

            # hidden_states: tuple of (batch, time, 768) for each layer
            layer_vecs = []
            for layer_idx in MERT_LAYERS:
                hidden = outputs.hidden_states[layer_idx]  # (1, T, 768)
                vec = hidden.mean(dim=1).squeeze(0).cpu().numpy()  # (768,)
                layer_vecs.append(vec)

            sub_embeds.append(np.concatenate(layer_vecs))  # (2304,)

        chunk_embeds.append(np.stack(sub_embeds).mean(axis=0))  # (2304,)

    return np.stack(chunk_embeds)  # (N_chunks, 2304)


def process_split(split: str, model, processor, device):
    split_dir = os.path.join(AUDIO_ROOT, split)
    out_dir = os.path.join(FEATURES_DIR, split)
    os.makedirs(out_dir, exist_ok=True)

    audio_files = []
    for cls_folder in CLASS_MAP:
        cls_dir = os.path.join(split_dir, cls_folder)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(".mp3"):
                audio_files.append((cls_folder, fname))

    print(f"\n[{split}] {len(audio_files)} files")
    errors = []

    for cls_folder, fname in tqdm(audio_files, desc=split):
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, cls_folder, stem + ".npy")
        if os.path.exists(out_path):
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        audio_path = os.path.join(split_dir, cls_folder, fname)

        try:
            feats = extract_file_features(audio_path, model, processor, device)
            np.save(out_path, feats)
        except Exception as e:
            errors.append((audio_path, str(e)))
            print(f"\n  ERROR: {fname}: {e}")

    if errors:
        print(f"\n{len(errors)} files failed in [{split}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all", choices=["train", "test", "all"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    ) if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")

    print(f"Loading MERT model: {MERT_MODEL}")
    model = AutoModel.from_pretrained(MERT_MODEL, trust_remote_code=True).to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_MODEL, trust_remote_code=True)
    model.eval()

    splits = ["train", "test"] if args.split == "all" else [args.split]
    for split in splits:
        process_split(split, model, processor, device)

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
