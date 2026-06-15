"""
Dual-stream feature extraction: separate audio into vocals and instruments
using Demucs (htdemucs), then encode each stem independently.

  instruments = drums + bass + other  → MERT layers 6,7,8 → (3, 768) per seg
  vocals                              → wav2vec2-xlsr-53 layers 6,7,8 → (3, 1024) per seg

Per segment, concatenate along last axis:
  (3, 768) + (3, 1024) → (3, 1792)

Saved as features_dual/<split>/<stem>.npy  shape (N_segs, 3, 1792)

Layer stacking/windowing happens at dataset time (same as single-stream).

Usage:
    python src/extract_features_dual.py [--split train|val|test|all]
"""

import argparse
import os
import sys

import numpy as np
import openpyxl
import torch
import librosa
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    NAVAHI_ROOT, SPLIT9_DIR, FEATURES_DUAL_DIR,
    MERT_MODEL, MERT_SAMPLE_RATE, MERT_LAYERS,
    VOCAL_MODEL, VOCAL_LAYERS,
    SEGMENT_SEC,
)

TARGET_DB = -20.0
EPS = 1e-9


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


def separate_stems(audio_path: str, device: torch.device):
    """Returns (vocals_np, instruments_np) at MERT_SAMPLE_RATE."""
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import convert_audio

    model = get_model("htdemucs")
    model.to(device).eval()

    audio_np, sr = librosa.load(audio_path, sr=None, mono=False)
    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, :]   # mono → (1, samples)
    wav = torch.from_numpy(audio_np.astype(np.float32))
    wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)
    wav = wav.unsqueeze(0).to(device)

    with torch.no_grad():
        sources = apply_model(model, wav, device=device)[0]  # (4, ch, samples)

    stem_names = model.sources   # ['drums', 'bass', 'other', 'vocals']
    stems = {name: sources[i] for i, name in enumerate(stem_names)}

    vocals = stems["vocals"].mean(0).cpu().numpy()
    instru = (stems["drums"] + stems["bass"] + stems["other"]).mean(0).cpu().numpy()

    native_sr = model.samplerate
    vocals = rms_normalize(librosa.resample(vocals, orig_sr=native_sr, target_sr=MERT_SAMPLE_RATE))
    instru = rms_normalize(librosa.resample(instru, orig_sr=native_sr, target_sr=MERT_SAMPLE_RATE))
    return vocals, instru


@torch.no_grad()
def embed_mert_seg(seg: np.ndarray, model, processor, device, layers: list) -> np.ndarray:
    """(3, 768) — selected MERT hidden states, mean-pooled over time."""
    inputs = processor(seg, sampling_rate=MERT_SAMPLE_RATE,
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs, output_hidden_states=True)
    vecs = [out.hidden_states[l].mean(1).squeeze(0).cpu().numpy() for l in layers]
    return np.stack(vecs)   # (3, 768)


@torch.no_grad()
def embed_wav2vec_seg(seg: np.ndarray, model, processor, device, layers: list) -> np.ndarray:
    """(3, 1024) — selected wav2vec2 hidden states, mean-pooled over time."""
    inputs = processor(seg, sampling_rate=MERT_SAMPLE_RATE,
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs, output_hidden_states=True)
    vecs = [out.hidden_states[l].mean(1).squeeze(0).cpu().numpy() for l in layers]
    return np.stack(vecs)   # (3, 1024)


def extract_dual_features(
    audio_path: str,
    mert_model, mert_proc,
    wav2vec_model, wav2vec_proc,
    device: torch.device,
) -> np.ndarray:
    """Returns (N_segs, 3, 1792) — instru (3,768) concat vocal (3,1024) per seg."""
    vocals_full, instru_full = separate_stems(audio_path, device)

    vocal_segs = segment_audio(vocals_full)
    instru_segs = segment_audio(instru_full)
    n_segs = min(len(vocal_segs), len(instru_segs))

    out = []
    for i in range(n_segs):
        instru_vec = embed_mert_seg(instru_segs[i], mert_model, mert_proc, device, MERT_LAYERS)
        vocal_vec  = embed_wav2vec_seg(vocal_segs[i], wav2vec_model, wav2vec_proc, device, VOCAL_LAYERS)
        # (3, 768) + (3, 1024) → (3, 1792)
        out.append(np.concatenate([instru_vec, vocal_vec], axis=-1))

    return np.stack(out).astype(np.float32)  # (N_segs, 3, 1792)


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


def process_split(split, file_index, mert_model, mert_proc,
                  wav2vec_model, wav2vec_proc, device):
    filenames = load_split_filenames(split)
    out_dir = os.path.join(FEATURES_DUAL_DIR, split)
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
            errors.append((fname, "not in index"))
            continue
        try:
            feats = extract_dual_features(audio_path, mert_model, mert_proc,
                                          wav2vec_model, wav2vec_proc, device)
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
                        choices=["train", "val", "test", "all"])
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
    mert_model = AutoModel.from_pretrained(MERT_MODEL, trust_remote_code=True).to(device).eval()
    mert_proc  = Wav2Vec2FeatureExtractor.from_pretrained(MERT_MODEL, trust_remote_code=True)

    print(f"Loading vocal model: {VOCAL_MODEL}")
    wav2vec_model = Wav2Vec2Model.from_pretrained(VOCAL_MODEL).to(device).eval()
    wav2vec_proc  = Wav2Vec2FeatureExtractor.from_pretrained(VOCAL_MODEL)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        process_split(split, file_index, mert_model, mert_proc,
                      wav2vec_model, wav2vec_proc, device)

    print("\nDual-stream extraction complete.")
    print(f"Saved to: {FEATURES_DUAL_DIR}  (shape per file: N_segs × 3 × 1792)")


if __name__ == "__main__":
    main()
