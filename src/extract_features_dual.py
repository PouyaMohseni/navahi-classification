"""
Dual-stream feature extraction: separate audio into vocals and instruments
using Demucs (htdemucs), then encode each stem with its own model.

  instruments = drums + bass + other  → MERT        → (N_segs, 13, 768)
  vocals                              → wav2vec2-xlsr-53 → (N_segs, 25, 1024)

Saved as two files per song:
  features_dual/<split>/<stem>_instru.npy   shape (N_segs, 13, 768)
  features_dual/<split>/<stem>_vocal.npy    shape (N_segs, 25, 1024)

Layer selection and window stacking happen at dataset time (same philosophy
as the single-stream pipeline).

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
    MERT_MODEL, MERT_SAMPLE_RATE,
    VOCAL_MODEL,
    SEGMENT_SEC,
)

VOCAL_SAMPLE_RATE = 16000
TARGET_DB = -20.0
EPS = 1e-9


def rms_normalize(audio: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    return audio * (10 ** (TARGET_DB / 20)) / (rms + EPS)


def segment_audio(audio: np.ndarray, sr: int) -> list:
    seg_len = SEGMENT_SEC * sr
    segs = []
    for start in range(0, len(audio), seg_len):
        s = audio[start:start + seg_len]
        if len(s) < seg_len:
            s = np.pad(s, (0, seg_len - len(s)))
        segs.append(s)
    return segs


def separate_stems(audio_path: str, device: torch.device):
    """Returns (vocals_np at 16kHz, instruments_np at 24kHz)."""
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import convert_audio

    model = get_model("htdemucs")
    model.to(device).eval()

    audio_np, sr = librosa.load(audio_path, sr=None, mono=False)
    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, :]
    wav = torch.from_numpy(audio_np.astype(np.float32))
    wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)
    wav = wav.unsqueeze(0).to(device)

    with torch.no_grad():
        sources = apply_model(model, wav, device=device)[0]

    stem_names = model.sources
    stems = {name: sources[i] for i, name in enumerate(stem_names)}

    vocals = stems["vocals"].mean(0).cpu().numpy()
    instru = (stems["drums"] + stems["bass"] + stems["other"]).mean(0).cpu().numpy()

    native_sr = model.samplerate
    vocals = rms_normalize(librosa.resample(vocals, orig_sr=native_sr, target_sr=VOCAL_SAMPLE_RATE))
    instru = rms_normalize(librosa.resample(instru, orig_sr=native_sr, target_sr=MERT_SAMPLE_RATE))
    return vocals, instru


@torch.no_grad()
def embed_mert(audio: np.ndarray, model, processor, device) -> np.ndarray:
    """Returns (13, 768) — all hidden states, mean-pooled over time."""
    inputs = processor(audio, sampling_rate=MERT_SAMPLE_RATE,
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs, output_hidden_states=True)
    # (13, 1, T, 768) → mean over T → (13, 768)
    return torch.stack(out.hidden_states).squeeze(1).mean(-2).cpu().numpy()


@torch.no_grad()
def embed_wav2vec2(audio: np.ndarray, model, processor, device) -> np.ndarray:
    """Returns (25, 1024) — all hidden states, mean-pooled over time."""
    inputs = processor(audio, sampling_rate=VOCAL_SAMPLE_RATE,
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs, output_hidden_states=True)
    return torch.stack(out.hidden_states).squeeze(1).mean(-2).cpu().numpy()


def extract_dual_features(
    audio_path: str,
    mert_model, mert_proc,
    wav2vec_model, wav2vec_proc,
    device: torch.device,
):
    """Returns (instru_feats, vocal_feats):
       instru_feats: (N_segs, 13, 768)
       vocal_feats:  (N_segs, 25, 1024)
    """
    vocals_full, instru_full = separate_stems(audio_path, device)

    vocal_segs = segment_audio(vocals_full, VOCAL_SAMPLE_RATE)
    instru_segs = segment_audio(instru_full, MERT_SAMPLE_RATE)
    n_segs = min(len(vocal_segs), len(instru_segs))

    instru_list, vocal_list = [], []
    for i in range(n_segs):
        instru_list.append(embed_mert(instru_segs[i], mert_model, mert_proc, device))
        vocal_list.append(embed_wav2vec2(vocal_segs[i], wav2vec_model, wav2vec_proc, device))

    return (np.stack(instru_list).astype(np.float32),   # (N_segs, 13, 768)
            np.stack(vocal_list).astype(np.float32))    # (N_segs, 25, 1024)


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
        instru_path = os.path.join(out_dir, stem + "_instru.npy")
        vocal_path  = os.path.join(out_dir, stem + "_vocal.npy")
        if os.path.exists(instru_path) and os.path.exists(vocal_path):
            continue
        audio_path = file_index.get(fname)
        if audio_path is None:
            errors.append((fname, "not in index"))
            continue
        try:
            instru_feats, vocal_feats = extract_dual_features(
                audio_path, mert_model, mert_proc, wav2vec_model, wav2vec_proc, device)
            np.save(instru_path, instru_feats)
            np.save(vocal_path,  vocal_feats)
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
    print(f"Saved to: {FEATURES_DUAL_DIR}")
    print("  <stem>_instru.npy: (N_segs, 13, 768)")
    print("  <stem>_vocal.npy:  (N_segs, 25, 1024)")


if __name__ == "__main__":
    main()
