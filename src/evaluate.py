"""
Evaluation metrics matching Table 3 in the paper.

Classification: Accuracy, Balanced Accuracy, Top-2 Accuracy, Precision, Recall, F1
Regression:     MSE, R²

Usage:
    python src/evaluate.py --checkpoint checkpoints/best_model.pt [--window_size 12]
    python src/evaluate.py --checkpoint checkpoints_dual/best_model.pt --dual
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    NUM_CLASSES, BATCH_SIZE, CLASS_NAMES,
    FEATURE_DIM, DUAL_FEATURE_DIM, EVAL_WINDOW_SIZE,
)
from dataset import NavahiDataset
from dataset_dual import NavahiDualDataset
from model import NavahiClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_metrics(logits, labels, coords_pred, coords_true) -> dict:
    preds     = logits.argmax(1).numpy()
    labels_np = labels.numpy()
    logits_np = logits.numpy()
    cp, ct    = coords_pred.numpy(), coords_true.numpy()

    top2 = np.argsort(logits_np, axis=1)[:, -2:]
    top2_acc = sum(labels_np[i] in top2[i] for i in range(len(labels_np))) / len(labels_np)

    return {
        "acc":          float((preds == labels_np).mean()),
        "balanced_acc": float(balanced_accuracy_score(labels_np, preds)),
        "top2_acc":     float(top2_acc),
        "precision":    float(precision_score(labels_np, preds, average="weighted", zero_division=0)),
        "recall":       float(recall_score(labels_np, preds, average="weighted", zero_division=0)),
        "f1":           float(f1_score(labels_np, preds, average="weighted", zero_division=0)),
        "reg_mse":      float(np.mean((cp - ct) ** 2)),
        "reg_r2":       float(r2_score(ct, cp)),
    }


def print_metrics(m: dict, header: str = ""):
    if header:
        print(f"\n=== {header} ===")
    print(f"  Accuracy:       {m['acc']:.4f}  ({m['acc']*100:.2f}%)")
    print(f"  Balanced Acc:   {m['balanced_acc']:.4f}  ({m['balanced_acc']*100:.2f}%)")
    print(f"  Top-2 Accuracy: {m['top2_acc']:.4f}  ({m['top2_acc']*100:.2f}%)")
    print(f"  Precision:      {m['precision']:.4f}")
    print(f"  Recall:         {m['recall']:.4f}")
    print(f"  F1:             {m['f1']:.4f}")
    print(f"  Reg MSE:        {m['reg_mse']:.4f}")
    print(f"  Reg R²:         {m['reg_r2']:.4f}")


@torch.no_grad()
def evaluate(checkpoint_path: str, split: str = "test", dual: bool = False,
             window_size: int = EVAL_WINDOW_SIZE, device: torch.device = None):
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        return None

    # Infer input_dim from checkpoint if available, else use config default
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_dim = ckpt.get("input_dim", DUAL_FEATURE_DIM if dual else FEATURE_DIM)

    model = NavahiClassifier(input_dim=input_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if dual:
        ds = NavahiDualDataset(split, stack_size=window_size, overlap=True)
    else:
        ds = NavahiDataset(split, stack_size=window_size, overlap=True)

    if len(ds) == 0:
        print(f"ERROR: 0 samples found for split='{split}' (dual={dual}). "
              "Features missing — run extract_features.py first.")
        return None

    pin = device.type == "cuda"
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=pin)
    print(f"Evaluating {split} split: {len(ds)} samples, "
          f"window={window_size}×5s={window_size*5}s, dual={dual}")

    all_preds, all_labels, all_cp, all_ct = [], [], [], []
    for x, labels, coords in loader:
        logits, cp = model(x.to(device))
        all_preds.append(logits.cpu())
        all_labels.append(labels)
        all_cp.append(cp.cpu())
        all_ct.append(coords)

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_cp     = torch.cat(all_cp)
    all_ct     = torch.cat(all_ct)

    m = compute_metrics(all_preds, all_labels, all_cp, all_ct)
    print_metrics(m, header=f"{window_size*5}s windows — {os.path.basename(checkpoint_path)}")

    preds_np  = all_preds.argmax(1).numpy()
    labels_np = all_labels.numpy()
    print("\nPer-class accuracy:")
    for c in range(NUM_CLASSES):
        mask = labels_np == c
        if mask.sum() == 0:
            continue
        cls_acc = (preds_np[mask] == c).mean()
        print(f"  {CLASS_NAMES[c]:<35} {cls_acc*100:5.1f}%  (n={mask.sum()})")

    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt"))
    parser.add_argument("--split",       default="test", choices=["train", "val", "test", "test_simplified"])
    parser.add_argument("--dual",        action="store_true")
    parser.add_argument("--window_size", type=int, default=EVAL_WINDOW_SIZE,
                        help="Number of 5s segments per eval window (12=60s, 6=30s)")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.split, dual=args.dual, window_size=args.window_size)


if __name__ == "__main__":
    main()
