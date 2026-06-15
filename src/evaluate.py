"""
Evaluation metrics matching Table 3 in the paper.

Classification: Accuracy, Balanced Accuracy, Top-2 Accuracy, Precision, Recall, F1
Regression:     MSE, R², Geo-accuracy (normalised), Geo-F1

Usage (standalone):
    python src/evaluate.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import NUM_CLASSES, BATCH_SIZE, CLASS_NAMES, FEATURE_DIM, DUAL_FEATURE_DIM
from dataset import NavahiDataset
from dataset_dual import NavahiDualDataset
from model import NavahiClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    coords_pred: torch.Tensor,
    coords_true: torch.Tensor,
) -> dict:
    preds = logits.argmax(1).numpy()
    labels_np = labels.numpy()
    logits_np = logits.numpy()
    cp = coords_pred.numpy()
    ct = coords_true.numpy()

    # Top-2 accuracy
    top2 = np.argsort(logits_np, axis=1)[:, -2:]
    top2_correct = sum(labels_np[i] in top2[i] for i in range(len(labels_np)))

    # Classification metrics
    acc = (preds == labels_np).mean()
    bal_acc = balanced_accuracy_score(labels_np, preds)
    top2_acc = top2_correct / len(labels_np)
    prec = precision_score(labels_np, preds, average="weighted", zero_division=0)
    rec = recall_score(labels_np, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels_np, preds, average="weighted", zero_division=0)

    # Regression metrics
    mse = float(np.mean((cp - ct) ** 2))
    r2 = float(r2_score(ct, cp))

    return {
        "acc": acc,
        "balanced_acc": bal_acc,
        "top2_acc": top2_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "reg_mse": mse,
        "reg_r2": r2,
    }


def print_metrics(m: dict, header: str = ""):
    if header:
        print(f"\n=== {header} ===")
    print(f"  Accuracy:         {m['acc']:.4f}  ({m['acc']*100:.2f}%)")
    print(f"  Balanced Acc:     {m['balanced_acc']:.4f}  ({m['balanced_acc']*100:.2f}%)")
    print(f"  Top-2 Accuracy:   {m['top2_acc']:.4f}  ({m['top2_acc']*100:.2f}%)")
    print(f"  Precision:        {m['precision']:.4f}  ({m['precision']*100:.2f}%)")
    print(f"  Recall:           {m['recall']:.4f}  ({m['recall']*100:.2f}%)")
    print(f"  F1:               {m['f1']:.4f}  ({m['f1']*100:.2f}%)")
    print(f"  Reg MSE:          {m['reg_mse']:.4f}")
    print(f"  Reg R²:           {m['reg_r2']:.4f}")


@torch.no_grad()
def evaluate(checkpoint_path: str, split: str = "test", dual: bool = False,
             window_size: int = 12, device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        return None

    input_dim = DUAL_FEATURE_DIM if dual else FEATURE_DIM
    model = NavahiClassifier(input_dim=input_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = NavahiDualDataset(split, window_size=window_size) if dual else NavahiDataset(split, window_size=window_size)
    if len(ds) == 0:
        print(f"ERROR: 0 samples found for split='{split}' (dual={dual}). Features missing?")
        return None
    pin = device.type == "cuda"
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=pin)
    print(f"Evaluating on {split} ({len(ds)} samples, dual={dual})")

    all_preds, all_labels, all_cp, all_ct = [], [], [], []
    for x, labels, coords in loader:
        x = x.to(device)
        logits, cp = model(x)
        all_preds.append(logits.cpu())
        all_labels.append(labels)
        all_cp.append(cp.cpu())
        all_ct.append(coords)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_cp = torch.cat(all_cp)
    all_ct = torch.cat(all_ct)

    label_sec = window_size * 5
    m = compute_metrics(all_preds, all_labels, all_cp, all_ct)
    print_metrics(m, header=f"{label_sec}s windows — {os.path.basename(checkpoint_path)}")

    preds_np = all_preds.argmax(1).numpy()
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
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--dual", action="store_true", help="Use dual-stream dataset and model")
    parser.add_argument("--window_size", type=int, default=12,
                        help="Number of 5s segments to mean-pool per eval sample (12=60s, 6=30s)")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.split, dual=args.dual, window_size=args.window_size)


if __name__ == "__main__":
    main()
