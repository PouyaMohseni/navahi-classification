"""
Train the dual-stream classifier on instruments+vocals features.

Identical training loop to train.py — only the dataset and input dim differ.

Usage:
    python src/train_dual.py [--epochs 10] [--batch_size 32] [--lr 2e-5]
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, SEED, DUAL_FEATURE_DIM, CHECKPOINTS_DUAL_DIR, NUM_CLASSES
from dataset_dual import NavahiDualDataset
from model import NavahiClassifier
from evaluate import compute_metrics, print_metrics


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = correct = n = 0
    for x, labels, coords in loader:
        x, labels, coords = x.to(device), labels.to(device), coords.to(device)
        optimizer.zero_grad()
        logits, coords_pred = model(x)
        loss, _, _ = model.compute_loss(logits, coords_pred, labels, coords)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(x)
    return {"loss": total_loss / n, "acc": correct / n}


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    all_logits, all_labels, all_cp, all_ct = [], [], [], []
    for x, labels, coords in loader:
        x, labels, coords = x.to(device), labels.to(device), coords.to(device)
        logits, cp = model(x)
        loss, _, _ = model.compute_loss(logits, cp, labels, coords)
        total_loss += loss.item() * len(x)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_cp.append(cp.cpu())
        all_ct.append(coords.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    cp = torch.cat(all_cp)
    ct = torch.cat(all_ct)
    m = compute_metrics(logits, labels, cp, ct)
    m["loss"] = total_loss / len(labels)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    parser.add_argument("--output",     default=CHECKPOINTS_DUAL_DIR)
    parser.add_argument("--device",     default="auto")
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(args.output, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    ) if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    train_ds = NavahiDualDataset("train")
    val_ds   = NavahiDualDataset("val")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    from collections import Counter
    import torch.nn as nn
    label_counts = Counter(s[2] for s in train_ds.samples)
    total = len(train_ds.samples)
    class_weights = torch.tensor(
        [total / (NUM_CLASSES * max(label_counts[c], 1)) for c in range(NUM_CLASSES)],
        dtype=torch.float32,
    ).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin)

    # Same model architecture, wider input
    model = NavahiClassifier(input_dim=DUAL_FEATURE_DIM).to(device)
    model.cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_ckpt = os.path.join(args.output, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch(model, train_loader, optimizer, device)
        val_m   = eval_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_m['loss']:.4f} train_acc={train_m['acc']:.4f}  "
            f"val_loss={val_m['loss']:.4f} val_acc={val_m['acc']:.4f} "
            f"val_top2={val_m['top2_acc']:.4f} val_f1={val_m['f1']:.4f}"
        )

        if val_m["acc"] > best_val_acc:
            best_val_acc = val_m["acc"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_m,
                "input_dim": DUAL_FEATURE_DIM,
            }, best_ckpt)
            print(f"  --> saved best (val_acc={best_val_acc:.4f})")

    print(f"\nDone. Best val acc: {best_val_acc:.4f}")
    print_metrics(val_m, header="Final val metrics")


if __name__ == "__main__":
    main()
