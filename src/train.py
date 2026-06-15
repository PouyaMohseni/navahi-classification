"""
Train the multi-task NavahiClassifier on pre-extracted MERT features.

Usage:
    python src/train.py [--epochs 10] [--batch_size 32] [--lr 2e-5] [--output checkpoints/]
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, SEED
from dataset import NavahiDataset
from model import NavahiClassifier
from evaluate import compute_metrics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = cls_loss_sum = reg_loss_sum = 0
    correct = n = 0

    for x, labels, coords in loader:
        x = x.to(device)
        labels = labels.to(device)
        coords = coords.to(device)

        optimizer.zero_grad()
        logits, coords_pred = model(x)
        loss, l_cls, l_reg = model.compute_loss(logits, coords_pred, labels, coords)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x)
        cls_loss_sum += l_cls * len(x)
        reg_loss_sum += l_reg * len(x)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(x)

    return {
        "loss": total_loss / n,
        "cls_loss": cls_loss_sum / n,
        "reg_loss": reg_loss_sum / n,
        "acc": correct / n,
    }


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_coords_pred, all_coords_true = [], [], [], []

    for x, labels, coords in loader:
        x = x.to(device)
        labels = labels.to(device)
        coords = coords.to(device)

        logits, coords_pred = model(x)
        loss, _, _ = model.compute_loss(logits, coords_pred, labels, coords)
        total_loss += loss.item() * len(x)

        all_preds.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_coords_pred.append(coords_pred.cpu())
        all_coords_true.append(coords.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_coords_pred = torch.cat(all_coords_pred)
    all_coords_true = torch.cat(all_coords_true)

    metrics = compute_metrics(all_preds, all_labels, all_coords_pred, all_coords_true)
    metrics["loss"] = total_loss / len(all_labels)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "checkpoints"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(args.output, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    ) if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    train_ds = NavahiDataset("train")
    val_ds = NavahiDataset("val")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin)

    model = NavahiClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_ckpt = os.path.join(args.output, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch(model, train_loader, optimizer, device)
        val_m = eval_epoch(model, val_loader, device)

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
            }, best_ckpt)
            print(f"  --> saved best model (val_acc={best_val_acc:.4f})")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
