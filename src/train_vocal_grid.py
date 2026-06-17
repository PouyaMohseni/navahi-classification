"""
Train one vocal-only (wav2vec2-xlsr-53) classifier config and save checkpoint.
Used by slurm/train_vocal_grid.sh (SLURM array job 0-8).

Grid: 3 lr × 3 layer_sets = 9 configs
  wav2vec2-xlsr-53 has 25 layers (0=CNN, 1-24=transformer).
  low=[6-8]   ~25% depth
  mid=[12-14] ~50% depth
  high=[19-21] ~75% depth

Usage (standalone):
    python src/train_vocal_grid.py --run_idx 3 --output_dir $SCRATCH/navahi-vocal-gs

After all vocal models train, run late fusion:
    python src/late_fusion_eval.py \
        --mert_ckpt $SCRATCH/navahi-checkpoints-v5/best_model.pt \
        --vocal_dir $SCRATCH/navahi-vocal-gs \
        --split test_simplified
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BATCH_SIZE, NUM_EPOCHS, SEED, EVAL_WINDOW_SIZE
from dataset_vocal import NavahiVocalDataset
from model import NavahiClassifier
from evaluate import compute_metrics

LRATES     = [1e-4, 5e-5, 2e-5]
LAYER_SETS = [
    {"name": "low",  "idx": [6,  7,  8]},
    {"name": "mid",  "idx": [12, 13, 14]},
    {"name": "high", "idx": [19, 20, 21]},
]
LAMBDA     = 0.5
CLS_WEIGHT = 2.5

GRID = [
    {"lr": lr, "lambda_reg": LAMBDA,
     "layer_name": ls["name"], "layers": ls["idx"]}
    for lr in LRATES for ls in LAYER_SETS
]  # 9 entries, indices 0-8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(run_idx, output_dir, epochs, stack_size):
    params  = GRID[run_idx]
    run_dir = os.path.join(output_dir, f"run_{run_idx:02d}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"Vocal run {run_idx:02d}/{len(GRID)-1}  "
          f"lr={params['lr']:.0e}  λ={params['lambda_reg']}  "
          f"layers={params['layer_name']} {params['layers']}")
    print(f"{'='*55}")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = NavahiVocalDataset("train", stack_size=stack_size, overlap=False,
                                   vocal_indices=params["layers"])
    val_ds   = NavahiVocalDataset("val",   stack_size=stack_size, overlap=True,
                                   vocal_indices=params["layers"])
    if len(train_ds) == 0:
        print("ERROR: no training samples — check NAVAHI_FEATURES_DUAL_DIR.")
        return None

    print(f"Train: {len(train_ds)} windows  Val: {len(val_ds)} windows")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    input_dim = train_ds[0][0].shape[0]   # 3 * 1024 * 12 = 36864
    print(f"Input dim: {input_dim}")
    torch.manual_seed(SEED)
    model = NavahiClassifier(input_dim=input_dim,
                             lambda_reg=params["lambda_reg"],
                             cls_weight=CLS_WEIGHT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=0)

    best_val_acc, best_ckpt = 0.0, os.path.join(run_dir, "best_model.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        correct = n = 0
        for x, labels, coords, _ in train_loader:
            x, labels, coords = x.to(device), labels.to(device), coords.to(device)
            optimizer.zero_grad()
            logits, cp = model(x)
            loss, _, _ = model.compute_loss(logits, cp, labels, coords)
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == labels).sum().item()
            n += len(x)

        model.eval()
        vl, vla, vcp, vct = [], [], [], []
        with torch.no_grad():
            for x, labels, coords, _ in val_loader:
                x, labels, coords = x.to(device), labels.to(device), coords.to(device)
                logits, cp = model(x)
                vl.append(logits.cpu()); vla.append(labels.cpu())
                vcp.append(cp.cpu());   vct.append(coords.cpu())
        val_m = compute_metrics(torch.cat(vl), torch.cat(vla),
                                torch.cat(vcp), torch.cat(vct))
        print(f"  Epoch {epoch:02d}/{epochs}  "
              f"train_acc={correct/n:.4f}  val_acc={val_m['acc']:.4f}")

        if val_m["acc"] > best_val_acc:
            best_val_acc = val_m["acc"]
            torch.save({"model_state":  model.state_dict(),
                        "params":       params,
                        "input_dim":    input_dim,
                        "val_metrics":  val_m}, best_ckpt)
            print(f"    --> saved (val_acc={best_val_acc:.4f})")

    print(f"\nBest val acc: {best_val_acc:.4f}")
    result = {"run_idx": run_idx, "params": params, "best_val_acc": best_val_acc,
              "input_dim": input_dim}
    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {json_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_idx",    type=int, required=True,
                        help=f"Config index 0–{len(GRID)-1}")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs",     type=int, default=NUM_EPOCHS)
    parser.add_argument("--stack_size", type=int, default=EVAL_WINDOW_SIZE)
    args = parser.parse_args()
    if args.run_idx >= len(GRID):
        print(f"ERROR: run_idx {args.run_idx} >= grid size {len(GRID)}")
        sys.exit(1)
    print(f"Grid size: {len(GRID)} vocal configs")
    run(args.run_idx, args.output_dir, args.epochs, args.stack_size)


if __name__ == "__main__":
    main()
