"""
Train one dual-stream grid-search config and evaluate on test + test_simplified.
Run as a SLURM array job (slurm/grid_search_dual.sh), one task per config.

Grid: 3 lr × 3 lambda_reg × 3 layer_sets = 27 configs
  - lr           : paper grid search values
  - lambda_reg   : weight of regression loss (Table 3 in paper)
  - layer_sets   : which MERT / wav2vec2 hidden layers to use (mid/high/low, as in paper)

Usage (standalone):
    python src/grid_train_eval.py --run_idx 5 --output_dir $SCRATCH/navahi-gs-dual

After all runs complete:
    python src/grid_compare.py --results_dir $SCRATCH/navahi-gs-dual
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
from dataset_dual import NavahiDualDataset
from model import NavahiClassifier
from evaluate import compute_metrics

# ── Parameter grid ────────────────────────────────────────────────────────────
LRATES  = [1e-4, 5e-5, 2e-5]   # from paper's grid search
LAMBDAS = [0.0,  0.5,  1.0]    # reg loss weight (Table 3)

# Layer sets use model-proportional positions:
#   MERT:        13 layers total (0=embed, 1-12=transformer)
#   wav2vec2-xlsr-53: 25 layers total (0=CNN,  1-24=transformer)
# "low/mid/high" map to the same relative depth in each model.
LAYER_SETS = [
    {"name": "low",  "instru": [3, 4, 5],   "vocal": [6,  7,  8]},   # ~25% depth
    {"name": "mid",  "instru": [6, 7, 8],   "vocal": [12, 13, 14]},  # ~50% depth
    {"name": "high", "instru": [9, 10, 11], "vocal": [19, 20, 21]},  # ~75% depth
]
CLS_WEIGHT = 2.5                # fixed (same as single-stream best)

GRID = [
    {
        "lr": lr, "lambda_reg": lam,
        "layer_name":   ls["name"],
        "instru_layers": ls["instru"],
        "vocal_layers":  ls["vocal"],
    }
    for lr  in LRATES
    for lam in LAMBDAS
    for ls  in LAYER_SETS
]  # 27 entries, indices 0-26


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _majority_vote(logits_cat, labels_cat, fidx_np):
    preds  = logits_cat.argmax(1).numpy()
    labels = labels_cat.numpy()
    fp, fl = [], []
    for fid in np.unique(fidx_np):
        m = fidx_np == fid
        fp.append(np.bincount(preds[m]).argmax())
        fl.append(labels[m][0])
    return float(np.mean(np.array(fp) == np.array(fl))), len(fl)


def _eval_split(model, split, params, stack_size, device):
    ds = NavahiDualDataset(split, stack_size=stack_size, overlap=True,
                           instru_indices=params["instru_layers"],
                           vocal_indices=params["vocal_layers"])
    if len(ds) == 0:
        print(f"  [{split}] 0 samples — skipping")
        return None
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    all_logits, all_labels, all_cp, all_ct, all_fidx = [], [], [], [], []
    with torch.no_grad():
        for x, labels, coords, fidx in loader:
            logits, cp = model(x.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_cp.append(cp.cpu())
            all_ct.append(coords)
            all_fidx.append(fidx)
    lc = torch.cat(all_logits)
    la = torch.cat(all_labels)
    cp = torch.cat(all_cp)
    ct = torch.cat(all_ct)
    fi = torch.cat(all_fidx).numpy()
    m  = compute_metrics(lc, la, cp, ct)
    mv_acc, n_files = _majority_vote(lc, la, fi)
    return {
        "window_acc":   m["acc"],
        "mv_acc":       mv_acc,
        "balanced_acc": m["balanced_acc"],
        "top2_acc":     m["top2_acc"],
        "f1":           m["f1"],
        "n_files":      n_files,
    }


def run(run_idx, output_dir, epochs, stack_size):
    params  = GRID[run_idx]
    run_dir = os.path.join(output_dir, f"run_{run_idx:02d}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Run {run_idx:02d}/{len(GRID)-1}  "
          f"lr={params['lr']:.0e}  λ={params['lambda_reg']}  layers={params['layer_name']}")
    print(f"{'='*60}")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = NavahiDualDataset("train", stack_size=stack_size, overlap=False,
                                  instru_indices=params["instru_layers"],
                                  vocal_indices=params["vocal_layers"])
    val_ds   = NavahiDualDataset("val",   stack_size=stack_size, overlap=True,
                                  instru_indices=params["instru_layers"],
                                  vocal_indices=params["vocal_layers"])
    if len(train_ds) == 0:
        print("ERROR: no training samples — check NAVAHI_FEATURES_DUAL_DIR.")
        return None

    print(f"Train: {len(train_ds)} windows  Val: {len(val_ds)} windows")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    input_dim = train_ds[0][0].shape[0]
    torch.manual_seed(SEED)
    model = NavahiClassifier(input_dim=input_dim,
                             lambda_reg=params["lambda_reg"],
                             cls_weight=CLS_WEIGHT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=0)

    best_val_acc = 0.0
    best_ckpt    = os.path.join(run_dir, "best_model.pt")

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
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

        # ── Val ──────────────────────────────────────────────────────────────
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
            torch.save({"model_state": model.state_dict(), "params": params,
                        "input_dim": input_dim, "val_metrics": val_m}, best_ckpt)
            print(f"    --> saved (val_acc={best_val_acc:.4f})")

    # ── Evaluate on both test splits ──────────────────────────────────────────
    print(f"\nBest val acc: {best_val_acc:.4f}  — loading best checkpoint for eval...")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    params_json = {k: (str(v) if isinstance(v, list) else v)
                   for k, v in params.items()}
    result = {"run_idx": run_idx, "params": params_json, "best_val_acc": best_val_acc}

    for split in ["test", "test_simplified"]:
        r = _eval_split(model, split, params, stack_size, device)
        if r:
            result[split] = r
            print(f"  {split:<20}  window={r['window_acc']:.4f}  "
                  f"mv={r['mv_acc']:.4f}  (n={r['n_files']} files)")

    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {json_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_idx",    type=int, required=True,
                        help=f"Config index 0–{len(GRID)-1}")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save checkpoints and results JSON")
    parser.add_argument("--epochs",     type=int, default=NUM_EPOCHS)
    parser.add_argument("--stack_size", type=int, default=EVAL_WINDOW_SIZE)
    args = parser.parse_args()

    if args.run_idx >= len(GRID):
        print(f"ERROR: run_idx {args.run_idx} >= grid size {len(GRID)}")
        sys.exit(1)

    print(f"Grid size: {len(GRID)} configs")
    run(args.run_idx, args.output_dir, args.epochs, args.stack_size)


if __name__ == "__main__":
    main()
