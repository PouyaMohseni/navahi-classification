"""
Late fusion evaluation: combine MERT (single-stream) + vocal-only model predictions.
Tries 16 fusion strategies across every vocal checkpoint in --vocal_dir.

Strategies:
  01  mert_only          — MERT baseline (no vocal)
  02  vocal_only         — vocal baseline (no MERT)
  03  logit_50m_50v      — average logits equally
  04-09                  — weighted logit average (30/40/60/70/80/90% MERT)
  10  softmax_avg        — average softmax probabilities
  11  softmax_prod       — product of softmax probabilities
  12  max_conf           — per window, pick the more confident model
  13  vote_equal         — window-level: both models cast equal votes
  14  vote_2mert_1vocal  — MERT votes count 2×
  15  vote_1mert_2vocal  — vocal votes count 2×
  16  borda_count        — rank classes by per-file mean logit, sum ranks

Usage:
    python src/late_fusion_eval.py \
        --mert_ckpt  $SCRATCH/navahi-checkpoints-v5/best_model.pt \
        --vocal_dir  $SCRATCH/navahi-vocal-gs \
        --split      test_simplified
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (balanced_accuracy_score,
                              precision_score, recall_score, f1_score)
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BATCH_SIZE, EVAL_WINDOW_SIZE, FEATURE_DIM, CLASS_NAMES, NUM_CLASSES
from dataset import NavahiDataset
from dataset_vocal import NavahiVocalDataset
from model import NavahiClassifier


def _infer(model, ds, device):
    """Run inference. Returns (logits N×8, labels N, file_idx N)."""
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    ll, la, lf = [], [], []
    with torch.no_grad():
        for x, labels, _, fidx in loader:
            logits, _ = model(x.to(device))
            ll.append(logits.cpu()); la.append(labels); lf.append(fidx)
    return torch.cat(ll), torch.cat(la).numpy(), torch.cat(lf).numpy()


def _mv(preds, labels, fidx):
    """Per-file majority vote. Returns (accuracy, n_files)."""
    fp, fl = [], []
    for fid in np.unique(fidx):
        m = fidx == fid
        fp.append(np.bincount(preds[m], minlength=8).argmax())
        fl.append(labels[m][0])
    return float(np.mean(np.array(fp) == np.array(fl))), len(fl)


def fuse(ml, vl, labels, fidx):
    """
    Apply all 16 fusion strategies.
    ml / vl: MERT / vocal logits (N, 8) as torch tensors.
    Returns {strategy_name: (mv_acc, n_files)}.
    """
    mn = ml.numpy()
    vn = vl.numpy()
    mp = F.softmax(ml, dim=1).numpy()   # MERT probs
    vp = F.softmax(vl, dim=1).numpy()   # vocal probs
    mp_preds = mn.argmax(1)              # window-level MERT predictions
    vp_preds = vn.argmax(1)              # window-level vocal predictions

    out = {}

    # ── Baselines ────────────────────────────────────────────────────────────
    out["01_mert_only"]  = _mv(mp_preds, labels, fidx)
    out["02_vocal_only"] = _mv(vp_preds, labels, fidx)

    # ── Logit-level weighted average ─────────────────────────────────────────
    for alpha, tag in [
        (0.5, "03_logit_50m_50v"),
        (0.3, "04_logit_30m_70v"),
        (0.4, "05_logit_40m_60v"),
        (0.6, "06_logit_60m_40v"),
        (0.7, "07_logit_70m_30v"),
        (0.8, "08_logit_80m_20v"),
        (0.9, "09_logit_90m_10v"),
    ]:
        out[tag] = _mv((alpha * mn + (1 - alpha) * vn).argmax(1), labels, fidx)

    # ── Probability-level fusion ──────────────────────────────────────────────
    out["10_softmax_avg"]  = _mv((mp + vp).argmax(1),        labels, fidx)
    out["11_softmax_prod"] = _mv((mp * vp).argmax(1),        labels, fidx)

    # ── Max confidence per window ─────────────────────────────────────────────
    use_mert = mp.max(1) >= vp.max(1)
    chosen   = np.where(use_mert, mp_preds, vp_preds)
    out["12_max_conf"] = _mv(chosen, labels, fidx)

    # ── Window-level vote fusion ──────────────────────────────────────────────
    # Both models cast window votes; treat each window prediction as a ballot.
    for label_tag, m_reps, v_reps in [
        ("13_vote_equal",        1, 1),
        ("14_vote_2mert_1vocal", 2, 1),
        ("15_vote_1mert_2vocal", 1, 2),
    ]:
        comb_preds  = np.concatenate([mp_preds] * m_reps + [vp_preds] * v_reps)
        comb_fidx   = np.concatenate([fidx]     * (m_reps + v_reps))
        comb_labels = np.concatenate([labels]   * (m_reps + v_reps))
        out[label_tag] = _mv(comb_preds, comb_labels, comb_fidx)

    # ── Borda count (rank-based, per file) ────────────────────────────────────
    # For each file: rank classes by mean logit from each model, sum ranks.
    fp, fl = [], []
    for fid in np.unique(fidx):
        m = fidx == fid
        m_rank = np.argsort(np.argsort(mn[m].mean(0)))   # rank 0=worst, 7=best
        v_rank = np.argsort(np.argsort(vn[m].mean(0)))
        fp.append((m_rank + v_rank).argmax())
        fl.append(labels[m][0])
    out["16_borda_count"] = (float(np.mean(np.array(fp) == np.array(fl))), len(fl))

    return out


def _fuse_one(strategy, ml, vl, labels, fidx):
    """Apply a single strategy. Returns (file_preds, file_labels)."""
    mn, vn = ml.numpy(), vl.numpy()
    mp = F.softmax(ml, dim=1).numpy()
    vp = F.softmax(vl, dim=1).numpy()
    mp_preds = mn.argmax(1)
    vp_preds = vn.argmax(1)

    if strategy == "01_mert_only":
        w_p, w_f, w_l = mp_preds, fidx, labels
    elif strategy == "02_vocal_only":
        w_p, w_f, w_l = vp_preds, fidx, labels
    elif "_logit_" in strategy:
        alpha = int(strategy.split("_")[2][:-1]) / 100   # "70m" → 0.7
        w_p, w_f, w_l = (alpha*mn + (1-alpha)*vn).argmax(1), fidx, labels
    elif strategy == "10_softmax_avg":
        w_p, w_f, w_l = (mp + vp).argmax(1), fidx, labels
    elif strategy == "11_softmax_prod":
        w_p, w_f, w_l = (mp * vp).argmax(1), fidx, labels
    elif strategy == "12_max_conf":
        w_p = np.where(mp.max(1) >= vp.max(1), mp_preds, vp_preds)
        w_f, w_l = fidx, labels
    elif strategy == "13_vote_equal":
        w_p = np.concatenate([mp_preds, vp_preds])
        w_f = np.concatenate([fidx,     fidx])
        w_l = np.concatenate([labels,   labels])
    elif strategy == "14_vote_2mert_1vocal":
        w_p = np.concatenate([mp_preds, mp_preds, vp_preds])
        w_f = np.concatenate([fidx,     fidx,     fidx])
        w_l = np.concatenate([labels,   labels,   labels])
    elif strategy == "15_vote_1mert_2vocal":
        w_p = np.concatenate([mp_preds, vp_preds, vp_preds])
        w_f = np.concatenate([fidx,     fidx,     fidx])
        w_l = np.concatenate([labels,   labels,   labels])
    elif strategy == "16_borda_count":
        fp, fl = [], []
        for fid in np.unique(fidx):
            m = fidx == fid
            mr = np.argsort(np.argsort(mn[m].mean(0)))
            vr = np.argsort(np.argsort(vn[m].mean(0)))
            fp.append((mr + vr).argmax()); fl.append(labels[m][0])
        return np.array(fp), np.array(fl)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    fp, fl = [], []
    for fid in np.unique(w_f):
        m = w_f == fid
        fp.append(np.bincount(w_p[m], minlength=8).argmax())
        fl.append(w_l[m][0])
    return np.array(fp), np.array(fl)


def _detailed_report(file_preds, file_labels, split, strategy, vocal_run):
    acc  = (file_preds == file_labels).mean()
    bal  = balanced_accuracy_score(file_labels, file_preds)
    prec = precision_score(file_labels, file_preds, average="macro", zero_division=0)
    rec  = recall_score(file_labels,  file_preds, average="macro", zero_division=0)
    f1   = f1_score(file_labels,      file_preds, average="macro", zero_division=0)

    # Top-2 not available at file level, note that
    print(f"\n{'='*60}")
    print(f"Detailed report — {split}  |  {vocal_run}  |  {strategy}")
    print(f"{'='*60}")
    print(f"  Files:          {len(file_labels)}")
    print(f"  Accuracy:       {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Balanced Acc:   {bal:.4f}  ({bal*100:.2f}%)")
    print(f"  Precision:      {prec:.4f}  (macro)")
    print(f"  Recall:         {rec:.4f}  (macro)")
    print(f"  F1:             {f1:.4f}  (macro)")
    print(f"\n  Per-class accuracy (majority vote):")
    for c in range(NUM_CLASSES):
        mask = file_labels == c
        if mask.sum() == 0:
            continue
        cls_acc = (file_preds[mask] == c).mean()
        print(f"    {CLASS_NAMES[c]:<35} {cls_acc*100:5.1f}%  (n={mask.sum()})")
    print(f"{'='*60}")


def load_model(ckpt_path, device, default_input_dim):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = NavahiClassifier(input_dim=ckpt.get("input_dim", default_input_dim)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mert_ckpt",  required=True,
                        help="MERT single-stream checkpoint (e.g. navahi-checkpoints-v5/best_model.pt)")
    parser.add_argument("--vocal_dir",  required=True,
                        help="Directory with vocal run_XX/ subdirs from train_vocal_grid.py")
    parser.add_argument("--split",      default="test_simplified",
                        choices=["train", "val", "test", "test_simplified"])
    parser.add_argument("--stack_size", type=int, default=EVAL_WINDOW_SIZE)
    parser.add_argument("--top",        type=int, default=20,
                        help="Number of top rows to print")
    parser.add_argument("--detail",     action="store_true",
                        help="Print full metrics + per-class breakdown for the best result")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Split: {args.split}")

    # ── MERT inference ────────────────────────────────────────────────────────
    print(f"\nLoading MERT: {args.mert_ckpt}")
    mert_model, _ = load_model(args.mert_ckpt, device, FEATURE_DIM)
    mert_ds = NavahiDataset(args.split, stack_size=args.stack_size, overlap=True)
    print(f"MERT dataset: {len(mert_ds)} windows")
    mert_logits, mert_labels, mert_fidx = _infer(mert_model, mert_ds, device)
    print(f"MERT inference done  ({len(np.unique(mert_fidx))} files)")

    # ── Find vocal checkpoints ────────────────────────────────────────────────
    vocal_runs = sorted(
        d for d in os.listdir(args.vocal_dir)
        if os.path.isdir(os.path.join(args.vocal_dir, d))
        and os.path.exists(os.path.join(args.vocal_dir, d, "best_model.pt"))
    )
    print(f"\nFound {len(vocal_runs)} vocal checkpoints")

    all_rows = []
    vocal_logits_cache = {}   # {run_name: (logits_tensor, layers_list)}

    for run_name in vocal_runs:
        run_dir   = os.path.join(args.vocal_dir, run_name)
        ckpt_path = os.path.join(run_dir, "best_model.pt")
        json_path = os.path.join(run_dir, "results.json")

        params = {}
        if os.path.exists(json_path):
            with open(json_path) as f:
                params = json.load(f).get("params", {})

        vocal_layers = params.get("layers", [6, 7, 8])
        if isinstance(vocal_layers, str):
            vocal_layers = json.loads(vocal_layers)

        vocal_model, vckpt = load_model(ckpt_path, device, 36864)
        vocal_ds = NavahiVocalDataset(args.split, stack_size=args.stack_size,
                                       overlap=True, vocal_indices=vocal_layers)
        vocal_logits, vocal_labels, vocal_fidx = _infer(vocal_model, vocal_ds, device)

        # Alignment check
        if len(vocal_logits) != len(mert_logits):
            print(f"  SKIP {run_name}: window count mismatch "
                  f"({len(vocal_logits)} vs {len(mert_logits)})")
            continue
        if not np.array_equal(mert_labels, vocal_labels):
            print(f"  SKIP {run_name}: label mismatch")
            continue

        vocal_logits_cache[run_name] = (vocal_logits, vocal_layers)
        strats = fuse(mert_logits, vocal_logits, mert_labels, mert_fidx)

        lr  = params.get("lr",         "?")
        lam = params.get("lambda_reg", "?")
        lay = params.get("layer_name", "?")
        for strat, (acc, n_files) in strats.items():
            all_rows.append({
                "vocal_run": run_name, "lr": lr, "lambda": lam,
                "layers": lay, "strategy": strat,
                "mv_acc": acc, "n_files": n_files,
            })
        print(f"  {run_name} done  ({len(np.unique(vocal_fidx))} files)")

    if not all_rows:
        print("No results.")
        return

    all_rows.sort(key=lambda r: r["mv_acc"], reverse=True)

    # ── Print top-N table ─────────────────────────────────────────────────────
    hdr = (f"{'Vocal':>8} {'LR':>7} {'Layers':>6}  "
           f"{'Strategy':<25} {'MV Acc':>9}")
    print(f"\nTop {args.top} results (split={args.split}):")
    print(hdr)
    print("─" * len(hdr))
    for r in all_rows[:args.top]:
        print(f"{r['vocal_run']:>8} {str(r['lr']):>7} {r['layers']:>6}  "
              f"{r['strategy']:<25} {r['mv_acc']:>8.4f}  ({r['mv_acc']*100:.2f}%)")

    # ── Summary ───────────────────────────────────────────────────────────────
    best = all_rows[0]
    # Find MERT-only baseline from the results
    mert_rows = [r for r in all_rows if r["strategy"] == "01_mert_only"]
    mert_base = mert_rows[0]["mv_acc"] if mert_rows else None

    print(f"\n{'='*55}")
    print(f"BEST: {best['vocal_run']}  +  {best['strategy']}")
    print(f"  Vocal config: lr={best['lr']}  λ={best['lambda']}  layers={best['layers']}")
    print(f"  MV Accuracy: {best['mv_acc']*100:.2f}%  (n={best['n_files']} files)")
    if mert_base is not None:
        delta = best["mv_acc"] - mert_base
        sign  = "+" if delta >= 0 else ""
        print(f"  vs MERT-only: {mert_base*100:.2f}%  (Δ {sign}{delta*100:.2f}%)")
    print(f"{'='*55}")

    if args.detail and best["vocal_run"] in vocal_logits_cache:
        best_vl, _ = vocal_logits_cache[best["vocal_run"]]
        fp, fl = _fuse_one(best["strategy"], mert_logits, best_vl,
                           mert_labels, mert_fidx)
        _detailed_report(fp, fl, args.split, best["strategy"], best["vocal_run"])


if __name__ == "__main__":
    main()
