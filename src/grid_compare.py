"""
Print a sorted comparison table of all grid-search results.
Run after all grid_train_eval.py jobs complete.

Usage:
    python src/grid_compare.py --results_dir $SCRATCH/navahi-gs-dual
    python src/grid_compare.py --results_dir $SCRATCH/navahi-gs-dual --sort_by test_mv
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--sort_by", default="ts_mv",
                        choices=["ts_mv", "test_mv", "ts_w", "test_w", "val"],
                        help="ts=test_simplified, w=window, mv=majority_vote")
    args = parser.parse_args()

    rows = []
    missing = []
    for d in sorted(os.listdir(args.results_dir)):
        path = os.path.join(args.results_dir, d, "results.json")
        if not os.path.isdir(os.path.join(args.results_dir, d)):
            continue
        if not os.path.exists(path):
            missing.append(d)
            continue
        with open(path) as f:
            r = json.load(f)
        p = r["params"]
        rows.append({
            "run":     d,
            "lr":      p["lr"],
            "lam":     p["lambda_reg"],
            "layers":  p["layer_name"],
            "val":     r["best_val_acc"],
            "test_w":  r.get("test", {}).get("window_acc", 0),
            "test_mv": r.get("test", {}).get("mv_acc",     0),
            "ts_w":    r.get("test_simplified", {}).get("window_acc", 0),
            "ts_mv":   r.get("test_simplified", {}).get("mv_acc",     0),
        })

    if not rows:
        print("No results found.")
        return

    rows.sort(key=lambda x: x[args.sort_by], reverse=True)

    sep = "─"
    hdr = (f"{'Run':<8} {'LR':>7} {'λ':>4} {'Layers':>6}"
           f"  {'Val':>6} │ {'Test-W':>7} {'Test-MV':>8}"
           f" │ {'TS-W':>7} {'TS-MV':>8}")
    print(f"\n{hdr}")
    print(sep * len(hdr))
    for r in rows:
        print(f"{r['run']:<8} {r['lr']:>7.0e} {r['lam']:>4.1f} {r['layers']:>6}"
              f"  {r['val']:>6.4f} │ {r['test_w']:>7.4f} {r['test_mv']:>8.4f}"
              f" │ {r['ts_w']:>7.4f} {r['ts_mv']:>8.4f}")

    b = rows[0]
    print(f"\nBest ({args.sort_by}): {b['run']}")
    print(f"  lr={b['lr']:.0e}  λ={b['lam']}  layers={b['layers']}")
    print(f"  Val:             {b['val']:.4f}")
    print(f"  Test:            window={b['test_w']:.4f}  majority_vote={b['test_mv']:.4f}")
    print(f"  Test_simplified: window={b['ts_w']:.4f}  majority_vote={b['ts_mv']:.4f}")

    print(f"\nRuns completed: {len(rows)}/27"
          + (f"  Missing: {', '.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
