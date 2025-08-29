import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"Could not find any of {candidates} in columns: {df.columns.tolist()}")

def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if np.std(x)==0 or np.std(y)==0: return np.nan
    return np.corrcoef(x, y)[0,1]

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    def rank(a):
        order = np.argsort(a)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(a))
        # average ties
        vals, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        avg = sums / counts
        return avg[inv]
    rx, ry = rank(x), rank(y)
    if np.std(rx)==0 or np.std(ry)==0: return np.nan
    return np.corrcoef(rx, ry)[0,1]

def make_plot(df, dataset_name, ycol, outdir):
    sub = df.dropna(subset=["score_norm", ycol]).copy()
    if sub.empty:
        print(f"[warn] no rows for {dataset_name}")
        return
    x = sub["score_norm"].values
    y = sub[ycol].values
    rP = pearson(x, y)
    rS = spearman(x, y)

    plt.figure(figsize=(6,5))
    plt.scatter(x, y, s=12, alpha=0.6)
    # regression line (least squares)
    if len(sub) >= 2 and np.std(x) > 0:
        a, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = a*xs + b
        plt.plot(xs, ys, linewidth=1)
    plt.xlabel("Normalized game score (score_norm)")
    plt.ylabel("Model internal design score")
    plt.title(f"{dataset_name}: score_norm vs DESIGN_MODEL\nPearson={rP:.3f}  Spearman={rS:.3f}")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    outp = Path(outdir, f"scatter_norm_vs_MODEL_{dataset_name}.png")
    plt.tight_layout(); plt.savefig(outp, dpi=180); plt.close()
    print("[OK] wrote", outp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_test",  default="outputs/analysis/per_seq_test.csv")
    ap.add_argument("--per_human", default="outputs/analysis/per_seq_human.csv")
    ap.add_argument("--outdir",    default="outputs/analysis")
    args = ap.parse_args()

    # load
    dfT = pd.read_csv(args.per_test)
    dfH = pd.read_csv(args.per_human)

    # auto-detect the model-design column name
    ycolT = pick_col(dfT, ["design_model", "DESIGN_MODEL"])
    ycolH = pick_col(dfH, ["design_model", "DESIGN_MODEL"])
    # (normally they’ll match; if not, we allow separate picks)

    make_plot(dfT, "TEST_AI", ycolT, args.outdir)
    make_plot(dfH, "HUMAN",   ycolH, args.outdir)

if __name__ == "__main__":
    main()
