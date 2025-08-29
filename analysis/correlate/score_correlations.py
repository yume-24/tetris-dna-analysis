# analysis/score_correlations.py
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_csv", default="outputs/analysis/per_seq_scores.csv",
                    help="CSV with per-sequence metrics (design + biological scores)")
    ap.add_argument("--outdir", default="outputs/analysis")
    ap.add_argument("--design_col", default="design",
                    help="column name for the Tetris/design score")
    ap.add_argument("--plot_top", type=int, default=3,
                    help="number of strongest correlations to plot (overall)")
    args = ap.parse_args()

    if not os.path.exists(args.scores_csv):
        raise SystemExit(f"Missing {args.scores_csv}. Generate it with describe_fastas.py first.")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.scores_csv)
    if args.design_col not in df.columns:
        # try a few common alternatives
        for alt in ["design_score","tetris","score","design_raw"]:
            if alt in df.columns:
                args.design_col = alt
                break
        else:
            raise SystemExit(f"Could not find a design score column in {args.scores_csv}")

    # pick biological metric columns (numeric, not design, not identifiers)
    ignore = {"dataset","seq_id","id","length","L","A","C","G","T"}
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    bio_cols = [c for c in num_cols if c not in ignore and c != args.design_col]
    if not bio_cols:
        raise SystemExit("No biological metric columns detected.")
    print("[INFO] biological metrics:", bio_cols)

    rows = []
    for scope, sub in [("ALL", df)] + [(d, g) for d, g in df.groupby("dataset")]:
        for m in bio_cols:
            x = sub[args.design_col].astype(float)
            y = sub[m].astype(float)
            n = len(sub.dropna(subset=[args.design_col, m]))
            pear = x.corr(y, method="pearson")
            spear = x.corr(y, method="spearman")
            rows.append({"scope": scope, "metric": m, "n": n,
                         "pearson": pear, "spearman": spear})
    out = pd.DataFrame(rows).sort_values(["scope","metric"])
    out_csv = os.path.join(args.outdir, "design_bio_correlations.csv")
    out.to_csv(out_csv, index=False)
    print("[OK] wrote", out_csv)

    # pick strongest absolute correlations overall and plot scatters
    top = (out[out["scope"]=="ALL"]
           .assign(absP=lambda d: d["pearson"].abs())
           .sort_values("absP", ascending=False)
           .head(args.plot_top))

    for _, r in top.iterrows():
        m = r["metric"]
        sub = df[[args.design_col, m, "dataset"]].dropna()
        plt.figure(figsize=(6,5))
        for name, g in sub.groupby("dataset"):
            plt.scatter(g[args.design_col], g[m], s=12, alpha=0.5, label=name)
        # fit simple least squares line on ALL
        x = sub[args.design_col].values
        y = sub[m].values
        if len(sub) >= 2:
            a, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ys = a*xs + b
            plt.plot(xs, ys)
        plt.xlabel(args.design_col)
        plt.ylabel(m)
        plt.title(f"{m} vs {args.design_col}\nPearson={r['pearson']:.3f}  Spearman={r['spearman']:.3f}")
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(args.outdir, f"scatter_{m}_vs_{args.design_col}.png")
        plt.savefig(out_png, dpi=180)
        plt.close()
        print("[OK] wrote", out_png)

if __name__ == "__main__":
    main()
