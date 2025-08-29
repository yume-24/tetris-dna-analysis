
# just combines everything. you can just run this
# analysis/correlate_scores.py
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CANON = {
    "dataset": ["dataset", "set", "source"],
    "raw_score": ["raw_score", "raw", "game_score", "tetris_raw", "raw_board_score"],
    "normalized_score": ["normalized_score", "normalized", "norm", "game_score_norm", "normalized_board_score"],
    "design_score": ["design_score", "design", "bio_score", "objective", "target_score"],
    "pparg_score": ["pparg_score", "pparg", "ppar", "pparγ", "pparg"],
    "nfkb_score": ["nfkb_score", "nfkb", "nf-κb", "nfkb1", "nfkB1"],
}

NICE = {
    "raw_score": "Raw Tetris Score",
    "normalized_score": "Normalized Tetris Score",
    "design_score": "Biological Design Score",
    "pparg_score": "PPARγ motif score",
    "nfkb_score": "NF-κB motif score",
}

PAIRS = [
    ("raw_score", "design_score"),
    ("normalized_score", "design_score"),
    ("pparg_score", "design_score"),
    ("nfkb_score", "design_score"),
]

def coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in df.columns}
    rename = {}
    for canon, aliases in CANON.items():
        for a in aliases:
            if a.lower() in cols_lower:
                rename[cols_lower[a.lower()]] = canon
                break
    df = df.rename(columns=rename)
    return df

def safe_corr(df: pd.DataFrame, x: str, y: str):
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(sub)
    if n < 3:
        return n, np.nan, np.nan
    pearson = sub[x].corr(sub[y], method="pearson")
    spearman = sub[x].corr(sub[y], method="spearman")
    return n, float(pearson), float(spearman)

def scatter_with_trend(df: pd.DataFrame, x: str, y: str, title: str, outpath: Path):
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(6, 5), dpi=140)
    plt.scatter(sub[x], sub[y], s=12, alpha=0.5)
    # simple linear trendline
    if len(sub) >= 3:
        m, b = np.polyfit(sub[x], sub[y], 1)
        xs = np.linspace(sub[x].min(), sub[x].max(), 100)
        plt.plot(xs, m * xs + b, linewidth=1)
    plt.xlabel(NICE.get(x, x))
    plt.ylabel(NICE.get(y, y))
    plt.title(title)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/analysis/per_seq_scores.csv")
    ap.add_argument("--outdir", default="outputs/analysis")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = coerce_columns(df)

    required = ["dataset", "raw_score", "normalized_score", "design_score"]
    for r in required:
        if r not in df.columns:
            raise SystemExit(f"Missing required column '{r}' in {args.csv}. Found: {list(df.columns)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = list(sorted(df["dataset"].unique()))
    results = []

    # Compute for each dataset and for ALL combined
    for subset_name, subdf in [("ALL", df)] + [(d, df[df["dataset"] == d]) for d in datasets]:
        for x, y in PAIRS:
            if x not in subdf.columns or y not in subdf.columns:
                continue
            n, pearson, spearman = safe_corr(subdf, x, y)
            results.append({
                "dataset": subset_name,
                "x": x,
                "y": y,
                "n": n,
                "pearson": pearson,
                "spearman": spearman,
            })
            # Plot
            title = f"{subset_name}: {NICE.get(y,y)} vs {NICE.get(x,x)}\n"
            title += f"Pearson={pearson:.3f}  Spearman={spearman:.3f}  (n={n})" if n >= 3 else f"(n={n})"
            png_name = f"{subset_name}_{y}_vs_{x}.png".replace(" ", "_")
            scatter_with_trend(subdf, x, y, title, outdir / png_name)

    resdf = pd.DataFrame(results)
    resdf.to_csv(outdir / "score_correlations.csv", index=False)
    print(f"[OK] wrote correlations to {outdir/'score_correlations.csv'}")
    print(f"[OK] wrote {len([p for p in os.listdir(outdir) if p.endswith('.png')])} plot(s) to {outdir}")

if __name__ == "__main__":
    main()
