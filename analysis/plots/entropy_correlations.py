# analysis/plots/entropy_correlations.py  (Py3.9 compatible)
import argparse, pathlib as P
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

# --------- FASTA I/O ----------
def read_fasta(path):
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s: continue
            if s.startswith(">"):
                if cur: seqs.append("".join(cur).upper())
                cur = []
            else:
                cur.append(s)
    if cur: seqs.append("".join(cur).upper())
    return seqs

# --------- entropy ----------
def shannon_entropy(seq):
    if not isinstance(seq, str) or not seq:
        return np.nan
    s = seq.upper()
    a = s.count("A"); c = s.count("C"); g = s.count("G"); t = s.count("T")
    n = a + c + g + t
    if n == 0:
        return np.nan
    p = np.array([a, c, g, t], dtype=float) / n
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def ensure_sequences(df: pd.DataFrame, who: str, fasta_path: Optional[str]):
    """
    Ensure df has a 'seq' column filled with strings.
    If missing/NaN, try to populate from FASTA (row order).
    """
    df = df.copy()
    src = None
    if "seq" in df.columns and df["seq"].notna().any():
        src = "seq"
    elif "sequence" in df.columns and df["sequence"].notna().any():
        src = "sequence"

    if src:
        df["seq"] = df[src].astype(str)
        # treat 'nan'/'None' strings as missing
        bad_mask = df["seq"].str.lower().isin(["nan", "none"])
        if bad_mask.all():
            src = None
        else:
            return df

    # fallback: use FASTA
    if fasta_path:
        try:
            fs = read_fasta(fasta_path)
            n = min(len(df), len(fs))
            if n > 0:
                df.loc[:n-1, "seq"] = fs[:n]
                if len(df) != len(fs):
                    print(f"[note] {who}: rows({len(df)}) != FASTA({len(fs)}); filled first {n} rows.")
                return df
            else:
                print(f"[warn] {who}: FASTA {fasta_path} is empty.")
        except Exception as e:
            print(f"[warn] {who}: FASTA read failed ({e}); cannot fill sequences.")

    raise ValueError(f"{who}: no usable sequences found (no 'seq'/'sequence' and no FASTA fallback).")

def ensure_entropy(df: pd.DataFrame, who: str) -> pd.DataFrame:
    df = df.copy()
    if "entropy" not in df.columns:
        df["entropy"] = df["seq"].map(shannon_entropy)
    n_nan = int(pd.isna(df["entropy"]).sum())
    if n_nan:
        print(f"[warn] {who}: {n_nan} rows have NaN entropy (non-ACGT or missing sequence).")
    return df

# --------- plotting ----------
def savefig(path: P.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def scatter_with_fit(df: pd.DataFrame, xcol: str, ycol: str, title: str, outpath: P.Path):
    d = df[[xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
    x = d[xcol].to_numpy()
    y = d[ycol].to_numpy()
    n = x.size
    if n < 3:
        print(f"[skip] not enough points for {outpath.name} (n={n})")
        return None

    try:
        from scipy.stats import pearsonr, spearmanr
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
    except Exception:
        pr = float(np.corrcoef(x, y)[0, 1])
        rx = pd.Series(x).rank(method="average").to_numpy()
        ry = pd.Series(y).rank(method="average").to_numpy()
        sr = float(np.corrcoef(rx, ry)[0, 1])
        pp = np.nan; sp = np.nan

    m, b = np.polyfit(x, y, deg=1)
    xs = np.linspace(x.min(), x.max(), 200)
    ys = m * xs + b

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(x, y, s=18, alpha=0.7, label=f"n={n}")
    ax.plot(xs, ys, lw=2, alpha=0.9, label=f"fit y={m:.3g}x+{b:.3g}")
    ax.set_xlabel(xcol); ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.legend(frameon=False)
    savefig(outpath)

    return dict(n=n, pearson=pr, pearson_p=pp, spearman=sr, spearman_p=sp,
                slope=m, intercept=b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai_csv",    required=True)
    ap.add_argument("--human_csv", required=True)
    ap.add_argument("--outdir",    required=True)
    # optional FASTA fallbacks
    ap.add_argument("--ai_fasta", default=None)
    ap.add_argument("--human_fasta", default=None)
    # column names in your CSVs
    ap.add_argument("--game_col",   default="score_norm")
    ap.add_argument("--design_col", default="design_model")
    args = ap.parse_args()

    outdir = P.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    ai = pd.read_csv(args.ai_csv)
    hu = pd.read_csv(args.human_csv)

    ai = ensure_sequences(ai, "AI", args.ai_fasta)
    hu = ensure_sequences(hu, "Human", args.human_fasta)

    ai = ensure_entropy(ai, "AI").dropna(subset=["entropy"])
    hu = ensure_entropy(hu, "Human").dropna(subset=["entropy"])

    rows = []
    r = scatter_with_fit(ai, "entropy", args.game_col,
                         "AI: entropy vs game score",
                         outdir / "AI_entropy_vs_game.png")
    if r: rows.append(dict(dataset="AI", x="entropy", y=args.game_col, **r))

    r = scatter_with_fit(ai, "entropy", args.design_col,
                         "AI: entropy vs design score",
                         outdir / "AI_entropy_vs_design.png")
    if r: rows.append(dict(dataset="AI", x="entropy", y=args.design_col, **r))

    r = scatter_with_fit(hu, "entropy", args.game_col,
                         "Human: entropy vs game score",
                         outdir / "Human_entropy_vs_game.png")
    if r: rows.append(dict(dataset="Human", x="entropy", y=args.game_col, **r))

    r = scatter_with_fit(hu, "entropy", args.design_col,
                         "Human: entropy vs design score",
                         outdir / "Human_entropy_vs_design.png")
    if r: rows.append(dict(dataset="Human", x="entropy", y=args.design_col, **r))

    if rows:
        pd.DataFrame(rows).to_csv(outdir / "entropy_correlation_summary.csv", index=False)
        print(f"[ok] wrote {outdir/'entropy_correlation_summary.csv'}")
    else:
        print("[warn] no plots produced.")

if __name__ == "__main__":
    main()
