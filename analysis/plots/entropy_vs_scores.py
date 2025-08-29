# analysis/plots/entropy_vs_scores.py
import os, math, argparse
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ----------------- entropy helpers -----------------
ALPH = "ACGT"
IDX = {c:i for i,c in enumerate(ALPH)}

def shannon_entropy_from_probs(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    return float(-(p * np.log2(p)).sum())

def entropy_from_sequence(seq: str) -> float:
    c = np.zeros(4, dtype=float)
    for ch in (seq or ""):
        i = IDX.get(ch.upper(), -1)
        if i >= 0: c[i] += 1
    n = c.sum()
    if n == 0:
        return np.nan
    p = c / n
    return shannon_entropy_from_probs(p)

def read_fasta(path: str) -> List[str]:
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur: seqs.append("".join(cur).upper())
                cur = []
            else:
                cur.append(line)
    if cur: seqs.append("".join(cur).upper())
    return seqs

def ensure_entropy(df: pd.DataFrame, who: str, fasta_path: Optional[str]) -> pd.DataFrame:
    """Ensure df has an 'entropy' column. Try sequence, then A/C/G/T fracs, then FASTA+seq_id."""
    if "entropy" in df.columns and df["entropy"].notna().any():
        return df

    # 1) Have raw sequence?
    seq_col = None
    for cand in ["sequence", "seq", "dna", "Sequence"]:
        if cand in df.columns:
            seq_col = cand; break
    if seq_col is not None:
        df["entropy"] = df[seq_col].astype(str).apply(entropy_from_sequence)
        return df

    # 2) Have A/C/G/T fractions?
    frac_cols = ["A_frac", "C_frac", "G_frac", "T_frac"]
    if all(c in df.columns for c in frac_cols):
        P = df[frac_cols].to_numpy(dtype=float)
        # ensure rows sum to 1
        row_sum = P.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        P = P / row_sum
        ents = []
        for row in P:
            ents.append(shannon_entropy_from_probs(row))
        df["entropy"] = ents
        return df

    # 3) If FASTA + seq_id exists, compute from FASTA sequence
    if fasta_path and os.path.exists(fasta_path) and "seq_id" in df.columns:
        seqs = read_fasta(fasta_path)
        # seq_id is 1-based in our pipeline
        def ent_from_id(x):
            try:
                i = int(x) - 1
                if 0 <= i < len(seqs):
                    return entropy_from_sequence(seqs[i])
            except Exception:
                pass
            return np.nan
        df["entropy"] = df["seq_id"].apply(ent_from_id)
        return df

    raise ValueError(f"{who}: missing 'entropy' and no usable sequence/A,C,G,T or FASTA+seq_id to derive it.")

def ensure_design(df: pd.DataFrame) -> pd.DataFrame:
    if "design_score" in df.columns:
        return df
    if "pparg_score" in df.columns and "nfkb_score" in df.columns:
        df["design_score"] = (df["pparg_score"].astype(float) - df["nfkb_score"].astype(float) + 1.0) / 2.0
        return df
    return df  # nothing to do

def find_game_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["game_score", "score_norm", "score"]:
        if c in df.columns:
            return c
    return None

# ----------------- plotting -----------------
def scatter_with_fit(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    title: str,
    outpath: str
) -> Tuple[float, float, int]:
    d = df[[xcol, ycol]].dropna()
    if d.empty:
        return np.nan, np.nan, 0
    x = d[xcol].to_numpy(dtype=float)
    y = d[ycol].to_numpy(dtype=float)

    # correlations
    pr = pearsonr(x, y)[0]
    sr = spearmanr(x, y)[0]

    # simple fit line
    try:
        b, a = np.polyfit(x, y, 1)
        yhat = a + b * x
    except Exception:
        a = y.mean(); b = 0.0; yhat = np.full_like(x, a)

    plt.figure(figsize=(7,6))
    plt.scatter(x, y, s=25, alpha=0.8)
    # plot line across full span
    xs = np.linspace(x.min(), x.max(), 100)
    plt.plot(xs, a + b*xs)
    plt.xlabel(xcol); plt.ylabel(ycol); plt.title(title)
    txt = f"Pearson r = {pr:+.3f}\nSpearman ρ = {sr:+.3f}\n n = {len(d)}"
    plt.gca().text(0.03, 0.97, txt, transform=plt.gca().transAxes,
                   va="top", ha="left",
                   bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6"))
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout(); plt.savefig(outpath, dpi=220); plt.close()
    return pr, sr, len(d)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai_csv", required=True)
    ap.add_argument("--human_csv", required=True)
    ap.add_argument("--ai_fasta", default=None)
    ap.add_argument("--human_fasta", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    ai = pd.read_csv(args.ai_csv)
    hu = pd.read_csv(args.human_csv)

    # Ensure entropy/design exist
    ai = ensure_entropy(ai, "AI", args.ai_fasta)
    hu = ensure_entropy(hu, "Human", args.human_fasta)
    ai = ensure_design(ai)
    hu = ensure_design(hu)

    # Figure out game score columns (can differ)
    ai_game = find_game_col(ai)
    hu_game = find_game_col(hu)

    out = args.outdir
    os.makedirs(out, exist_ok=True)

    # Entropy vs Design
    if "design_score" in ai.columns:
        scatter_with_fit(ai, "entropy", "design_score",
                         "AI: Entropy vs Design score",
                         os.path.join(out, "AI_entropy_vs_design.png"))
    if "design_score" in hu.columns:
        scatter_with_fit(hu, "entropy", "design_score",
                         "Human: Entropy vs Design score",
                         os.path.join(out, "Human_entropy_vs_design.png"))

    # Entropy vs Game score
    if ai_game:
        scatter_with_fit(ai, "entropy", ai_game,
                         "AI: Entropy vs Game score",
                         os.path.join(out, "AI_entropy_vs_game.png"))
    else:
        print("[warn] No game-score column found in AI CSV.")
    if hu_game:
        scatter_with_fit(hu, "entropy", hu_game,
                         "Human: Entropy vs Game score",
                         os.path.join(out, "Human_entropy_vs_game.png"))
    else:
        print("[warn] No game-score column found in Human CSV.")

    # Save a summary CSV of correlations
    rows = []
    def corr_row(name: str, df: pd.DataFrame, y: str) -> Tuple[float,float,int]:
        d = df[["entropy", y]].dropna()
        if d.empty: return (np.nan, np.nan, 0)
        x = d["entropy"].to_numpy(float)
        yy = d[y].to_numpy(float)
        return (pearsonr(x, yy)[0], spearmanr(x, yy)[0], len(d))

    if "design_score" in ai.columns:
        r, s, n = corr_row("AI", ai, "design_score")
        rows.append(["AI", "design_score", r, s, n])
    if "design_score" in hu.columns:
        r, s, n = corr_row("Human", hu, "design_score")
        rows.append(["Human", "design_score", r, s, n])
    if ai_game:
        r, s, n = corr_row("AI", ai, ai_game)
        rows.append(["AI", ai_game, r, s, n])
    if hu_game:
        r, s, n = corr_row("Human", hu, hu_game)
        rows.append(["Human", hu_game, r, s, n])

    summ = pd.DataFrame(rows, columns=["dataset","target","pearson_r","spearman_r","n"])
    summ.to_csv(os.path.join(out, "entropy_correlation_summary.csv"), index=False)
    print(summ)

if __name__ == "__main__":
    main()
