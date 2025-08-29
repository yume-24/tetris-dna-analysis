#!/usr/bin/env python3
import argparse, os, itertools
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ALPH = "ACGT"

def all_kmers(k): return ["".join(t) for t in itertools.product(ALPH, repeat=k)]
def kmer_index(k):
    km = all_kmers(k);
    return {kmer:i for i,kmer in enumerate(km)}, km

def seq_kmer_vec(s, k, idx):
    v = np.zeros(len(idx), dtype=float); valid = 0
    U = set(ALPH); s = s.upper()
    for i in range(len(s)-k+1):
        sub = s[i:i+k]
        if not set(sub).issubset(U): continue
        v[idx[sub]] += 1; valid += 1
    if valid>0: v /= valid
    return v

def stack_kmers(seqs, k, idx):
    if not seqs: return np.zeros((0, len(idx)))
    return np.vstack([seq_kmer_vec(s, k, idx) for s in seqs])

def read_fasta(path):
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s: continue
            if s.startswith(">"):
                if cur: seqs.append("".join(cur).upper()); cur=[]
            else:
                cur.append(s)
    if cur: seqs.append("".join(cur).upper())
    return seqs

def pca_2d(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    U,S,VT = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :2] * S[:2]

def tsne_2d(X):
    try:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, init="random", learning_rate="auto", perplexity=30).fit_transform(X)
    except Exception as e:
        print("[WARN] t-SNE skipped:", e); return None

def scatter_2groups(Z, labels, title, out_png):
    plt.figure(figsize=(6,5))
    for lab in sorted(set(labels)):
        m = [i for i,l in enumerate(labels) if l==lab]
        plt.scatter(Z[m,0], Z[m,1], s=12, alpha=0.7, label=f"{lab} (n={len(m)})")
    plt.title(title); plt.legend(markerscale=2, frameon=False)
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()
    print("[OK] wrote", out_png)

def load_seqs_from_csv(csv_path, seq_col_candidates=("seq","sequence","dna","dna_seq")):
    df = pd.read_csv(csv_path)
    for c in seq_col_candidates:
        if c in df.columns:
            return df[c].astype(str).str.upper().tolist(), df
    raise SystemExit(f"[ERR] No sequence column found in {csv_path}. "
                     f"Looked for {seq_col_candidates}")

def main():
    ap = argparse.ArgumentParser()
    # Preferred input: CSVs created by correlate_game_vs_dna.py
    ap.add_argument("--ai_csv",    default="outputs/analysis/per_seq_test.csv")
    ap.add_argument("--human_csv", default="outputs/analysis/per_seq_human.csv")
    # Fallback: FASTAs (used only if CSVs missing)
    ap.add_argument("--ai_fasta",    default="outputs/raw/seqs_main_ai.fasta")
    ap.add_argument("--human_fasta", default="outputs/raw/seqs_main_human.fasta")
    # Filtering
    ap.add_argument("--metric", default="score_norm",
                    help="Human CSV column to threshold (e.g. score_norm, score_raw, DESIGN_MODEL)")
    ap.add_argument("--min", type=float, default=0.9,
                    help="Minimum value for --metric")
    # Featurization/outputs
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--outdir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # --- AI sequences ---
    if os.path.exists(args.ai_csv):
        ai_seqs, ai_df = load_seqs_from_csv(args.ai_csv)
        print(f"[load] AI from CSV: {len(ai_seqs)} sequences")
    else:
        ai_seqs = read_fasta(args.ai_fasta)
        print(f"[load] AI from FASTA: {len(ai_seqs)} sequences")

    # --- Human sequences + filter ---
    if os.path.exists(args.human_csv):
        hu_seqs_all, hu_df = load_seqs_from_csv(args.human_csv)
        if args.metric not in hu_df.columns:
            raise SystemExit(f"[ERR] '{args.metric}' not found in {args.human_csv}. "
                             f"Available: {list(hu_df.columns)}")
        m = hu_df[args.metric].astype(float) >= float(args.min)
        hu_seqs = hu_df.loc[m, hu_df.columns[0]].index  # indices of passing rows
        hu_seqs = hu_df.loc[m, hu_df.columns.intersection(['seq','sequence','dna','dna_seq'])[0]].astype(str).str.upper().tolist()
        print(f"[filter] Human CSV: {args.metric} >= {args.min} → {len(hu_seqs)} sequences")
        if len(hu_seqs) == 0:
            print("[WARN] No Human sequences pass the threshold. "
                  "Try a lower --min (e.g., 0.88 for score_norm).")
            return
    else:
        # Fallback to FASTA without filtering (no scores available)
        hu_seqs = read_fasta(args.human_fasta)
        print(f"[load] Human from FASTA (no filter): {len(hu_seqs)} sequences")

    # --- Build features & plot ---
    idx,_ = kmer_index(args.k)
    X_ai = stack_kmers(ai_seqs, args.k, idx)
    X_hu = stack_kmers(hu_seqs, args.k, idx)
    if X_ai.shape[0]==0 or X_hu.shape[0]==0:
        raise SystemExit("[ERR] One group is empty after filtering/selection.")

    X = np.vstack([X_ai, X_hu])
    labels = (["AI"]*len(X_ai)) + (["Human(≥thr)"]*len(X_hu))

    Zp = pca_2d(X)
    scatter_2groups(Zp, labels,
                    f"PCA (k={args.k}) AI (n={len(X_ai)}) vs Human≥thr (n={len(X_hu)})",
                    os.path.join(args.outdir, f"pca_k{args.k}_AI_vs_HumanFiltered.png"))

    Zt = tsne_2d(X)
    if Zt is not None:
        scatter_2groups(Zt, labels,
                        f"t-SNE (k={args.k}) AI (n={len(X_ai)}) vs Human≥thr (n={len(X_hu)})",
                        os.path.join(args.outdir, f"tsne_k{args.k}_AI_vs_HumanFiltered.png"))

if __name__ == "__main__":
    main()
