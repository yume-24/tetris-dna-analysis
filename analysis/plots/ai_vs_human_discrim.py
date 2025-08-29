# analysis/plots/ai_vs_human_discrim.py
import argparse, csv, os, math, itertools, pathlib as p
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ALPH = "ACGT"
IDX = {c:i for i,c in enumerate(ALPH)}

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

def all_kmers(k):
    return ["".join(t) for t in itertools.product(ALPH, repeat=k)]

def kmer_index(k):
    km = all_kmers(k)
    return {kmer:i for i,kmer in enumerate(km)}, km

def seq_kmer_vec(s, k, idx):
    v = np.zeros(len(idx), dtype=float)
    valid = 0
    for i in range(len(s)-k+1):
        sub = s[i:i+k]
        if any(ch not in ALPH for ch in sub):  # ignore Ns etc
            continue
        v[idx[sub]] += 1.0
        valid += 1
    if valid > 0:
        v /= valid
    return v

def pca_np(X, n=2):
    """Center then plain SVD; return first n PCs."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :n] * S[:n]
    return Z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai_fasta", required=True)
    ap.add_argument("--human_fasta", required=True)
    ap.add_argument("--human_scores", required=True, help="CSV with metrics; if seq_id missing, assumes same order as FASTA")
    ap.add_argument("--metric", default="score_norm")
    ap.add_argument("--min", type=float, default=0.88)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tsne", action="store_true")
    ap.add_argument("--outdir", default="outputs/analysis/ai_vs_human")
    args = ap.parse_args()

    out = p.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    S_ai = read_fasta(args.ai_fasta)
    S_hu = read_fasta(args.human_fasta)
    print(f"[load] AI={len(S_ai)}  Human(all)={len(S_hu)}")

    # Load Human scores table
    dfh = pd.read_csv(args.human_scores)
    # If mixed datasets, keep Human only when possible
    if "dataset" in dfh.columns:
        m = dfh["dataset"].astype(str).str.upper().str.contains("HUMAN")
        if m.any():
            dfh = dfh[m].reset_index(drop=True)

    if "seq_id" not in dfh.columns:
        print("[note] 'seq_id' missing; assuming score rows align with Human FASTA order.")
        dfh = dfh.copy()
        dfh["seq_id"] = np.arange(1, len(dfh)+1)

    if args.metric not in dfh.columns:
        raise SystemExit(f"metric '{args.metric}' not found in {args.human_scores}. Available: {list(dfh.columns)}")

    # Align lengths defensively
    if len(dfh) != len(S_hu):
        n = min(len(dfh), len(S_hu))
        print(f"[warn] human_scores rows ({len(dfh)}) != Human FASTA ({len(S_hu)}); truncating to {n}.")
        dfh = dfh.iloc[:n].copy()
        S_hu = S_hu[:n]

    keep_ids = set(dfh.loc[dfh[args.metric] >= args.min, "seq_id"].astype(int).tolist())
    S_hu_f = [s for i,s in enumerate(S_hu, start=1) if i in keep_ids]
    print(f"[filter] metric='{args.metric}' >= {args.min}  → Human(filtered)={len(S_hu_f)}")
    if len(S_hu_f) == 0:
        raise SystemExit("[ERR] No Human sequences pass the threshold.")

    # Build k-mer features
    idx, _ = kmer_index(args.k)
    Xa = np.vstack([seq_kmer_vec(s, args.k, idx) for s in S_ai])
    Xh = np.vstack([seq_kmer_vec(s, args.k, idx) for s in S_hu_f])

    # === PCA (NumPy SVD)
    X = np.vstack([Xa, Xh])
    Z = pca_np(X, n=2)
    Za, Zh = Z[:len(Xa)], Z[len(Xa):]

    plt.figure(figsize=(8,7))
    plt.scatter(Za[:,0], Za[:,1], s=28, alpha=0.8, label=f"AI (n={len(Za)})")
    plt.scatter(Zh[:,0], Zh[:,1], s=28, alpha=0.8, label=f"Human(≥thr) (n={len(Zh)})")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"PCA (k={args.k}) AI (n={len(Za)}) vs Human≥thr (n={len(Zh)})")
    plt.legend(frameon=False)
    plt.tight_layout()
    p_pca = out / f"pca_k{args.k}_AI_vs_HumanFiltered.png"
    plt.savefig(p_pca, dpi=220); plt.close()
    print(f"[OK] {p_pca}")

    # === t-SNE (optional)
    if args.tsne:
        try:
            from sklearn.manifold import TSNE
            perp = min(30, max(2, len(X)-1))
            Zt = TSNE(n_components=2, init="random", learning_rate="auto",
                      perplexity=perp, random_state=0).fit_transform(X)
            Zta, Zth = Zt[:len(Xa)], Zt[len(Xa):]
            plt.figure(figsize=(8,7))
            plt.scatter(Zta[:,0], Zta[:,1], s=28, alpha=0.8, label=f"AI (n={len(Zta)})")
            plt.scatter(Zth[:,0], Zth[:,1], s=28, alpha=0.8, label=f"Human(≥thr) (n={len(Zth)})")
            plt.title(f"t-SNE (k={args.k}) AI vs Human≥thr")
            plt.legend(frameon=False)
            plt.tight_layout()
            p_tsne = out / f"tsne_k{args.k}_AI_vs_HumanFiltered.png"
            plt.savefig(p_tsne, dpi=220); plt.close()
            print(f"[OK] {p_tsne}")
        except Exception as e:
            print("[warn] t-SNE skipped:", e)

    # Optional: quick linear probe (1-NN) to quantify separability
    try:
        from sklearn.neighbors import KNeighborsClassifier
        Y = np.array([0]*len(Xa) + [1]*len(Xh))
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, Y)
        acc = clf.score(X, Y)
        print(f"[probe] 5-NN training accuracy (rough separability hint): {acc:.3f}")
    except Exception as e:
        pass

if __name__ == "__main__":
    main()
