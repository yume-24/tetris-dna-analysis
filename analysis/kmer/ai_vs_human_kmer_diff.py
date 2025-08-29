# analysis/ai_vs_human_kmer_diff.py
import argparse, os, math
from collections import Counter
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")                 # ensure files are saved headless
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

"""run 
python analysis/ai_vs_human_kmer_diff.py --k 4

"""

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

def count_k(seqs, k):
    from collections import Counter
    c = Counter(); tot = 0
    for s in seqs:
        s = "".join(ch for ch in s if ch in "ACGT")
        for i in range(len(s)-k+1):
            c[s[i:i+k]] += 1; tot += 1
    return c, tot

def bh_fdr(p):
    p = np.asarray(p, float); n = len(p)
    order = np.argsort(p); out = np.empty(n)
    prev = 1.0
    # Benjamini–Hochberg (monotone)
    for rank, idx in enumerate(order[::-1], start=1):
        q = p[idx] * n / (n - rank + 1)
        prev = min(prev, q); out[idx] = prev
    return out

def volcano(df, out_png, k):
    x = df["log2fc"].values
    y = -np.log10(np.maximum(df["q"].values, 1e-300))
    plt.figure(figsize=(7,5))
    plt.scatter(x, y, s=14, alpha=0.6)
    sig = df["q"].values < 0.05
    if sig.any():
        plt.scatter(x[sig], y[sig], s=18)
    plt.axvline(0, lw=1, color="k")
    plt.xlabel(f"log2(AI / Human) for k={k}")
    plt.ylabel("-log10(FDR q)")
    plt.title(f"Human vs AI k={k} differential (Fisher)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def heatmap_top(df, out_png, topn=50):
    # Use .iloc to avoid label-slice weirdness
    top = df.reindex(df["log2fc"].abs().sort_values(ascending=False).index).iloc[:topn].copy()
    if top.empty:
        print("[WARN] top set empty; skipping heatmap.")
        return
    M = np.vstack([top["freq_ai"].values, top["freq_human"].values]).T  # (rows=kmer, cols=2)
    kmers = top["kmer"].tolist()
    plt.figure(figsize=(9, max(4, topn*0.14)))
    plt.imshow(M, aspect="auto")
    plt.yticks(range(len(kmers)), kmers, fontsize=7)
    plt.xticks([0,1], ["AI","Human"])
    plt.colorbar(label="frequency")
    plt.title(f"Top differential k-mers (|log2FC|, n={len(kmers)})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--ai", default="outputs/raw/seqs_main_ai.fasta")
    ap.add_argument("--human", default="outputs/raw/seqs_main_human.fasta")
    ap.add_argument("--outdir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ai_seqs = read_fasta(args.ai)
    hu_seqs = read_fasta(args.human)

    c_ai, A = count_k(ai_seqs, args.k)
    c_hu, H = count_k(hu_seqs, args.k)

    kmers = sorted(set(c_ai) | set(c_hu))
    rows = []
    for kmer in kmers:
        a = c_ai.get(kmer,0); h = c_hu.get(kmer,0)
        _, p = fisher_exact([[a, A-a],[h, H-h]], alternative="two-sided")
        fa = (a+0.5)/(A+1.0)
        fh = (h+0.5)/(H+1.0)
        rows.append(dict(kmer=kmer, a=a, A=A, h=h, H=H,
                         freq_ai=fa, freq_human=fh,
                         log2fc=np.log2(fa/fh), p=p))
    df = pd.DataFrame(rows)
    df["q"] = bh_fdr(df["p"].values)

    out_csv = os.path.join(args.outdir, f"ai_human_kmer_diff_k{args.k}.csv")
    df.to_csv(out_csv, index=False)

    # tops (use k in filenames)
    ai_up = df.sort_values("log2fc", ascending=False).head(25)
    hu_up = df.sort_values("log2fc", ascending=True).head(25)
    ai_csv = os.path.join(args.outdir, f"ai_human_k{args.k}_top25_AI.csv")
    hu_csv = os.path.join(args.outdir, f"ai_human_k{args.k}_top25_Human.csv")
    ai_up.to_csv(ai_csv, index=False)
    hu_up.to_csv(hu_csv, index=False)

    volcano_png = os.path.join(args.outdir, f"ai_vs_human_k{args.k}_volcano.png")
    heatmap_png = os.path.join(args.outdir, f"ai_human_k{args.k}_top50_heatmap.png")
    volcano(df, volcano_png, args.k)
    heatmap_top(df, heatmap_png, topn=50)

    print("[OK] Wrote:")
    for p in (out_csv, ai_csv, hu_csv, volcano_png, heatmap_png):
        print(" ", p, "(exists:", os.path.exists(p), ")")

if __name__ == "__main__":
    main()
