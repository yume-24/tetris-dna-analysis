# analysis/kmer_enrichment.py
import argparse, os, math, itertools
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



"""
run 
python analysis/kmer_enrichment.py --k 4
or something like 
python analysis/kmer_enrichment.py --k 6 --collapse_rc

"""
def read_fasta(path):
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur: seqs.append("".join(cur).strip().upper()); cur=[]
            else:
                cur.append(line.strip())
    if cur: seqs.append("".join(cur).strip().upper())
    return seqs

_comp = str.maketrans("ACGTNacgtn","TGCANtgcan")
def rc(s): return s.translate(_comp)[::-1]

def all_kmers(k):
    return ["".join(p) for p in itertools.product("ACGT", repeat=k)]

def kmer_counts(seqs, k, collapse_rc=False):
    c = Counter()
    total = 0
    for s in seqs:
        s = "".join(ch for ch in s if ch in "ACGTacgt").upper()
        for i in range(len(s)-k+1):
            kmer = s[i:i+k]
            if collapse_rc:
                kmer = min(kmer, rc(kmer))
            c[kmer]+=1
            total+=1
    return c, total

def jsd(p, q, eps=1e-12):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p/ (p.sum()+eps); q = q/ (q.sum()+eps)
    m = 0.5*(p+q)
    def kl(a,b):
        a = np.clip(a, eps, None); b = np.clip(b, eps, None)
        return np.sum(a*np.log2(a/b))
    return 0.5*kl(p,m) + 0.5*kl(q,m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--random", default="outputs/raw/seqs_random.fasta")
    ap.add_argument("--ai",      default="outputs/raw/seqs_main_ai.fasta")
    ap.add_argument("--human",   default="outputs/raw/seqs_main_human.fasta")
    ap.add_argument("--outdir",  default="outputs/analysis")
    ap.add_argument("--collapse_rc", action="store_true",
                    help="collapse k-mer with its reverse complement")
    ap.add_argument("--top", type=int, default=20, help="top N to plot")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    datasets = {
        "Random": args.random,
        "AI": args.ai,
        "Human": args.human,
    }

    # count kmers
    kmers = all_kmers(args.k)
    if args.collapse_rc:
        # keep canonical representatives
        seen = set(); keep=[]
        for kmer in kmers:
            canon = min(kmer, rc(kmer))
            if canon not in seen:
                seen.add(canon); keep.append(canon)
        kmers = keep

    counts = {}
    totals = {}
    for name, path in datasets.items():
        seqs = read_fasta(path)
        c, tot = kmer_counts(seqs, args.k, collapse_rc=args.collapse_rc)
        counts[name] = c
        totals[name] = tot

    # build frequency table with Laplace smoothing
    alpha = 1.0
    df = pd.DataFrame(index=kmers)
    for name in datasets:
        df[f"count_{name}"] = [counts[name].get(k,0) for k in kmers]
        denom = totals[name] + alpha*len(kmers)
        df[f"freq_{name}"]  = (df[f"count_{name}"] + alpha) / denom

    # log2 fold-change vs Random
    for name in ("AI","Human"):
        df[f"log2FC_{name}_vs_Random"] = np.log2(df[f"freq_{name}"] / df["freq_Random"])

    out_csv = os.path.join(args.outdir, f"kmer_enrichment_k{args.k}{'_rc' if args.collapse_rc else ''}.csv")
    df.reset_index().rename(columns={"index":"kmer"}).to_csv(out_csv, index=False)
    print("[OK] wrote", out_csv)

    # plots: top enriched/depleted per dataset vs Random
    for name in ("AI","Human"):
        col = f"log2FC_{name}_vs_Random"
        top_enr = df[col].sort_values(ascending=False).head(args.top)
        top_dep = df[col].sort_values(ascending=True).head(args.top)
        plot_df = pd.concat([top_enr, top_dep]).sort_values()
        plt.figure(figsize=(10,5))
        plot_df.plot(kind="barh")
        plt.axvline(0, lw=1, color="k")
        plt.title(f"k={args.k} enrichment: {name} vs Random")
        plt.xlabel("log2 fold-change")
        plt.tight_layout()
        out_png = os.path.join(args.outdir, f"kmer_enrichment_{name}_k{args.k}.png")
        plt.savefig(out_png, dpi=180)
        plt.close()
        print("[OK] wrote", out_png)

    # pairwise JSD matrix on frequencies
    freqs = {name: df[f"freq_{name}"].values for name in datasets}
    names = list(datasets.keys())
    M = np.zeros((len(names), len(names)))
    for i,a in enumerate(names):
        for j,b in enumerate(names):
            M[i,j] = jsd(freqs[a], freqs[b])
    mat = pd.DataFrame(M, index=names, columns=names)
    out_mat = os.path.join(args.outdir, f"kmer_jsd_k{args.k}.csv")
    mat.to_csv(out_mat)
    print("[OK] wrote", out_mat)

    # heatmap
    plt.figure(figsize=(4,3))
    im = plt.imshow(M, cmap="viridis")
    plt.colorbar(im, label="Jensen–Shannon distance")
    plt.xticks(range(len(names)), names)
    plt.yticks(range(len(names)), names)
    plt.title(f"Pairwise JSD (k={args.k})")
    plt.tight_layout()
    out_png = os.path.join(args.outdir, f"kmer_jsd_k{args.k}.png")
    plt.savefig(out_png, dpi=180)
    plt.close()
    print("[OK] wrote", out_png)

if __name__ == "__main__":
    main()
