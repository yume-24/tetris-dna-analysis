# analysis/describe/describe_ai_human.py
"""
Describe two FASTAs (AI & Human) with external MEME PWM scoring (PPARγ::RXRA, NF-κB)
and k-mer/entropy/composition plots.

Example:
  python analysis/describe/describe_ai_human.py \
  --ai outputs/raw/seqs_main_ai.fasta \
  --human outputs/raw/seqs_main_human.fasta \
  --match_equal_n \
  --outdir outputs/describe_two_sets_matched

"""

import argparse, math, itertools, pathlib as p
import numpy as np
import matplotlib.pyplot as plt

# ---------- FASTA I/O ----------
ALPH = "ACGT"
IDX = {c:i for i,c in enumerate(ALPH)}

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

# ---------- Composition & Entropy ----------
# counts A/C/G/T with fixed 4 long vec. ignores non acgt characters
def seq_counts(s):
    c = np.zeros(4, dtype=np.int64)
    for ch in s:
        i = IDX.get(ch, -1)
        if i>=0: c[i]+=1
    return c

#converts counts -> fractions for a,c,g,t and GC
def base_fracs(s):
    c = seq_counts(s).astype(float)
    n = c.sum() if c.sum()>0 else 1.0
    a, c_, g, t = c / n
    return float(a), float(c_), float(g), float(t), float((c_+g))

#shannon entropy over a/c/g/t
def shannon_entropy(s):
    c = seq_counts(s).astype(float)
    n = c.sum()
    if n == 0: return np.nan
    p = c / n
    p = p[p>0]
    return float(-(p*np.log2(p)).sum())

# ---------- k-mer featurization ----------
#enumerates every k-mer over a/c/g/t
def all_kmers(k):
    return ["".join(t) for t in itertools.product(ALPH, repeat=k)]

#builds {k-mer -> column index} + ordered list
def kmer_index(k):
    km = all_kmers(k)
    return {kmer:i for i,kmer in enumerate(km)}, km

#slides window. normalizes by num of valid windows. produces freq vector for later use (pca, tsne)
def seq_kmer_vec(s, k, idx):
    v = np.zeros(len(idx), dtype=float)
    valid = 0
    for i in range(len(s)-k+1):
        sub = s[i:i+k]
        if any(ch not in ALPH for ch in sub):
            continue
        v[idx[sub]] += 1
        valid += 1
    if valid>0: v /= valid
    return v

# ---------- MEME PWM parsing & scoring (external, max-pooled & normalized) ----------
#parses MEME, collects rows (a,c,g,t),clips tiny probs, row-normalizes to probailities
def parse_meme_pwm(path):
    rows = []
    in_mat = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("letter-probability matrix"):
                in_mat = True
                continue
            if in_mat:
                if not line or line[0].isalpha():
                    break
                vals = [float(x) for x in line.split()]
                if len(vals)>=4:
                    rows.append(vals[:4])
    if not rows:
        raise ValueError(f"No PWM read from {path}")
    pwm = np.array(rows, dtype=float)  # (w,4) ordered A C G T
    pwm = np.clip(pwm, 1e-9, 1.0)
    pwm /= pwm.sum(axis=1, keepdims=True)
    return pwm

RC = str.maketrans("ACGT", "TGCA")
def revcomp(s): return s.translate(RC)[::-1]


def pwm_max_norm_score(seq, pwm):
    L, w = len(seq), pwm.shape[0]
    if L < w: return 0.0
    maxcol = np.max(pwm, axis=1)
    max_log = np.log(maxcol).sum()
    def score_one_strand(s):
        best = -1e30
        for i in range(L-w+1):
            sub = s[i:i+w]
            ok = True
            lp = 0.0
            for j,ch in enumerate(sub):
                a = IDX.get(ch, -1)
                if a < 0: ok=False; break
                lp += math.log(pwm[j,a])
            if ok and lp > best: best = lp
        if best <= -1e29: return 0.0
        return math.exp(best - max_log)
    return max(score_one_strand(seq), score_one_strand(revcomp(seq)))

# ---------- Plot helpers ----------
def savefig(path):
    p.Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match_equal_n", action="store_true",
                    help="Downsample the larger set to match the smaller")
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--ai", required=True, help="AI fasta")
    ap.add_argument("--human", required=True, help="Human fasta")
    ap.add_argument("--k", type=int, default=4, help="k for k-mer features (default 4)")
    ap.add_argument("--outdir", default="outputs/describe_two_sets")
    ap.add_argument("--ppar", default="motifs/MA0065.2.meme")
    ap.add_argument("--nfkb", default="motifs/MA0105.4.meme")
    ap.add_argument("--tsne", action="store_true", help="Try t-SNE if scikit-learn is available")
    args = ap.parse_args()

    ds = [
        ("AI", args.ai, read_fasta(args.ai)),
        ("Human", args.human, read_fasta(args.human)),
    ]
    if args.match_equal_n:
        import random
        random.seed(args.seed)
        n_ai = len(ds[0][2]);
        n_hu = len(ds[1][2])
        target = min(n_ai, n_hu)
        for i, (name, path, seqs) in enumerate(ds):
            if len(seqs) > target:
                idx = list(range(len(seqs)))
                random.shuffle(idx)
                idx = sorted(idx[:target])
                ds[i] = (name, path, [seqs[j] for j in idx])

    for name, path, seqs in ds:
        print(f"[load] {name}: {len(seqs)} sequences from {path}")

    out = p.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    n_map = {name: len(seqs) for name, _, seqs in ds}
    def with_n(name): return f"{name} (n={n_map[name]})"

    # 1) AT/GC composition
    labels = []; fracs = []
    for name, _, seqs in ds:
        tot = np.zeros(4, dtype=np.int64)
        for s in seqs: tot += seq_counts(s)
        n = tot.sum()
        frac = tot / n if n>0 else np.zeros(4)
        labels.append(with_n(name)); fracs.append(frac)
    fracs = np.vstack(fracs)
    fig, ax = plt.subplots(figsize=(7,4))
    x = np.arange(len(labels)); width = 0.18
    for i, base in enumerate(ALPH):
        ax.bar(x + (i-1.5)*width, fracs[:,i], width, label=base)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction"); ax.set_title("AT/GC composition")
    ax.legend()
    savefig(out/"composition_bars.png")

    # 2) Entropy per sequence
    ent = []
    for name,_,seqs in ds:
        vals = [shannon_entropy(s) for s in seqs]
        ent.append((with_n(name), np.array(vals)))
    fig, ax = plt.subplots(figsize=(7,4))
    ax.boxplot([v for _,v in ent], labels=[n for n,_ in ent], showmeans=True)
    ax.set_ylabel("Shannon entropy (bits)")
    ax.set_title("Per-sequence nucleotide entropy")
    savefig(out/"entropy_boxplot.png")

    # 3) k-mer distribution (dataset-level) and PCA/t-SNE (per sequence)
    idx, klist = kmer_index(args.k)
    K = len(idx)
    # dataset-level frequencies
    ds_freq = []
    for name,_,seqs in ds:
        tot = np.zeros(K, dtype=float); denom = 0
        for s in seqs:
            v = seq_kmer_vec(s, args.k, idx)
            tot += v
            denom += 1 if v.sum()>0 else 0
        if denom>0: tot /= denom
        ds_freq.append((name, tot))
    # heatmap across datasets
    mat = np.vstack([v for _,v in ds_freq])
    fig, ax = plt.subplots(figsize=(max(6, K*0.12), 2+0.4*len(ds)))
    im = ax.imshow(mat, aspect='auto')
    ax.set_yticks(np.arange(len(ds_freq)))
    ax.set_yticklabels([with_n(n) for n,_ in ds_freq])
    ax.set_xticks(np.arange(K)); ax.set_xticklabels(klist, rotation=90)
    ax.set_title(f"{args.k}-mer frequency (avg across sequences)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    savefig(out/f"kmer{args.k}_heatmap.png")

    # per-seq matrix for PCA/TSNE
    X_list, y_list = [], []
    for name,_,seqs in ds:
        for s in seqs:
            X_list.append(seq_kmer_vec(s, args.k, idx))
            y_list.append(name)
    X = np.vstack(X_list) if X_list else np.zeros((0,K))
    # PCA (numpy SVD)
    if X.shape[0] >= 2:
        Xc = X - X.mean(axis=0, keepdims=True)
        U,S,VT = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]
        fig, ax = plt.subplots(figsize=(6,5))
        for name in [n for n,_,_ in ds]:
            m = [i for i,t in enumerate(y_list) if t==name]
            ax.scatter(Z[m,0], Z[m,1], s=10, alpha=0.6, label=with_n(name))
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(f"PCA on {args.k}-mer features")
        ax.legend(markerscale=2, frameon=False)
        savefig(out/f"pca_k{args.k}.png")

    # t-SNE (optional)
    if args.tsne and X.shape[0] >= 10:
        try:
            from sklearn.manifold import TSNE
            Zt = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=30).fit_transform(X)
            fig, ax = plt.subplots(figsize=(6,5))
            for name in [n for n,_,_ in ds]:
                m = [i for i,t in enumerate(y_list) if t==name]
                ax.scatter(Zt[m,0], Zt[m,1], s=10, alpha=0.6, label=with_n(name))
            ax.set_title(f"t-SNE on {args.k}-mer features")
            ax.legend(markerscale=2, frameon=False)
            savefig(out/f"tsne_k{args.k}.png")
        except Exception as e:
            print("[warn] t-SNE skipped:", e)

    # 4) Motif/biological scores (external MEME PWMs)
    motif_ok = True
    try:
        pwm_ppar = parse_meme_pwm(args.ppar)
        pwm_nfkb = parse_meme_pwm(args.nfkb)
    except Exception as e:
        motif_ok = False
        print("[warn] Motif scoring skipped:", e)

    rows = []
    if motif_ok:
        for name,_,seqs in ds:
            for i,s in enumerate(seqs, start=1):
                a_frac, c_frac, g_frac, t_frac, gc_frac = base_fracs(s)
                entv  = shannon_entropy(s)
                ppar  = pwm_max_norm_score(s, pwm_ppar)
                nfkb  = pwm_max_norm_score(s, pwm_nfkb)
                design = (ppar - nfkb + 1.0)/2.0
                rows.append(dict(
                    dataset=name, seq_id=i, sequence=s, length=len(s),
                    entropy=entv,
                    A_frac=a_frac, C_frac=c_frac, G_frac=g_frac, T_frac=t_frac, GC_frac=gc_frac,
                    pparg_score=ppar, nfkb_score=nfkb, design_score=design
                ))
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(out/"per_seq_scores.csv", index=False)

        # boxplots
        for col, ylabel, fname in [
            ("pparg_score","PPARγ score","pparg_boxplot.png"),
            ("nfkb_score","NF-κB score","nfkb_boxplot.png"),
            ("design_score","Design score","design_boxplot.png"),
        ]:
            fig, ax = plt.subplots(figsize=(7,4))
            parts = [df[df.dataset==n][col].values for n,_,_ in ds]
            ax.boxplot(parts, labels=[with_n(n) for n,_,_ in ds], showmeans=True)
            ax.set_ylabel(ylabel); ax.set_title(f"{ylabel} by dataset")
            savefig(out/fname)

        # design histogram overlay
        fig, ax = plt.subplots(figsize=(7,4))
        for name,_,_ in ds:
            vals = df[df.dataset==name]["design_score"].values
            ax.hist(vals, bins=30, histtype="step", density=True, label=with_n(name))
        ax.set_xlabel("Design score (external PWM)"); ax.set_ylabel("density"); ax.set_title("Design score distribution")
        ax.legend()
        savefig(out/"design_hist.png")

    # 5) Write a small dataset summary
    import csv
    with open(out/"dataset_summary.csv","w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset","n_sequences","mean_len","mean_entropy","GC_fraction"])
        for (name,_,seqs) in ds:
            n = len(seqs)
            mean_len = np.mean([len(s) for s in seqs]) if n else 0
            mean_ent = np.nanmean([shannon_entropy(s) for s in seqs]) if n else np.nan
            comp = np.zeros(4, dtype=np.int64)
            for s in seqs: comp += seq_counts(s)
            gc = (comp[1]+comp[2]) / max(1, comp.sum())
            w.writerow([name, n, round(float(mean_len),2), round(float(mean_ent),3), round(float(gc),3)])

    print(f"[done] Wrote figures & CSVs to: {args.outdir}")

if __name__ == "__main__":
    main()
