# analysis/describe_fastas.py
# Compare multiple FASTA sets: composition, entropy, k-mer, PCA/t-SNE, motif scores (PPARγ, NF-κB, design)
import sklearn




"""
RUN THIS(at the end)
python analysis/describe_fastas.py \
  --input Random=outputs/raw/seqs_random.fasta \
  --input AI=outputs/raw/seqs_main_ai.fasta \
  --input Human=outputs/raw/seqs_main_human.fasta \
  --k 4 \
  --outdir outputs/describe_run \
  --tsne

(random comes from make_random_fasta.py)

OUTPUTS
composition_bars.png : AT/GC fractions for each dataset
entropy_boxplot.png : per sequence entropy
kmer4_heatmap.png : 4-mer freqs
pca_k4.png  also tsne is enabled
per_seq_scores.csv : entropy and motif scores for each sequence
pparg_boxplot.png
nfkb_boxplot.png
design_boxplot.png
design_hist.png
dataset_summary.csv : simple table with n mean len, mean entropy, gc %
"""

import argparse, os, math, itertools, collections, sys, pathlib as p
import numpy as np
import matplotlib.pyplot as plt

# ---------- FASTA I/O ----------
def read_fasta(path):
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if cur: seqs.append("".join(cur).upper())
                cur = []
            else:
                cur.append(line)
    if cur: seqs.append("".join(cur).upper())
    return seqs

# ---------- Composition & Entropy ----------
ALPH = "ACGT"
IDX = {c:i for i,c in enumerate(ALPH)}

def seq_counts(s):
    c = np.zeros(4, dtype=np.int64)
    for ch in s:
        i = IDX.get(ch, -1)
        if i>=0: c[i]+=1
    return c

def shannon_entropy(s):
    c = seq_counts(s).astype(float)
    n = c.sum()
    if n == 0: return np.nan
    p = c / n
    p = p[p>0]
    return float(-(p*np.log2(p)).sum())

# ---------- k-mer featurization ----------
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
        if any(ch not in ALPH for ch in sub):
            continue
        v[idx[sub]] += 1
        valid += 1
    if valid>0: v /= valid
    return v

# ---------- MEME PWM parsing & scoring ----------
def parse_meme_pwm(path):
    # Parses JASPAR-style MEME "letter-probability matrix" (rows = positions, cols = A C G T)
    rows = []
    in_mat = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("letter-probability matrix"):
                in_mat = True
                continue
            if in_mat:
                if not line or line[0].isalpha():  # reached next section
                    break
                vals = [float(x) for x in line.split()]
                if len(vals)>=4:
                    rows.append(vals[:4])
    if not rows:
        raise ValueError(f"No PWM read from {path}")
    pwm = np.array(rows, dtype=float)  # shape (w,4) ordered A C G T
    pwm = np.clip(pwm, 1e-9, 1.0)      # avoid zeros
    pwm /= pwm.sum(axis=1, keepdims=True)  # re-normalize rows
    return pwm

RC = str.maketrans("ACGT", "TGCA")
def revcomp(s): return s.translate(RC)[::-1]

def pwm_max_norm_score(seq, pwm):
    # normalized in [0,1]: max over windows of exp(sum log p(b|pos)) / exp(sum log maxcol(pos))
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

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True,
                    help="Label=path.fasta (use 3 entries: Random=..., AI=..., Human=...)")
    ap.add_argument("--k", type=int, default=4, help="k for k-mer features (default 4)")
    ap.add_argument("--outdir", default="outputs/describe_run")
    ap.add_argument("--ppar", default="motifs/MA0065.2.meme")
    ap.add_argument("--nfkb", default="motifs/MA0105.4.meme")
    ap.add_argument("--tsne", action="store_true", help="Try t-SNE if scikit-learn is available")
    args = ap.parse_args()

    # Load datasets
    datasets = []
    for spec in args.input:
        if "=" not in spec: raise SystemExit(f"--input needs Label=path, got: {spec}")
        name, path = spec.split("=",1)
        seqs = read_fasta(path)
        datasets.append((name, path, seqs))
        print(f"[load] {name}: {len(seqs)} sequences from {path}")

    out = p.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # 1) AT/GC composition
    comp_rows = []
    for name, _, seqs in datasets:
        tot = np.zeros(4, dtype=np.int64)
        for s in seqs: tot += seq_counts(s)
        n = tot.sum()
        frac = tot / n if n>0 else np.zeros(4)
        comp_rows.append((name, n, *frac.tolist()))
    # bar plot
    labels = [r[0] for r in comp_rows]
    fracs = np.array([r[2:] for r in comp_rows])  # shape (D,4)
    fig, ax = plt.subplots(figsize=(7,4))
    x = np.arange(len(labels))
    width = 0.18
    for i, base in enumerate(ALPH):
        ax.bar(x + (i-1.5)*width, fracs[:,i], width, label=base)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction"); ax.set_title("AT/GC composition")
    ax.legend()
    savefig(out/"composition_bars.png")

    # 2) Entropy per sequence
    ent = []
    for name,_,seqs in datasets:
        vals = [shannon_entropy(s) for s in seqs]
        ent.append((name, np.array(vals)))
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
    for name,_,seqs in datasets:
        tot = np.zeros(K, dtype=float)
        denom = 0
        for s in seqs:
            v = seq_kmer_vec(s, args.k, idx)
            tot += v
            denom += 1 if v.sum()>0 else 0
        if denom>0: tot /= denom
        ds_freq.append((name, tot))
    # heatmap across datasets
    mat = np.vstack([v for _,v in ds_freq])
    fig, ax = plt.subplots(figsize=(max(6, K*0.12), 2+0.4*len(ds_freq)))
    im = ax.imshow(mat, aspect='auto')
    ax.set_yticks(np.arange(len(ds_freq)))
    ax.set_yticklabels([n for n,_ in ds_freq])
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels(klist, rotation=90)
    ax.set_title(f"{args.k}-mer frequency (avg across sequences)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    savefig(out/f"kmer{args.k}_heatmap.png")

    # per-seq matrix for PCA/TSNE
    X_list, y_list, lab_list = [], [], []
    for name,_,seqs in datasets:
        for s in seqs:
            X_list.append(seq_kmer_vec(s, args.k, idx))
            y_list.append(name)
            lab_list.append(name)
    X = np.vstack(X_list) if X_list else np.zeros((0,K))
    # PCA (numpy SVD, no sklearn needed)
    if X.shape[0] >= 2:
        Xc = X - X.mean(axis=0, keepdims=True)
        U,S,VT = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]  # first 2 PCs
        fig, ax = plt.subplots(figsize=(6,5))
        for name in {n for n,_,_ in datasets}:
            m = [i for i,t in enumerate(y_list) if t==name]
            ax.scatter(Z[m,0], Z[m,1], s=10, alpha=0.6, label=name)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(f"PCA on {args.k}-mer features")
        ax.legend(markerscale=2, frameon=False)
        savefig(out/f"pca_k{args.k}.png")

    # t-SNE (optional)
    if args.tsne and X.shape[0] >= 10:
        try:
            from sklearn.manifold import TSNE
            Zt = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=30).fit_transform(X)
            fig, ax = plt.subplots(figsize=(6,5))
            for name in {n for n,_,_ in datasets}:
                m = [i for i,t in enumerate(y_list) if t==name]
                ax.scatter(Zt[m,0], Zt[m,1], s=10, alpha=0.6, label=name)
            ax.set_title(f"t-SNE on {args.k}-mer features")
            ax.legend(markerscale=2, frameon=False)
            savefig(out/f"tsne_k{args.k}.png")
        except Exception as e:
            print("[warn] t-SNE skipped:", e)

    # 4) Motif/biological scores
    motif_ok = True
    try:
        pwm_ppar = parse_meme_pwm(args.ppar)
        pwm_nfkb = parse_meme_pwm(args.nfkb)
    except Exception as e:
        motif_ok = False
        print("[warn] Motif scoring skipped:", e)

    per_seq_rows = []
    if motif_ok:
        for name,_,seqs in datasets:
            for i,s in enumerate(seqs):
                ppar = pwm_max_norm_score(s, pwm_ppar)
                nfkb = pwm_max_norm_score(s, pwm_nfkb)
                design = (ppar - nfkb + 1.0)/2.0
                entv = shannon_entropy(s)
                per_seq_rows.append((name, i+1, len(s), entv, ppar, nfkb, design))
        # save per-seq csv
        import csv
        with open(out/"per_seq_scores.csv","w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset","seq_id","length","entropy","PPARg","NFkB","design"])
            w.writerows(per_seq_rows)
        # box/violin plots
        import pandas as pd
        df = pd.DataFrame(per_seq_rows, columns=["dataset","seq_id","length","entropy","PPARg","NFkB","design"])
        for col in ["PPARg","NFkB","design"]:
            fig, ax = plt.subplots(figsize=(7,4))
            parts = [df[df.dataset==n][col].values for n,_,_ in datasets]
            ax.boxplot(parts, labels=[n for n,_,_ in datasets], showmeans=True)
            ax.set_ylabel(col); ax.set_title(f"{col} score by dataset")
            savefig(out/f"{col.lower()}_boxplot.png")
        # design histogram overlay
        fig, ax = plt.subplots(figsize=(7,4))
        for name,_,_ in datasets:
            vals = df[df.dataset==name]["design"].values
            ax.hist(vals, bins=30, histtype="step", density=True, label=name)
        ax.set_xlabel("design score"); ax.set_ylabel("density"); ax.set_title("Design score distribution")
        ax.legend()
        savefig(out/"design_hist.png")

    # 5) Write a small dataset summary
    import csv
    with open(out/"dataset_summary.csv","w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset","n_sequences","mean_len","mean_entropy","GC_fraction"])
        for (name,_,seqs),(_,totfrac) in zip(datasets, ds_freq):
            n = len(seqs)
            mean_len = np.mean([len(s) for s in seqs]) if n else 0
            mean_ent = np.nanmean([shannon_entropy(s) for s in seqs]) if n else np.nan
            # simple GC fraction from composition above
            comp = np.zeros(4, dtype=np.int64)
            for s in seqs: comp += seq_counts(s)
            gc = (comp[1]+comp[2]) / max(1, comp.sum())
            w.writerow([name, n, round(float(mean_len),2), round(float(mean_ent),3), round(float(gc),3)])

    print(f"[done] Wrote figures & CSVs to: {args.outdir}")

if __name__ == "__main__":
    main()
