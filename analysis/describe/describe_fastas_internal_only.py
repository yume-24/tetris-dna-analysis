# analysis/describe/describe_fastas_internal_only.py
"""
Internal-only version of describe_fastas:
- Reads fasta(s)
- Computes composition / entropy / k-mer PCA (+ optional t-SNE)
- Computes *internal/training-style* motif scores with PWMs:
    PPARγ (default motifs/MA0065.2.meme)
    NF-κB (default motifs/MA0105.4.meme)
  using log-prob + max-pooling normalization.
- Writes:
  composition_bars.png
  entropy_boxplot.png
  kmer{k}_heatmap.png, pca_k{k}.png (+ tsne_k{k}.png if --tsne)
  per_seq_scores_internal.csv  (with A/C/G/T/GC, entropy, PPARγ, NF-κB, design_internal, sequence)
  pparg_internal_boxplot.png
  nfkb_internal_boxplot.png
  design_internal_boxplot.png
  design_internal_hist.png
  dataset_summary.csv



  python analysis/describe/describe_fastas_internal_only.py \
  --input Random=outputs/raw/seqs_random.fasta \
  --input AI=outputs/raw/seqs_main_ai.fasta \
  --input Human=outputs/raw/seqs_main_human.fasta \
  --k 4 \
  --outdir outputs/describe_internal \
  --tsne



"""

import argparse, os, math, itertools, sys, pathlib as p
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

def base_fracs(s):
    c = seq_counts(s).astype(float)
    n = c.sum() if c.sum()>0 else 1.0
    a,c_,g,t = c / n
    return float(a), float(c_), float(g), float(t), float(c_+g)

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
        if any(ch not in ALPH for ch in sub): continue
        v[idx[sub]] += 1
        valid += 1
    if valid>0: v /= valid
    return v

# ---------- PWM parsing & internal scorer ----------
def parse_meme_pwm(path):
    rows, in_mat = [], False
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
                if len(vals) >= 4:
                    rows.append(vals[:4])
    if not rows:
        raise ValueError(f"No PWM read from {path}")
    pwm = np.array(rows, dtype=float)  # (w,4), A C G T
    pwm = np.clip(pwm, 1e-9, 1.0)
    pwm /= pwm.sum(axis=1, keepdims=True)  # ensure rows are probabilities
    return pwm

_RC = str.maketrans("ACGT", "TGCA")
def revcomp(s): return s.translate(_RC)[::-1]

def pwm_max_norm_score(seq, pwm):
    """Internal/training-style score:
       exp(max log-prob over windows) / exp(sum log maxcol)  in [0,1]
       and max over forward/revcomp strands.
    """
    L, w = len(seq), pwm.shape[0]
    if L < w: return 0.0
    maxcol = np.max(pwm, axis=1)
    max_log = np.log(maxcol).sum()

    def strand(s):
        best = -1e30
        for i in range(L-w+1):
            sub = s[i:i+w]
            ok, lp = True, 0.0
            for j,ch in enumerate(sub):
                a = IDX.get(ch, -1)
                if a < 0: ok = False; break
                lp += math.log(pwm[j,a])
            if ok and lp > best: best = lp
        if best <= -1e29: return 0.0
        return math.exp(best - max_log)

    return max(strand(seq), strand(revcomp(seq)))

# ---------- Plot helper ----------
def savefig(path):
    p.Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True,
                    help="Label=path.fasta  (e.g. Random=..., AI=..., Human=...)")
    ap.add_argument("--outdir", default="outputs/describe_internal")
    ap.add_argument("--k", type=int, default=4)
    # PWMs used in training (adjust if yours live elsewhere)
    ap.add_argument("--ppar", default="motifs/MA0065.2.meme")
    ap.add_argument("--nfkb", default="motifs/MA0105.4.meme")
    ap.add_argument("--tsne", action="store_true")
    args = ap.parse_args()

    out = p.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # Load sequences
    datasets = []
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(f"--input needs Label=path, got: {spec}")
        name, path = spec.split("=",1)
        seqs = read_fasta(path)
        datasets.append((name, path, seqs))
        print(f"[load] {name}: {len(seqs)} sequences from {path}")

    # 1) Composition
    labels = [n for n,_,_ in datasets]
    fr = []
    for _,_,seqs in datasets:
        tot = np.zeros(4, dtype=np.int64)
        for s in seqs: tot += seq_counts(s)
        n = max(1, tot.sum())
        fr.append((tot / n).tolist())
    fr = np.array(fr)
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(labels)); w = 0.18
    for i,b in enumerate(ALPH):
        ax.bar(x+(i-1.5)*w, fr[:,i], w, label=b)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction"); ax.set_title("AT/GC composition")
    ax.legend()
    savefig(out/"composition_bars.png")

    # 2) Entropy
    ent = []
    for name,_,seqs in datasets:
        vals = [shannon_entropy(s) for s in seqs]
        ent.append((name, np.array(vals)))
    fig, ax = plt.subplots(figsize=(10,6))
    ax.boxplot([v for _,v in ent], labels=[n for n,_ in ent], showmeans=True)
    ax.set_ylabel("Shannon entropy (bits)")
    ax.set_title("Per-sequence entropy")
    savefig(out/"entropy_boxplot.png")

    # 3) k-mer heatmap + PCA (+ optional t-SNE)
    idx, klist = kmer_index(args.k)
    K = len(idx)
    ds_freq = []
    X_list, y_list = [], []
    for name,_,seqs in datasets:
        tot = np.zeros(K, dtype=float); denom = 0
        for s in seqs:
            v = seq_kmer_vec(s, args.k, idx)
            tot += v; denom += 1 if v.sum()>0 else 0
            X_list.append(v); y_list.append(name)
        if denom>0: tot /= denom
        ds_freq.append((name, tot))
    mat = np.vstack([v for _,v in ds_freq]) if ds_freq else np.zeros((0,K))
    fig, ax = plt.subplots(figsize=(20,2 + 0.4*len(ds_freq)))
    im = ax.imshow(mat, aspect='auto')
    ax.set_yticks(np.arange(len(ds_freq))); ax.set_yticklabels([n for n,_ in ds_freq])
    ax.set_xticks(np.arange(K)); ax.set_xticklabels(klist, rotation=90, fontsize=6)
    ax.set_title(f"{args.k}-mer frequency (avg across sequences)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    savefig(out/f"kmer{args.k}_heatmap.png")

    X = np.vstack(X_list) if X_list else np.zeros((0,K))
    if X.shape[0] >= 2:
        Xc = X - X.mean(axis=0, keepdims=True)
        U,S,VT = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]
        fig, ax = plt.subplots(figsize=(9,8))
        for name in {n for n,_,_ in datasets}:
            m = [i for i,t in enumerate(y_list) if t==name]
            ax.scatter(Z[m,0], Z[m,1], s=10, alpha=0.7, label=name)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title(f"PCA on {args.k}-mer features")
        ax.legend(markerscale=2, frameon=False)
        savefig(out/f"pca_k{args.k}.png")

    if args.tsne and X.shape[0] >= 10:
        try:
            from sklearn.manifold import TSNE
            Zt = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=30).fit_transform(X)
            fig, ax = plt.subplots(figsize=(9,8))
            for name in {n for n,_,_ in datasets}:
                m = [i for i,t in enumerate(y_list) if t==name]
                ax.scatter(Zt[m,0], Zt[m,1], s=10, alpha=0.7, label=name)
            ax.set_title(f"t-SNE on {args.k}-mer features")
            ax.legend(markerscale=2, frameon=False)
            savefig(out/f"tsne_k{args.k}.png")
        except Exception as e:
            print("[warn] t-SNE skipped:", e)

    # 4) INTERNAL motif scores (PPARγ, NF-κB) + design_internal
    pwm_ppar = parse_meme_pwm(args.ppar)
    pwm_nfkb = parse_meme_pwm(args.nfkb)

    rows = []
    for name,_,seqs in datasets:
        for i,s in enumerate(seqs, start=1):
            a,c_,g,t = base_fracs(s)[:4]
            gc = (c_+g)
            entv = shannon_entropy(s)
            ppar = pwm_max_norm_score(s, pwm_ppar)
            nfkb = pwm_max_norm_score(s, pwm_nfkb)
            design = (ppar - nfkb + 1.0)/2.0
            rows.append(dict(
                dataset=name, seq_id=i, sequence=s, length=len(s),
                entropy=entv, A_frac=a, C_frac=c_, G_frac=g, T_frac=t, GC_frac=gc,
                pparg_internal=ppar, nfkb_internal=nfkb, design_internal=design
            ))

    # Save per-seq CSV
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out/"per_seq_scores_internal.csv", index=False)

    # Boxplots/hist for internal scores
    def boxplot(col, title, ylab, fname):
        fig, ax = plt.subplots(figsize=(10,6))
        parts = [df[df.dataset==n][col].values for n,_,_ in datasets]
        ax.boxplot(parts, labels=[n for n,_,_ in datasets], showmeans=True)
        ax.set_ylabel(ylab); ax.set_title(title)
        savefig(out/fname)

    boxplot("pparg_internal", "PPARγ (internal) by dataset", "PPARγ score", "pparg_internal_boxplot.png")
    boxplot("nfkb_internal", "NF-κB (internal) by dataset", "NF-κB score", "nfkb_internal_boxplot.png")
    boxplot("design_internal", "Design (internal) by dataset", "Design score", "design_internal_boxplot.png")

    fig, ax = plt.subplots(figsize=(12,6))
    for name,_,_ in datasets:
        vals = df[df.dataset==name]["design_internal"].values
        ax.hist(vals, bins=30, histtype="step", density=True, label=name)
    ax.set_xlabel("design score"); ax.set_ylabel("density")
    ax.set_title("Design score distribution (internal)")
    ax.legend()
    savefig(out/"design_internal_hist.png")

    # 5) Lightweight dataset summary
    import csv
    with open(out/"dataset_summary.csv","w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset","n_sequences","mean_len","mean_entropy","GC_fraction"])
        for name,_,seqs in datasets:
            n = len(seqs)
            mean_len = np.mean([len(s) for s in seqs]) if n else 0
            mean_ent = np.nanmean([shannon_entropy(s) for s in seqs]) if n else np.nan
            comp = np.zeros(4, dtype=np.int64)
            for s in seqs: comp += seq_counts(s)
            gc = (comp[1]+comp[2]) / max(1, comp.sum())
            w.writerow([name, n, round(float(mean_len),2), round(float(mean_ent),3), round(float(gc),3)])

    print(f"[done] Wrote figures & CSVs to: {out}")
if __name__ == "__main__":
    main()
