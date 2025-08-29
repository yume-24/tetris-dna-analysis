# analysis/describe/describe_fastas_dual.py
"""
Side-by-side external (MEME) vs internal (model) design plots.

Usage (example):
  # external: 3 FASTAs
  # internal: pass CSVs produced by correlate_game_vs_dna.py (they contain 'sequence' and 'design_model')
  python analysis/describe/describe_fastas_dual.py \
    --input Random=outputs/raw/seqs_random.fasta \
    --input AI=outputs/raw/seqs_main_ai.fasta \
    --input Human=outputs/raw/seqs_main_human.fasta \
    --internal_csv AI=outputs/analysis/per_seq_test.csv \
    --internal_csv Human=outputs/analysis/per_seq_human.csv \
    --k 4 --outdir outputs/describe_run_dual --tsne

Outputs (into --outdir):
  composition_bars.png, entropy_boxplot.png, kmer{k}_heatmap.png, pca_k{k}.png
  tsne_k{k}.png (if --tsne)
  design_external_boxplot.png
  design_internal_boxplot.png  (only datasets with internal scores)
  design_external_vs_internal_scatter_<DATASET>.png (for each dataset with both)
  pparg_boxplot.png, nfkb_boxplot.png, design_external_hist.png
  per_seq_scores.csv  (external columns + internal column if provided)
  dataset_summary.csv
"""
import argparse, os, math, itertools, sys, pathlib as p
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    a, c_, g, t = c / n
    return float(a), float(c_), float(g), float(t), float((c_+g))

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
    pwm = np.array(rows, dtype=float)  # (w,4) A C G T
    pwm = np.clip(pwm, 1e-9, 1.0)
    pwm /= pwm.sum(axis=1, keepdims=True)
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

# ---------- Plot helper ----------
def savefig(path):
    p.Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True,
                    help="Label=path.fasta (e.g., Random=..., AI=..., Human=...)")
    ap.add_argument("--internal_csv", action="append", default=[],
                    help="Label=path.csv containing 'sequence' (or 'seq') and an internal score column")
    ap.add_argument("--internal_col", default="design_model",
                    help="column name in internal CSV with the model score (default: design_model)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--outdir", default="outputs/describe_run_dual")
    ap.add_argument("--ppar", default="motifs/MA0065.2.meme")
    ap.add_argument("--nfkb", default="motifs/MA0105.4.meme")
    ap.add_argument("--tsne", action="store_true")
    args = ap.parse_args()

    out = p.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # load FASTAs
    datasets = []
    for spec in args.input:
        if "=" not in spec: raise SystemExit(f"--input needs Label=path, got: {spec}")
        label, path = spec.split("=",1)
        seqs = read_fasta(path)
        datasets.append((label, path, seqs))
        print(f"[load] {label}: {len(seqs)} seqs from {path}")

    # compute external scores
    pwm_ppar = parse_meme_pwm(args.ppar)
    pwm_nfkb = parse_meme_pwm(args.nfkb)

    rows = []
    for name,_,seqs in datasets:
        for i,s in enumerate(seqs, start=1):
            a,c,g,t,gc = base_fracs(s)
            ent = shannon_entropy(s)
            ppar = pwm_max_norm_score(s, pwm_ppar)
            nfkb = pwm_max_norm_score(s, pwm_nfkb)
            design_ext = (ppar - nfkb + 1.0)/2.0
            rows.append(dict(dataset=name, seq_id=i, sequence=s, length=len(s),
                             entropy=ent, A_frac=a, C_frac=c, G_frac=g, T_frac=t, GC_frac=gc,
                             pparg_score=ppar, nfkb_score=nfkb, design_external=design_ext))
    ext_df = pd.DataFrame(rows)

    # read optional internal CSVs and align by sequence
    int_parts = []
    for spec in args.internal_csv:
        if "=" not in spec: raise SystemExit(f"--internal_csv needs Label=path, got: {spec}")
        label, path = spec.split("=",1)
        t = pd.read_csv(path)
        seq_col = "sequence" if "sequence" in t.columns else ("seq" if "seq" in t.columns else None)
        if seq_col is None:
            raise SystemExit(f"{path} must contain a 'sequence' or 'seq' column")
        if args.internal_col not in t.columns:
            # tolerate a few common alternatives
            for alt in ["DESIGN_MODEL","design_model","design_internal","model_design","design"]:
                if alt in t.columns:
                    args.internal_col = alt
                    break
            else:
                raise SystemExit(f"{path} must contain an internal score column (e.g., 'design_model')")
        t = t[[seq_col, args.internal_col]].copy()
        t.columns = ["sequence","design_internal"]
        t["dataset"] = label
        # de-duplicate by exact sequence (mean if repeated)
        t = t.groupby(["dataset","sequence"], as_index=False)["design_internal"].mean()
        int_parts.append(t)

    if int_parts:
        int_df = pd.concat(int_parts, ignore_index=True)
    else:
        int_df = pd.DataFrame(columns=["dataset","sequence","design_internal"])

    # merge external + internal when available
    merged = pd.merge(ext_df, int_df, on=["dataset","sequence"], how="left")
    merged.to_csv(out/"per_seq_scores.csv", index=False)

    # ========== standard plots ==========
    # composition
    labels = [n for n,_,_ in datasets]
    fracs = []
    for name,_,seqs in datasets:
        tot = np.zeros(4, dtype=np.int64)
        for s in seqs: tot += seq_counts(s)
        n = tot.sum() if tot.sum()>0 else 1
        fracs.append((tot / n).tolist())
    fracs = np.array(fracs)
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(labels)); width = 0.18
    for i, base in enumerate(ALPH):
        ax.bar(x + (i-1.5)*width, fracs[:,i], width, label=base)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction"); ax.set_title("AT/GC composition")
    ax.legend()
    savefig(out/"composition_bars.png")

    # entropy
    fig, ax = plt.subplots(figsize=(10,6))
    parts = [merged[merged.dataset==n]["entropy"].values for n,_,_ in datasets]
    ax.boxplot(parts, labels=[n for n,_,_ in datasets], showmeans=True)
    ax.set_ylabel("Shannon entropy (bits)")
    ax.set_title("Per-sequence entropy")
    savefig(out/"entropy_boxplot.png")

    # k-mer heatmap + PCA + optional t-SNE
    k = args.k
    idx, klist = kmer_index(k)
    K = len(idx)
    ds_freq = []
    X_list, y_list = [], []
    for name,_,seqs in datasets:
        tot = np.zeros(K, dtype=float); denom = 0
        for s in seqs:
            v = seq_kmer_vec(s, k, idx)
            tot += v; denom += 1 if v.sum()>0 else 0
            X_list.append(v); y_list.append(name)
        tot = tot/denom if denom>0 else tot
        ds_freq.append((name, tot))
    mat = np.vstack([v for _,v in ds_freq])
    fig, ax = plt.subplots(figsize=(max(6, K*0.12), 2+0.4*len(ds_freq)))
    im = ax.imshow(mat, aspect='auto')
    ax.set_yticks(np.arange(len(ds_freq))); ax.set_yticklabels([n for n,_ in ds_freq])
    ax.set_xticks(np.arange(K)); ax.set_xticklabels(klist, rotation=90)
    ax.set_title(f"{k}-mer frequency (avg across sequences)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    savefig(out/f"kmer{k}_heatmap.png")

    X = np.vstack(X_list) if X_list else np.zeros((0,K))
    if X.shape[0] >= 2:
        Xc = X - X.mean(axis=0, keepdims=True)
        U,S,VT = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]
        fig, ax = plt.subplots(figsize=(8,7))
        for name in {n for n,_,_ in datasets}:
            m = [i for i,t in enumerate(y_list) if t==name]
            ax.scatter(Z[m,0], Z[m,1], s=12, alpha=0.6, label=name)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(f"PCA on {k}-mer features")
        ax.legend(markerscale=2, frameon=False)
        savefig(out/f"pca_k{k}.png")

    if args.tsne and X.shape[0] >= 10:
        try:
            from sklearn.manifold import TSNE
            Zt = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=30).fit_transform(X)
            fig, ax = plt.subplots(figsize=(8,7))
            for name in {n for n,_,_ in datasets}:
                m = [i for i,t in enumerate(y_list) if t==name]
                ax.scatter(Zt[m,0], Zt[m,1], s=12, alpha=0.6, label=name)
            ax.set_title(f"t-SNE on {k}-mer features")
            ax.legend(markerscale=2, frameon=False)
            savefig(out/f"tsne_k{k}.png")
        except Exception as e:
            print("[warn] t-SNE skipped:", e)

    # ========== external vs internal design: side-by-side ==========
    # external boxplot
    fig, ax = plt.subplots(figsize=(10,6))
    parts = [merged[merged.dataset==n]["design_external"].values for n,_,_ in datasets]
    ax.boxplot(parts, labels=[n for n,_,_ in datasets], showmeans=True)
    ax.set_ylabel("Design score"); ax.set_title("Design (external, MEME) by dataset")
    savefig(out/"design_external_boxplot.png")

    # internal boxplot (only datasets present in internal CSVs)
    present_internal = sorted(merged[~merged["design_internal"].isna()]["dataset"].unique().tolist())
    if present_internal:
        fig, ax = plt.subplots(figsize=(10,6))
        parts = [merged[(merged.dataset==n) & (~merged.design_internal.isna())]["design_internal"].values
                 for n in present_internal]
        ax.boxplot(parts, labels=present_internal, showmeans=True)
        ax.set_ylabel("Design score"); ax.set_title("Design (internal, model) by dataset")
        savefig(out/"design_internal_boxplot.png")
    else:
        print("[note] No internal CSVs provided (or no overlap with FASTAs) — internal plots skipped.")

    # PPARγ / NF-κB external boxplots
    for col, title in [("pparg_score","PPARγ (external) by dataset"),
                       ("nfkb_score","NF-κB (external) by dataset")]:
        fig, ax = plt.subplots(figsize=(10,6))
        parts = [merged[merged.dataset==n][col].values for n,_,_ in datasets]
        ax.boxplot(parts, labels=[n for n,_,_ in datasets], showmeans=True)
        ax.set_ylabel(col.replace("_"," "))
        ax.set_title(title)
        savefig(out/f"{col}_boxplot.png")

    # external design histogram overlay
    fig, ax = plt.subplots(figsize=(10,6))
    for name,_,_ in datasets:
        vals = merged[merged.dataset==name]["design_external"].values
        ax.hist(vals, bins=30, histtype="step", density=True, label=name)
    ax.set_xlabel("design score")
    ax.set_ylabel("density")
    ax.set_title("Design score distribution (external)")
    ax.legend()
    savefig(out/"design_external_hist.png")

    # scatter: external vs internal per dataset
    for name in present_internal:
        sub = merged[(merged.dataset==name) & (~merged.design_internal.isna())].copy()
        if len(sub) < 2: continue
        x = sub["design_external"].values
        y = sub["design_internal"].values
        pear = float(np.corrcoef(x, y)[0,1]) if np.std(x)>0 and np.std(y)>0 else np.nan
        # line fit
        a,b = np.polyfit(x,y,1) if len(sub)>=2 else (np.nan,np.nan)
        xs = np.linspace(x.min(), x.max(), 100); ys = a*xs+b if not np.isnan(a) else None
        plt.figure(figsize=(7,6))
        plt.scatter(x, y, s=12, alpha=0.6)
        if ys is not None: plt.plot(xs, ys)
        plt.xlabel("design_external (MEME)")
        plt.ylabel("design_internal (model)")
        plt.title(f"{name}: external vs internal (Pearson={pear:.3f}, n={len(sub)})")
        savefig(out/f"design_external_vs_internal_scatter_{name}.png")

    # dataset summary
    with open(out/"dataset_summary.csv","w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["dataset","n_sequences","mean_len","mean_entropy","GC_fraction"])
        for name,_,seqs in datasets:
            n = len(seqs)
            mean_len = np.mean([len(s) for s in seqs]) if n else 0
            mean_ent = np.nanmean([shannon_entropy(s) for s in seqs]) if n else np.nan
            comp = np.zeros(4, dtype=np.int64)
            for s in seqs: comp += seq_counts(s)
            gc = (comp[1]+comp[2]) / max(1, comp.sum()) if comp.sum()>0 else 0.0
            w.writerow([name, n, round(float(mean_len),2), round(float(mean_ent),3), round(float(gc),3)])

    print(f"[done] wrote figures & CSVs to: {args.outdir}")

if __name__ == "__main__":
    main()
