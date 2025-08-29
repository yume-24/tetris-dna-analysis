# analysis/describe/describe_fastas.py
"""
Describe FASTAs with external MEME PWM scoring (PPARγ::RXRA, NF-κB)
and k-mer/entropy/composition plots. Does NOT call the model.

Example:
  python analysis/describe/describe_fastas.py \
    --input Random=outputs/raw/seqs_random.fasta \
    --input AI=outputs/raw/seqs_main_ai.fasta \
    --input Human=outputs/raw/seqs_main_human.fasta \
    --k 4 \
    --outdir outputs/describe_run \
    --tsne
"""
import argparse, os, math, itertools, pathlib as p, sys
import numpy as np, matplotlib.pyplot as plt
from scipy import stats

# ---- stats helpers -----------------------------------------------------------
def cliffs_delta(x, y):
    x = np.asarray(x); y = np.asarray(y)
    x_sorted = np.sort(x); y_sorted = np.sort(y)
    i = j = gt = lt = 0
    nx, ny = len(x_sorted), len(y_sorted)
    while i < nx and j < ny:
        if x_sorted[i] > y_sorted[j]:
            gt += nx - i; j += 1
        elif x_sorted[i] < y_sorted[j]:
            lt += ny - j; i += 1
        else:
            if (nx - i) <= (ny - j): i += 1
            else: j += 1
    return float((gt - lt) / (nx * ny))

def summarize_two_groups(x, y):
    x = np.asarray(x); y = np.asarray(y)
    u = stats.mannwhitneyu(x, y, alternative="two-sided")
    ks = stats.ks_2samp(x, y, alternative="two-sided", method="auto")
    cd = cliffs_delta(x, y)
    return {
        "n_x": len(x), "n_y": len(y),
        "mw_U": u.statistic, "mw_p": u.pvalue,
        "ks_D": ks.statistic, "ks_p": ks.pvalue,
        "cliff_delta": cd
    }

def annotate_stats_box(ax, stats_dict, where="upper right", title=None):
    t = stats_dict
    lines = []
    if title: lines.append(title)
    lines += [
        f"n₁={t['n_x']}, n₂={t['n_y']}",
        f"Mann–Whitney U p={t['mw_p']:.2e}",
        f"KS D={t['ks_D']:.3f}, p={t['ks_p']:.2e}",
        f"Cliff’s Δ={t['cliff_delta']:.2f}",
    ]
    ha = {"upper left":"left","upper right":"right"}.get(where, "right")
    xy = {"upper left":(0.01,0.99),"upper right":(0.99,0.99)}.get(where,(0.99,0.99))
    ax.text(*xy, "\n".join(lines), transform=ax.transAxes,
            va="top", ha=ha, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9))

# ---------- FASTA ----------
ALPH = "ACGT"; IDX = {c:i for i,c in enumerate(ALPH)}
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
    c = seq_counts(s).astype(float); n = c.sum()
    if n == 0: return np.nan
    p = c / n; p = p[p>0]
    return float(-(p*np.log2(p)).sum())

# ---------- k-mers ----------
def all_kmers(k): return ["".join(t) for t in itertools.product(ALPH, repeat=k)]
def kmer_index(k): km = all_kmers(k); return {kmer:i for i,kmer in enumerate(km)}, km
def seq_kmer_vec(s, k, idx):
    v = np.zeros(len(idx), dtype=float); valid=0
    for i in range(len(s)-k+1):
        sub = s[i:i+k]
        if any(ch not in ALPH for ch in sub): continue
        v[idx[sub]] += 1; valid += 1
    if valid>0: v /= valid
    return v

# ---------- external MEME PWM scoring (max-pool, normalized) ----------
def parse_meme_pwm(path):
    rows=[]; in_mat=False
    with open(path) as f:
        for line in f:
            line=line.strip()
            if line.lower().startswith("letter-probability matrix"):
                in_mat=True; continue
            if in_mat:
                if not line or line[0].isalpha(): break
                vals=[float(x) for x in line.split()]
                if len(vals)>=4: rows.append(vals[:4])
    if not rows: raise ValueError(f"No PWM read from {path}")
    pwm=np.array(rows, dtype=float)
    pwm=np.clip(pwm,1e-9,1.0); pwm/=pwm.sum(axis=1, keepdims=True)
    return pwm
RC=str.maketrans("ACGT","TGCA")
def revcomp(s): return s.translate(RC)[::-1]
def pwm_max_norm_score(seq, pwm):
    L,w=len(seq), pwm.shape[0]
    if L<w: return 0.0
    maxcol=np.max(pwm,axis=1); max_log=np.log(maxcol).sum()
    def strand(s):
        best=-1e30
        for i in range(L-w+1):
            sub=s[i:i+w]; ok=True; lp=0.0
            for j,ch in enumerate(sub):
                a=IDX.get(ch,-1)
                if a<0: ok=False; break
                lp+=math.log(pwm[j,a])
            if ok and lp>best: best=lp
        if best<=-1e29: return 0.0
        return math.exp(best-max_log)
    return max(strand(seq), strand(revcomp(seq)))

def savefig(path):
    p.Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=220); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True,
                    help="Label=path.fasta (Random=..., AI=..., Human=...)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--outdir", default="outputs/describe_run")
    ap.add_argument("--ppar", default="motifs/MA0065.2.meme")
    ap.add_argument("--nfkb", default="motifs/MA0105.4.meme")
    ap.add_argument("--tsne", action="store_true")
    args = ap.parse_args()

    datasets=[]
    for spec in args.input:
        if "=" not in spec: raise SystemExit(f"--input needs Label=path, got: {spec}")
        name, path = spec.split("=",1)
        seqs = read_fasta(path); datasets.append((name, path, seqs))
        print(f"[load] {name}: {len(seqs)} seqs from {path}")

    # helper: label with n
    n_map = {name: len(seqs) for name, _, seqs in datasets}
    def with_n(name): return f"{name} (n={n_map[name]})"

    out = p.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # composition bars
    labels=[]; fr=[]
    for name,_,seqs in datasets:
        tot=np.zeros(4,dtype=np.int64)
        for s in seqs: tot+=seq_counts(s)
        n=tot.sum(); frac=tot/n if n>0 else np.zeros(4)
        labels.append(with_n(name)); fr.append(frac)
    fr=np.vstack(fr)
    fig,ax=plt.subplots(figsize=(7,4)); x=np.arange(len(labels)); w=0.18
    for i,b in enumerate(ALPH):
        ax.bar(x+(i-1.5)*w, fr[:,i], w, label=b)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Fraction")
    ax.set_title("AT/GC composition"); ax.legend()
    savefig(out/"composition_bars.png")

    # entropy boxplot
    ent=[(with_n(n), np.array([shannon_entropy(s) for s in seqs])) for n,_,seqs in datasets]
    fig,ax=plt.subplots(figsize=(7,4))
    ax.boxplot([v for _,v in ent], labels=[n for n,_ in ent], showmeans=True)
    ax.set_ylabel("Shannon entropy (bits)"); ax.set_title("Per-sequence entropy")
    savefig(out/"entropy_boxplot.png")

    # k-mer heatmap + PCA/tSNE
    idx, klist = kmer_index(args.k); K=len(idx)
    mats=[]
    for _,_,seqs in datasets:
        tot=np.zeros(K); denom=0
        for s in seqs:
            v=seq_kmer_vec(s,args.k,idx); tot+=v
            denom+= 1 if v.sum()>0 else 0
        if denom>0: tot/=denom
        mats.append(tot)
    M=np.vstack(mats)
    fig,ax=plt.subplots(figsize=(max(6,K*0.12), 2+0.4*len(datasets)))
    im=ax.imshow(M,aspect="auto")
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels([with_n(n) for n,_,_ in datasets])
    ax.set_xticks(np.arange(K)); ax.set_xticklabels(klist, rotation=90)
    ax.set_title(f"{args.k}-mer frequency (avg)"); fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    savefig(out/f"kmer{args.k}_heatmap.png")

    # PCA
    X=[]; y=[]
    for name,_,seqs in datasets:
        for s in seqs:
            X.append(seq_kmer_vec(s,args.k,idx)); y.append(name)
    X=np.vstack(X) if X else np.zeros((0,K))
    if X.shape[0]>=2:
        Xc=X-X.mean(axis=0,keepdims=True)
        U,S,VT=np.linalg.svd(Xc,full_matrices=False)
        Z=U[:,:2]*S[:2]
        fig,ax=plt.subplots(figsize=(6,5))
        for name in [n for n,_,_ in datasets]:
            m=[i for i,t in enumerate(y) if t==name]
            ax.scatter(Z[m,0],Z[m,1],s=10,alpha=0.6,label=with_n(name))
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(f"PCA on {args.k}-mer features")
        ax.legend(markerscale=2, frameon=False)
        savefig(out/f"pca_k{args.k}.png")
    if args.tsne and X.shape[0]>=10:
        try:
            from sklearn.manifold import TSNE
            Zt=TSNE(n_components=2,init="random",learning_rate="auto",perplexity=30).fit_transform(X)
            fig,ax=plt.subplots(figsize=(6,5))
            for name in [n for n,_,_ in datasets]:
                m=[i for i,t in enumerate(y) if t==name]
                ax.scatter(Zt[m,0],Zt[m,1],s=10,alpha=0.6,label=with_n(name))
            ax.set_title(f"t-SNE on {args.k}-mer features")
            ax.legend(markerscale=2,frameon=False)
            savefig(out/f"tsne_k{args.k}.png")
        except Exception as e:
            print("[warn] t-SNE skipped:", e)

    # External MEME PWM scoring (PPARγ::RXRA, NF-κB), max-pooled & normalized
    try:
        pwm_ppar = parse_meme_pwm(args.ppar)
        pwm_nfkb = parse_meme_pwm(args.nfkb)
    except Exception as e:
        print("[warn] Motif scoring skipped:", e)
        pwm_ppar = pwm_nfkb = None

    rows=[]
    if pwm_ppar is not None and pwm_nfkb is not None:
        for name,_,seqs in datasets:
            for i,s in enumerate(seqs, start=1):
                a,c,g,t,gc = base_fracs(s)
                ent = shannon_entropy(s)
                ppar = pwm_max_norm_score(s, pwm_ppar)
                nfkb = pwm_max_norm_score(s, pwm_nfkb)
                design = (ppar - nfkb + 1.0)/2.0
                rows.append(dict(
                    dataset=name, seq_id=i, sequence=s, length=len(s),
                    entropy=ent, A_frac=a, C_frac=c, G_frac=g, T_frac=t, GC_frac=gc,
                    pparg_score=ppar, nfkb_score=nfkb, design_score=design
                ))
        import pandas as pd
        df=pd.DataFrame(rows); df.to_csv(out/"per_seq_scores.csv", index=False)

        for col,label in [("pparg_score","PPARγ score"), ("nfkb_score","NF-κB score"), ("design_score","Design score")]:
            fig,ax=plt.subplots(figsize=(7,4))
            parts=[df[df.dataset==n][col].values for n,_,_ in datasets]
            ax.boxplot(parts, labels=[with_n(n) for n,_,_ in datasets], showmeans=True)
            ax.set_ylabel(label); ax.set_title(f"{label} by dataset")
            fname = "design_boxplot.png" if col=="design_score" else f"{col}_boxplot.png".replace("_score","")
            savefig(out/fname)

        fig,ax=plt.subplots(figsize=(7,4))
        for name,_,_ in datasets:
            vals=df[df.dataset==name]["design_score"].values
            ax.hist(vals,bins=30,histtype="step",density=True,label=with_n(name))
        ax.set_xlabel("Design score (external PWM)"); ax.set_ylabel("density"); ax.set_title("Design score distribution")
        ax.legend()
        savefig(out/"design_hist.png")

    # dataset summary CSV
    import csv
    with open(out/"dataset_summary.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["dataset","n_sequences","mean_len","mean_entropy","GC_fraction"])
        for name,_,seqs in datasets:
            n=len(seqs)
            mean_len=float(np.mean([len(s) for s in seqs])) if n else 0.0
            mean_ent=float(np.nanmean([shannon_entropy(s) for s in seqs])) if n else float("nan")
            comp=np.zeros(4,dtype=np.int64)
            for s in seqs: comp+=seq_counts(s)
            gc=(comp[1]+comp[2])/max(1,comp.sum())
            w.writerow([name, n, round(mean_len,2), round(mean_ent,3), round(float(gc),3)])

    print(f"[done] Wrote figures & CSVs to: {args.outdir}")

if __name__ == "__main__":
    import math, itertools  # keep local imports near usage
    main()
