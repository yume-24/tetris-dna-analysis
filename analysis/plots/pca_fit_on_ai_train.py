#!/usr/bin/env python3
"""
Fit PCA on AI-TRAIN features, project AI-TEST + Human.

python analysis/plots/pca_fit_on_ai_train.py \
  --ai_train_fasta  outputs/raw/seqs_main_ai.fasta \
  --ai_train_scores outputs/analysis/per_seq_test.csv \
  --ai_test_fasta   outputs/analysis/matched_raw4/ai_filtered.fasta \
  --ai_test_scores  outputs/analysis/matched_raw4/ai_filtered.csv \
  --human_fasta     outputs/analysis/matched_raw4/human_filtered.fasta \
  --human_scores    outputs/analysis/matched_raw4/human_filtered.csv \
  --score_col score_raw --min_score=-1 \
  --features kmer pwm --k 4 \
  --ppar motifs/MA0065.2.meme --nfkb motifs/MA0105.4.meme \
  --max_test 100000 --seed 7 \
  --outdir outputs/analysis/pca_refit_matched


Features supported:
  - k-mer frequencies (default k=4)
  - PWM logit features: max & mean for PPARγ / NF-κB (MEME PWMs)
  - DNABERT-2 [CLS] or mean-pooled embedding (optional, CPU ok if model loads)

Filtering:
  - Keep sequences with game score >= --min_score (default 4) from a CSV
  - After filtering: shuffle with --seed and take up to --max_test from AI-TEST, Human

Outputs:
  - pca_scatter.png (AI-TEST vs Human, PCA fit on AI-TRAIN)
  - explained_variance.csv
  - If dnabert selected: dnabert_corr_*.png (embedding mean vs game/design/entropy)
  - entropy_vs_design.png (if both are available)
  - counts printed to stdout

Usage example
-------------
python analysis/plots/pca_fit_on_ai_train.py \
  --ai_train_fasta outputs/raw/seqs_train_ai_10k.fasta \
  --ai_train_scores outputs/analysis/per_seq_train_ai.csv \
  --ai_test_fasta outputs/raw/seqs_main_ai.fasta \
  --ai_test_scores outputs/analysis/per_seq_test.csv \
  --human_fasta outputs/raw/seqs_main_human.fasta \
  --human_scores outputs/analysis/per_seq_human.csv \
  --score_col score_raw --min_score 4 \
  --features kmer pwm \
  --k 4 --ppar motifs/MA0065.2.meme --nfkb motifs/MA0105.4.meme \
  --max_test 200 --seed 7 \
  --outdir outputs/analysis/pca_refit_k4

To try DNABERT as well (if your env supports it):
  add: --features dnabert --dnabert_model zhihan1996/DNABERT-2-117M
"""

import os, argparse, math, itertools, pathlib as p
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ------------------ I/O helpers ------------------
ALPH = "ACGT"
IDX = {c:i for i,c in enumerate(ALPH)}
RC = str.maketrans("ACGT","TGCA")

def read_fasta(path):
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            if line.startswith(">"):
                if cur: seqs.append("".join(cur).upper())
                cur=[]
            else:
                cur.append(line)
    if cur: seqs.append("".join(cur).upper())
    return seqs

def ensure_cols(df, need):
    missing=[c for c in need if c not in df.columns]
    if missing: raise ValueError(f"Missing columns {missing} in CSV.")
    return df

def filter_by_score(df, score_col, thr):
    m = df[score_col] >= thr
    return df.loc[m].reset_index(drop=True)

def take_seeded(df, n, seed):
    if n is None or len(df)<=n: return df.copy()
    return df.sample(n=n, random_state=seed).reset_index(drop=True)

def shannon_entropy(s):
    c = np.zeros(4)
    for ch in s:
        i = IDX.get(ch, -1)
        if i>=0: c[i]+=1
    n=c.sum()
    if n==0: return np.nan
    p=c/n; p=p[p>0]
    return float(-(p*np.log2(p)).sum())

# ------------------ Features: k-mer ------------------
def all_kmers(k): return ["".join(t) for t in itertools.product(ALPH, repeat=k)]

def kmer_index(k):
    km = all_kmers(k)
    return {kmer:i for i,kmer in enumerate(km)}, km

def seq_kmer_vec(s, k, idx):
    v = np.zeros(len(idx), dtype=float)
    valid=0
    for i in range(len(s)-k+1):
        sub = s[i:i+k]
        if any(ch not in ALPH for ch in sub): continue
        v[idx[sub]] += 1
        valid+=1
    if valid>0: v/=valid
    return v

# ------------------ Features: PWM ------------------
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
    pwm=np.clip(pwm,1e-9,1.0)
    pwm/=pwm.sum(axis=1, keepdims=True)
    return pwm  # (w,4) A C G T

def pwm_scan_logits(seq, pwm):
    """Return per-position log-prob (logit) along sequence for fwd+rev best."""
    w = pwm.shape[0]
    if len(seq)<w: return np.array([])
    logp = np.log(pwm)  # (w,4)
    # forward
    out_f = np.full(len(seq)-w+1, -1e30)
    for i in range(len(out_f)):
        s = seq[i:i+w]
        ok=True; lp=0.0
        for j,ch in enumerate(s):
            a=IDX.get(ch,-1)
            if a<0: ok=False; break
            lp += logp[j,a]
        out_f[i]=lp if ok else -1e30
    # reverse complement
    r = seq.translate(RC)[::-1]
    out_r = np.full(len(seq)-w+1, -1e30)
    for i in range(len(out_r)):
        s = r[i:i+w]
        ok=True; lp=0.0
        for j,ch in enumerate(s):
            a=IDX.get(ch,-1)
            if a<0: ok=False; break
            lp += logp[j,a]
        out_r[i]=lp if ok else -1e30
    return np.maximum(out_f, out_r)

def pwm_features_for_seq(seq, pwm):
    prof = pwm_scan_logits(seq, pwm)
    if prof.size==0: return dict(max=-1e9, mean=-1e9)
    return dict(max=float(np.max(prof)), mean=float(np.mean(prof)))

# ------------------ Features: DNABERT (optional) ------------------
def dnabert_embed_batch(seqs, model_name, device="cpu", pool="cls"):
    """Return (N, D) embedding. Requires transformers installed."""
    try:
        import torch, os
        from transformers import AutoTokenizer, AutoModel
        os.environ.setdefault("DISABLE_TRITON","1")  # avoid GPU-only import paths
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.to(device).eval()
        outs=[]
        with torch.no_grad():
            for s in seqs:
                b = tok(s, return_tensors="pt")
                for k in b: b[k]=b[k].to(device)
                o = model(**b)
                h = o.last_hidden_state  # (1, L, D)
                if pool=="cls":
                    # some models prepend special token at 0
                    emb = h[:,0,:]
                else:
                    emb = h.mean(dim=1)
                outs.append(emb.detach().cpu().numpy().squeeze())
        return np.vstack(outs)
    except Exception as e:
        print("[warn] DNABERT skipped:", e)
        return None

# ------------------ Build feature matrix ------------------
def build_features(seqs, which, k, ppar_pwm, nfkb_pwm, dnabert_model):
    feats=[]
    cols=[]
    if "kmer" in which:
        idx,_ = kmer_index(k)
        X = np.vstack([seq_kmer_vec(s, k, idx) for s in seqs])
        feats.append(X); cols += [f"kmer_{k}={t}" for t in range(X.shape[1])]
    if "pwm" in which:
        fp=[]; fn=[]
        for s in seqs:
            p = pwm_features_for_seq(s, ppar_pwm)
            n = pwm_features_for_seq(s, nfkb_pwm)
            fp.append([p["max"], p["mean"]])
            fn.append([n["max"], n["mean"]])
        Xp=np.array(fp); Xn=np.array(fn)
        feats.append(Xp); feats.append(Xn)
        cols += ["ppar_max","ppar_mean","nfkb_max","nfkb_mean"]
    if "dnabert" in which:
        X = dnabert_embed_batch(seqs, dnabert_model, device="cpu", pool="cls")
        if X is not None:
            feats.append(X); cols += [f"dnabert_{i}" for i in range(X.shape[1])]
    if not feats:
        raise SystemExit("No features selected.")
    X = np.hstack(feats)
    return X, cols

# ------------------ Plot helpers ------------------
def savefig(path):
    p.Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220); plt.close()

def scatter_two(ax, Z1, Z2, lab1, lab2):
    ax.scatter(Z1[:,0], Z1[:,1], s=28, alpha=0.8, label=f"{lab1} (n={len(Z1)})")
    ax.scatter(Z2[:,0], Z2[:,1], s=28, alpha=0.8, label=f"{lab2} (n={len(Z2)})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(frameon=False)

def corr_plot(x, y, title, outpng, xlabel, ylabel):
    import scipy.stats as st
    r = np.corrcoef(x, y)[0,1] if len(x)>1 else np.nan
    rho,_ = st.spearmanr(x, y) if len(x)>1 else (np.nan, np.nan)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(x, y, s=22, alpha=0.8)
    # regression line
    if len(x)>1:
        m,b = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 100)
        ax.plot(xs, m*xs+b, lw=2)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.text(0.02, 0.98, f"Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}\n n = {len(x)}",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    savefig(outpng)

# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai_train_fasta", required=True)
    ap.add_argument("--ai_train_scores", required=True)
    ap.add_argument("--ai_test_fasta", required=True)
    ap.add_argument("--ai_test_scores", required=True)
    ap.add_argument("--human_fasta", required=True)
    ap.add_argument("--human_scores", required=True)
    ap.add_argument("--score_col", default="score_raw")
    ap.add_argument("--min_score", type=float, default=4.0)
    ap.add_argument("--features", nargs="+", default=["kmer","pwm"],
                    choices=["kmer","pwm","dnabert"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--ppar", default="motifs/MA0065.2.meme")
    ap.add_argument("--nfkb", default="motifs/MA0105.4.meme")
    ap.add_argument("--dnabert_model", default="zhihan1996/DNABERT-2-117M")
    ap.add_argument("--max_test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = p.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # Load FASTA + CSV
    train_seqs = read_fasta(args.ai_train_fasta)
    test_seqs  = read_fasta(args.ai_test_fasta)
    hum_seqs   = read_fasta(args.human_fasta)

    train_csv = pd.read_csv(args.ai_train_scores)
    test_csv  = pd.read_csv(args.ai_test_scores)
    hum_csv   = pd.read_csv(args.human_scores)

    need_cols = [args.score_col, "seq"]
    ensure_cols(train_csv, need_cols); ensure_cols(test_csv, need_cols); ensure_cols(hum_csv, need_cols)

    # Align safety: assume rows correspond 1:1 to FASTA order (your export does this)
    assert len(train_seqs)==len(train_csv), "AI-TRAIN fasta/csv length mismatch"
    assert len(test_seqs)==len(test_csv),   "AI-TEST fasta/csv length mismatch"
    assert len(hum_seqs)==len(hum_csv),     "Human fasta/csv length mismatch"

    # Threshold and sample
    f_train = filter_by_score(train_csv, args.score_col, args.min_score)
    f_test  = filter_by_score(test_csv,  args.score_col, args.min_score)
    f_hum   = filter_by_score(hum_csv,   args.score_col, args.min_score)

    train_seqs = [train_seqs[i] for i in f_train.index]
    test_seqs  = [test_seqs[i]  for i in f_test.index]
    hum_seqs   = [hum_seqs[i]   for i in f_hum.index]

    f_test = take_seeded(f_test, args.max_test, args.seed)
    f_hum  = take_seeded(f_hum,  args.max_test, args.seed)
    test_seqs = [test_seqs[i] for i in f_test.index]
    hum_seqs  = [hum_seqs[i]  for i in f_hum.index]

    print(f"[counts] AI-TRAIN≥{args.min_score}: {len(train_seqs)}  | AI-TEST: {len(test_seqs)} | Human: {len(hum_seqs)}")

    # Entropy (for corrs later)
    f_test["entropy"] = [shannon_entropy(s) for s in test_seqs]
    f_hum["entropy"]  = [shannon_entropy(s) for s in hum_seqs]

    # PWMs if needed
    if "pwm" in args.features:
        pwm_ppar = parse_meme_pwm(args.ppar)
        pwm_nfkb = parse_meme_pwm(args.nfkb)
    else:
        pwm_ppar = pwm_nfkb = None

    # Build FEATURES
    Xtr, cols = build_features(train_seqs, args.features, args.k, pwm_ppar, pwm_nfkb, args.dnabert_model)
    Xai, _    = build_features(test_seqs,  args.features, args.k, pwm_ppar, pwm_nfkb, args.dnabert_model)
    Xhu, _    = build_features(hum_seqs,   args.features, args.k, pwm_ppar, pwm_nfkb, args.dnabert_model)

    # Scale + PCA (fit on TRAIN only!)
    scaler = StandardScaler(with_mean=True, with_std=True).fit(Xtr)
    Ztr = scaler.transform(Xtr)
    pca = PCA(n_components=2, svd_solver="full").fit(Ztr)

    # Project test/human
    Zai = pca.transform(scaler.transform(Xai))
    Zhu = pca.transform(scaler.transform(Xhu))

    # Save explained variance
    pd.DataFrame({
        "component":[1,2],
        "explained_variance_ratio": pca.explained_variance_ratio_
    }).to_csv(out/"explained_variance.csv", index=False)

    # PCA scatter
    fig, ax = plt.subplots(figsize=(7.5,6.5))
    scatter_two(ax, Zai, Zhu, "AI (Test)", "Human")
    ax.set_title("PCA (fit on AI-TRAIN) — features: " + "+".join(args.features))
    savefig(out/"pca_scatter.png")

    # DNABERT correlations (if we have dnabert features, take mean across dims)
    if "dnabert" in args.features:
        # mean embedding along feature axis (already pooled per seq)
        def emb_mean(X): return X.mean(axis=1)
        e_ai = emb_mean(Xai); e_hu = emb_mean(Xhu)
        # try to guess design column; fallbacks
        des_col = "design_model" if "design_model" in f_test.columns else ("design_score" if "design_score" in f_test.columns else None)

        corr_plot(f_test[args.score_col].values, e_ai,
                  "AI: DNABERT mean vs Game score",
                  out/"dnabert_mean_vs_game_AI.png",
                  args.score_col, "DNABERT mean")
        corr_plot(f_hum[args.score_col].values, e_hu,
                  "Human: DNABERT mean vs Game score",
                  out/"dnabert_mean_vs_game_Human.png",
                  args.score_col, "DNABERT mean")
        if des_col:
            corr_plot(f_test[des_col].values, e_ai,
                      "AI: DNABERT mean vs Design",
                      out/"dnabert_mean_vs_design_AI.png",
                      des_col, "DNABERT mean")
            corr_plot(f_hum[des_col].values, e_hu,
                      "Human: DNABERT mean vs Design",
                      out/"dnabert_mean_vs_design_Human.png",
                      des_col, "DNABERT mean")
        corr_plot(f_test["entropy"].values, e_ai,
                  "AI: DNABERT mean vs Entropy",
                  out/"dnabert_mean_vs_entropy_AI.png",
                  "entropy", "DNABERT mean")
        corr_plot(f_hum["entropy"].values, e_hu,
                  "Human: DNABERT mean vs Entropy",
                  out/"dnabert_mean_vs_entropy_Human.png",
                  "entropy", "DNABERT mean")

    # Entropy vs Design (Ben is very interested)
    des_col = "design_model" if "design_model" in f_test.columns else ("design_score" if "design_score" in f_test.columns else None)
    if des_col:
        corr_plot(f_test["entropy"].values, f_test[des_col].values,
                  "AI: Entropy vs Design", out/"AI_entropy_vs_design.png",
                  "entropy", des_col)
        corr_plot(f_hum["entropy"].values, f_hum[des_col].values,
                  "Human: Entropy vs Design", out/"Human_entropy_vs_design.png",
                  "entropy", des_col)

    print(f"[done] wrote figures to: {out}")

if __name__ == "__main__":
    main()
