# analysis/plots/pwm_logit_embed.py
"""
Make PCA / t-SNE plots from PWM *logit* features (top-K window log-odds per motif).

Example:
python analysis/plots/pwm_logit_embed.py \
  --ai_fasta outputs/raw/seqs_main_ai.fasta \
  --human_fasta outputs/raw/seqs_main_human.fasta \
  --human_scores outputs/correlate_gcloud/per_seq_human.csv \
  --metric score_norm --min -1 \
  --ppar motifs/MA0065.2.meme \
  --nfkb motifs/MA0105.4.meme \
  --k_top 10 --tsne \
  --match_ai_to_human --seed 7 \
  --outdir outputs/analysis/pwm_logits_matched
"""

import argparse, math, pathlib as P
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------- FASTA I/O ----------
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

ALPH = "ACGT"; IDX = {c:i for i,c in enumerate(ALPH)}
RC = str.maketrans("ACGT","TGCA")
def revcomp(s): return s.translate(RC)[::-1]

# ---------- MEME PWM parsing ----------
def parse_meme_pwm(path):
    rows, in_mat = [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("letter-probability matrix"):
                in_mat = True; continue
            if in_mat:
                if (not line) or line[0].isalpha():
                    break
                vals = [float(x) for x in line.split()]
                if len(vals) >= 4: rows.append(vals[:4])
    if not rows:
        raise ValueError(f"No PWM read from {path}")
    pwm = np.array(rows, dtype=float)   # shape (w,4) ordered A C G T
    pwm = np.clip(pwm, 1e-9, 1.0)
    pwm /= pwm.sum(axis=1, keepdims=True)
    return pwm

# ---------- log-odds scanning ----------
def pwm_logodds_windows(seq, pwm, bg=None):
    """
    Returns per-window *max-of-strands* log-odds scores.
    log-odds = sum_j log( pwm[j, base] / bg[base] )
    """
    if bg is None:
        bg = np.array([0.25,0.25,0.25,0.25], dtype=float)
    L, w = len(seq), pwm.shape[0]
    if L < w:
        return np.array([], dtype=float)

    log_p = np.log(pwm) - np.log(bg[None,:])  # (w,4)

    def scan_one(s):
        scores = np.full(L-w+1, -1e30, dtype=float)
        for i in range(L-w+1):
            total = 0.0; ok = True
            for j,ch in enumerate(s[i:i+w]):
                a = IDX.get(ch, -1)
                if a < 0: ok=False; break
                total += log_p[j, a]
            if ok: scores[i] = total
        return scores

    fwd = scan_one(seq)
    rev = scan_one(revcomp(seq))
    return np.maximum(fwd, rev)  # best strand at each position

def topk_features(scores, k):
    """Return top-K values (descending). If fewer windows, pad with min."""
    if scores.size == 0:
        return np.full(k, -1e30, dtype=float)
    srt = np.sort(scores)[::-1]
    if srt.size >= k:
        return srt[:k]
    out = np.full(k, srt[-1], dtype=float)
    out[:srt.size] = srt
    return out

# ---------- plotting helpers ----------
def savefig(path):
    P.Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def build_pwm_logit_features(seq_list, pwm_ppar, pwm_nfkb, k_top, bg):
    F = []
    for s in seq_list:
        v1 = topk_features(pwm_logodds_windows(s, pwm_ppar, bg), k_top)
        v2 = topk_features(pwm_logodds_windows(s, pwm_nfkb, bg), k_top)
        F.append(np.concatenate([v1, v2]))
    return np.vstack(F) if F else np.zeros((0, 2*k_top), dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai_fasta", required=True)
    ap.add_argument("--human_fasta", required=True)
    ap.add_argument("--human_scores", required=True)  # CSV with metric + optional seq_id
    ap.add_argument("--metric", default="score_norm")
    ap.add_argument("--min", type=float, default=-1.0)
    ap.add_argument("--ppar", default="motifs/MA0065.2.meme")
    ap.add_argument("--nfkb", default="motifs/MA0105.4.meme")
    ap.add_argument("--k_top", type=int, default=10, help="top-K windows per motif")
    ap.add_argument("--tsne", action="store_true", help="run t-SNE")
    ap.add_argument("--match_ai_to_human", action="store_true",
                    help="Randomly subsample AI to match number of filtered Human sequences.")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for subsampling.")
    ap.add_argument("--outdir", default="outputs/analysis/pwm_logits")
    args = ap.parse_args()

    outdir = P.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load sequences
    ai = read_fasta(args.ai_fasta)
    hu = read_fasta(args.human_fasta)
    print(f"[load] AI={len(ai)}  Human(all)={len(hu)}")

    # Filter human by score threshold
    df_scores = pd.read_csv(args.human_scores)
    if "seq_id" not in df_scores.columns:
        print("[note] 'seq_id' missing; assuming rows align with Human FASTA order (1..N).")
        df_scores["seq_id"] = np.arange(1, len(df_scores)+1)
    keep_ids = df_scores[df_scores[args.metric] >= args.min]["seq_id"].values
    keep_ids = [i for i in keep_ids if 1 <= i <= len(hu)]
    hu_f = [hu[i-1] for i in keep_ids]
    print(f"[filter] metric='{args.metric}' >= {args.min}  → Human(filtered)={len(hu_f)}")
    if len(hu_f) == 0:
        raise SystemExit("[ERR] No human sequences left after filtering.")

    # Optionally subsample AI to match human n (before feature extraction)
    if args.match_ai_to_human:
        nH = len(hu_f)
        if len(ai) > nH:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(ai), size=nH, replace=False)
            ai = [ai[i] for i in idx]
            print(f"[match] Subsampled AI to n={len(ai)} to match Human (n={nH}).")
        else:
            print(f"[match] AI (n={len(ai)}) ≤ Human (n={nH}); no subsample.")

    # Load PWMs
    pwm_ppar = parse_meme_pwm(args.ppar)
    pwm_nfkb = parse_meme_pwm(args.nfkb)
    bg = np.array([0.25,0.25,0.25,0.25], dtype=float)

    # Build features
    FA = build_pwm_logit_features(ai,  pwm_ppar, pwm_nfkb, args.k_top, bg)
    FH = build_pwm_logit_features(hu_f, pwm_ppar, pwm_nfkb, args.k_top, bg)

    # Save features CSV
    cols = [f"ppar_top{i+1}" for i in range(args.k_top)] + [f"nfkb_top{i+1}" for i in range(args.k_top)]
    df_ai = pd.DataFrame(FA, columns=cols); df_ai.insert(0,"dataset","AI"); df_ai.insert(1,"seq_id", np.arange(1,len(ai)+1))
    df_hu = pd.DataFrame(FH, columns=cols); df_hu.insert(0,"dataset","Human"); df_hu.insert(1,"seq_id", keep_ids[:len(hu_f)])
    pd.concat([df_ai, df_hu], ignore_index=True).to_csv(outdir/"pwm_logit_features.csv", index=False)
    print(f"[ok] wrote {outdir/'pwm_logit_features.csv'}")

    # Standardize features (z-score) and PCA
    X = np.vstack([FA, FH])
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    Xz = (X - mu) / sd

    try:
        from sklearn.decomposition import PCA
        Z = PCA(n_components=2, random_state=0).fit_transform(Xz)
    except Exception:
        # fallback to numpy SVD
        Xc = Xz - Xz.mean(axis=0, keepdims=True)
        U,S,VT = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]

    Zai, Zhu = Z[:len(FA)], Z[len(FA):]

    fig, ax = plt.subplots(figsize=(7,6))
    ax.scatter(Zai[:,0], Zai[:,1], s=30, alpha=0.8, label=f"AI (n={len(FA)})")
    ax.scatter(Zhu[:,0], Zhu[:,1], s=30, alpha=0.8, label=f"Human(≥thr) (n={len(FH)})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(f"PCA on PWM logit features (top-{args.k_top})")
    ax.legend(frameon=False)
    savefig(outdir/"pca_pwm_logits.png")

    # t-SNE (optional)
    if args.tsne and X.shape[0] >= 10:
        try:
            from sklearn.manifold import TSNE
            perpl = max(5, min(30, (X.shape[0]-1)//3))  # safe perplexity
            Zt = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=perpl, random_state=0).fit_transform(Xz)
            Zai, Zhu = Zt[:len(FA)], Zt[len(FA):]
            fig, ax = plt.subplots(figsize=(7,6))
            ax.scatter(Zai[:,0], Zai[:,1], s=30, alpha=0.8, label=f"AI (n={len(FA)})")
            ax.scatter(Zhu[:,0], Zhu[:,1], s=30, alpha=0.8, label=f"Human(≥thr) (n={len(FH)})")
            ax.set_title(f"t-SNE on PWM logit features (top-{args.k_top})")
            ax.legend(frameon=False)
            savefig(outdir/"tsne_pwm_logits.png")
        except Exception as e:
            print("[warn] t-SNE skipped:", e)

    # Optional: boxplot of max logit per motif (with n in labels)
    if FA.shape[0] and FH.shape[0]:
        maxA = np.column_stack([FA[:, 0], FA[:, args.k_top]])  # AI: [PPARγ_top1, NF-κB_top1]
        maxH = np.column_stack([FH[:, 0], FH[:, args.k_top]])  # Human: same order
        labels = [f"AI (n={maxA.shape[0]})", f"Human (n={maxH.shape[0]})"]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].boxplot([maxA[:, 0], maxH[:, 0]], labels=labels, showmeans=True)
        ax[0].set_title("PPARγ max logit"); ax[0].set_ylabel("log-odds")
        ax[1].boxplot([maxA[:, 1], maxH[:, 1]], labels=labels, showmeans=True)
        ax[1].set_title("NF-κB max logit")
        savefig(outdir / "max_logit_boxplots.png")

        # design delta diagnostics
        d_ai = FA[:, 0] - FA[:, args.k_top]
        d_hu = FH[:, 0] - FH[:, args.k_top]
        pd.DataFrame({
            "dataset": ["AI"]*len(d_ai) + ["Human"]*len(d_hu),
            "design_delta": np.concatenate([d_ai, d_hu]),
            "ppar_top1": np.concatenate([FA[:,0], FH[:,0]]),
            "nfkb_top1": np.concatenate([FA[:,args.k_top], FH[:,args.k_top]])
        }).to_csv(outdir/"design_delta.csv", index=False)

        fig, ax = plt.subplots(figsize=(7,4))
        ax.boxplot([d_ai, d_hu], labels=[f"AI (n={len(d_ai)})", f"Human≥thr (n={len(d_hu)})"], showmeans=True)
        ax.set_ylabel("PPARγ_top1 − NF-κB_top1 (log-odds)")
        ax.set_title("DesignΔ (logit) by dataset")
        savefig(outdir/"design_delta_boxplot.png")

        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(d_ai, bins=30, histtype="step", density=True, label="AI")
        ax.hist(d_hu, bins=30, histtype="step", density=True, label="Human≥thr")
        ax.set_xlabel("designΔ (log-odds)"); ax.set_ylabel("density")
        ax.set_title("DesignΔ distribution"); ax.legend()
        savefig(outdir/"design_delta_hist.png")

        # quick 1-D probe
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            X1 = np.concatenate([d_ai, d_hu])[:,None]
            y  = np.array([0]*len(d_ai) + [1]*len(d_hu))  # 0=AI, 1=Human
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            auc = cross_val_score(LogisticRegression(solver="liblinear"), X1, y, cv=cv, scoring="roc_auc")
            print(f"[probe] 1-D logistic ROC-AUC: mean={auc.mean():.3f} ± {auc.std():.3f}")
        except Exception as e:
            print("[probe] logistic AUC skipped:", e)

    print(f"[done] Wrote plots to: {outdir}")

if __name__ == "__main__":
    main()
