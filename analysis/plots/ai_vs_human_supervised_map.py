# analysis/plots/ai_vs_human_supervised_map.py
"""
Supervised map & diagnostics in PWM-engineered space.

Examples
--------
python analysis/plots/ai_vs_human_supervised_map.py \
  --ai_fasta outputs/raw/seqs_main_ai.fasta \
  --human_fasta outputs/raw/seqs_main_human.fasta \
  --outdir outputs/analysis/pwm_supervised \
  --ppar motifs/MA0065.2.meme --nfkb motifs/MA0105.4.meme \
  --cv 5 --dump_profiles --ai_match_human --seed 7

Notes
-----
- Computes full sliding-window PWM *log-odds* profiles for PPARγ and NF-κB on both strands.
- Derives features: max logit, pos of max, count above threshold, mean of top-K, spacing between best sites,
  plus a probability-normalized "design_norm" (like your external scorer) so you can compare.
- Trains Logistic Regression (with z-scored features) using Stratified K-fold CV; reports ROC-AUC, accuracy,
  confusion matrix, and feature weights. Also fits 1D LDA and PC1 for a 2D map.
- Optionally dumps per-sequence PWM profiles to NPZ for deeper analysis.

Dependencies: numpy, pandas, matplotlib, scikit-learn
"""

import argparse, os, math, pathlib as P
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ALPH = "ACGT"
IDX = {c:i for i,c in enumerate(ALPH)}
RCMAP = str.maketrans("ACGT", "TGCA")

def revcomp(s: str) -> str:
    return s.translate(RCMAP)[::-1]

# ----------------------------- FASTA I/O --------------------------------------
def read_fasta(path: str) -> List[str]:
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

# --------------------------- MEME PWM parsing ---------------------------------
def parse_meme_pwm(path: str) -> np.ndarray:
    """Return probability matrix of shape (w,4) ordered A C G T."""
    rows, in_mat = [], False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.lower().startswith("letter-probability matrix"):
                in_mat = True
                continue
            if in_mat:
                if not s or s[0].isalpha():
                    break
                vals = [float(x) for x in s.split()]
                if len(vals) >= 4:
                    rows.append(vals[:4])
    if not rows:
        raise ValueError(f"No PWM read from {path}")
    pwm = np.array(rows, dtype=float)
    pwm = np.clip(pwm, 1e-9, 1.0)
    pwm /= pwm.sum(axis=1, keepdims=True)
    return pwm

def pwm_log_odds(pwm: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Convert prob PWM to log-odds matrix (natural log)."""
    return np.log(pwm / bg[None,:])

def max_possible_logprob(pwm: np.ndarray) -> float:
    """For probability-normalized score: sum over positions of log(max column prob)."""
    return float(np.log(pwm.max(axis=1)).sum())

# --------------------------- sliding-window scoring ---------------------------
@dataclass
class PWMProfile:
    forward: np.ndarray  # (L-w+1,) forward-strand log-odds
    reverse: np.ndarray  # (L-w+1,) reverse-strand log-odds
    best:    np.ndarray  # max across strands per window

def pwm_profile_logodds(seq: str, lo_mat: np.ndarray) -> PWMProfile:
    """Return per-window log-odds profiles on FWD/REV and the max across strands."""
    L = len(seq); w = lo_mat.shape[0]
    nwin = max(0, L - w + 1)
    fwd = np.full(nwin, -np.inf, dtype=float)
    rev = np.full(nwin, -np.inf, dtype=float)
    if nwin == 0:
        return PWMProfile(fwd, rev, np.maximum(fwd, rev))
    # pre-encode sequence to indices (-1 for non-ACGT)
    S = np.array([IDX.get(ch, -1) for ch in seq], dtype=int)
    Sr = np.array([IDX.get(ch, -1) for ch in revcomp(seq)], dtype=int)
    # forward
    for i in range(nwin):
        sub = S[i:i+w]
        if np.any(sub < 0):  # ambiguous bases
            continue
        fwd[i] = float(lo_mat[np.arange(w), sub].sum())
    # reverse
    for i in range(nwin):
        sub = Sr[i:i+w]
        if np.any(sub < 0):
            continue
        rev[i] = float(lo_mat[np.arange(w), sub].sum())
    return PWMProfile(fwd, rev, np.maximum(fwd, rev))

def topk_stats(vec: np.ndarray, k: int = 3) -> Tuple[float, float]:
    """Return (max, mean of top-k). Handles -inf gracefully."""
    if vec.size == 0:
        return float("-inf"), float("-inf")
    v = vec[~np.isinf(vec)]
    if v.size == 0:
        return float("-inf"), float("-inf")
    vmax = float(v.max())
    k = min(k, v.size)
    mtk = float(np.sort(v)[-k:].mean())
    return vmax, mtk

def count_above(vec: np.ndarray, thr: float) -> int:
    return int(np.sum(vec > thr))

# probability-normalized "design" (like your external scorer)
def pwm_max_norm_prob(seq: str, pwm: np.ndarray) -> float:
    L, w = len(seq), pwm.shape[0]
    if L < w: return 0.0
    max_log = max_possible_logprob(pwm)
    # forward
    best = -1e30
    S = np.array([IDX.get(ch, -1) for ch in seq], dtype=int)
    for i in range(L-w+1):
        sub = S[i:i+w]
        if np.any(sub < 0):
            continue
        best = max(best, float(np.log(pwm[np.arange(w), sub]).sum()))
    # reverse
    Sr = np.array([IDX.get(ch, -1) for ch in revcomp(seq)], dtype=int)
    for i in range(L-w+1):
        sub = Sr[i:i+w]
        if np.any(sub < 0):
            continue
        best = max(best, float(np.log(pwm[np.arange(w), sub]).sum()))
    if best <= -1e29:
        return 0.0
    return float(np.exp(best - max_log))

# ------------------------------- pipeline -------------------------------------
def build_features(
    seqs_ai: List[str],
    seqs_hu: List[str],
    pwm_ppar: np.ndarray,
    pwm_nfkb: np.ndarray,
    bg: np.ndarray,
    topk: int = 3,
    thr_lo: float = 0.0,
) -> pd.DataFrame:
    lo_ppar = pwm_log_odds(pwm_ppar, bg)
    lo_nfkb = pwm_log_odds(pwm_nfkb, bg)

    rows = []
    for label, seqs in (("AI", seqs_ai), ("Human", seqs_hu)):
        for i, s in enumerate(seqs, 1):
            prof_ppar = pwm_profile_logodds(s, lo_ppar)
            prof_nfkb = pwm_profile_logodds(s, lo_nfkb)

            ppar_max,  ppar_topk = topk_stats(prof_ppar.best, topk)
            nfkb_max,  nfkb_topk = topk_stats(prof_nfkb.best, topk)
            ppar_cnt = count_above(prof_ppar.best, thr_lo)
            nfkb_cnt = count_above(prof_nfkb.best, thr_lo)

            # best-site positions (index of window start). If all -inf, set NaN
            p_idx = int(np.argmax(prof_ppar.best)) if np.isfinite(ppar_max) else -1
            n_idx = int(np.argmax(prof_nfkb.best)) if np.isfinite(nfkb_max) else -1
            spacing = (p_idx - n_idx) if (p_idx >= 0 and n_idx >= 0) else np.nan

            # probability-normalized "design" (0..1)
            ppar_norm = pwm_max_norm_prob(s, pwm_ppar)
            nfkb_norm = pwm_max_norm_prob(s, pwm_nfkb)
            design_norm = (ppar_norm - nfkb_norm + 1.0) / 2.0

            rows.append(dict(
                dataset=label, seq_id=i, length=len(s),
                ppar_max_lo=ppar_max, ppar_topk_lo=ppar_topk, ppar_cnt=ppar_cnt,
                nfkb_max_lo=nfkb_max, nfkb_topk_lo=nfkb_topk, nfkb_cnt=nfkb_cnt,
                spacing=spacing,
                ppar_norm=ppar_norm, nfkb_norm=nfkb_norm, design_norm=design_norm,
            ))
    return pd.DataFrame(rows)

def plot_lda_pc(dfZ: pd.DataFrame, outpath: str):
    fig, ax = plt.subplots(figsize=(7.6,6.2))
    for name, col in (("AI","#1f77b4"), ("Human","#ff7f0e")):
        m = dfZ.dataset==name
        ax.scatter(dfZ.loc[m,"lda1"], dfZ.loc[m,"pc1"], s=35, alpha=0.8, label=f"{name} (n={m.sum()})")
    ax.set_xlabel("LDA1 (supervised separation)")
    ax.set_ylabel("PC1 (unsupervised variance)")
    ax.set_title("AI vs Human in PWM-engineered space")
    ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(outpath, dpi=220); plt.close()

def plot_coeffs(coefs: pd.Series, outpath: str, top: int = 12):
    c = coefs.reindex(coefs.abs().sort_values(ascending=False).index)[:top]
    fig, ax = plt.subplots(figsize=(7.6,4.2))
    ax.barh(range(len(c)), c.values)
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.index)
    ax.invert_yaxis()
    ax.set_xlabel("logistic weight (std. features)"); ax.set_title("Most-informative features")
    plt.tight_layout(); plt.savefig(outpath, dpi=220); plt.close()

def plot_roc(y_true, y_score, outpath: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    A = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6.2,5.6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC AUC={A:.3f}")
    ax.plot([0,1],[0,1], ls="--", lw=1, c="k")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("Logistic regression (CV) ROC")
    ax.legend(frameon=False); ax.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(outpath, dpi=220); plt.close()
    return A

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai_fasta", required=True)
    ap.add_argument("--human_fasta", required=True)
    ap.add_argument("--ppar", default="motifs/MA0065.2.meme")
    ap.add_argument("--nfkb", default="motifs/MA0105.4.meme")
    ap.add_argument("--bg", nargs=4, type=float, default=[0.25,0.25,0.25,0.25],
                    help="background A C G T probabilities")
    ap.add_argument("--topk", type=int, default=3, help="mean of top-k windows feature")
    ap.add_argument("--thr_lo", type=float, default=0.0,
                    help="count windows with log-odds > thr_lo")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--outdir", default="outputs/analysis/pwm_supervised")
    ap.add_argument("--dump_profiles", action="store_true",
                    help="save per-seq PWM profiles to NPZ")

    # NEW: match AI n to Human n + reproducible seed
    ap.add_argument("--ai_match_human", action="store_true",
                    help="Randomly subsample AI sequences to match the number of Human sequences.")
    ap.add_argument("--seed", type=int, default=7,
                    help="Random seed for any subsampling.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = P.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # Load sequences
    seqs_ai = read_fasta(args.ai_fasta)
    seqs_hu = read_fasta(args.human_fasta)
    print(f"[load] AI={len(seqs_ai)}  Human={len(seqs_hu)}")

    # Optional: match AI n to Human n
    if args.ai_match_human and len(seqs_ai) > len(seqs_hu):
        n_target = len(seqs_hu)
        idx = rng.choice(len(seqs_ai), size=n_target, replace=False)
        seqs_ai = [seqs_ai[i] for i in idx]
        print(f"[match] Subsampled AI from {len(idx)} to {n_target} to match Human n.")
    elif args.ai_match_human and len(seqs_ai) < len(seqs_hu):
        print(f"[match] AI ({len(seqs_ai)}) < Human ({len(seqs_hu)}); leaving as-is (no upsampling).")

    print(f"[sizes] Using AI={len(seqs_ai)}  Human={len(seqs_hu)}")

    # Load PWMs
    pwm_ppar = parse_meme_pwm(args.ppar)
    pwm_nfkb = parse_meme_pwm(args.nfkb)
    bg = np.array(args.bg, dtype=float); bg /= bg.sum()

    # Build features (+ optional profile dump)
    df = build_features(seqs_ai, seqs_hu, pwm_ppar, pwm_nfkb, bg,
                        topk=args.topk, thr_lo=args.thr_lo)
    df.to_csv(out/"features_pwm.csv", index=False)
    print("[save]", out/"features_pwm.csv")

    # ---------------- dump full sliding-window profiles (optional) -------------
    if args.dump_profiles:
        print("[profiles] dumping per-base PWM log-odds…")

        lo_ppar = pwm_log_odds(pwm_ppar, bg)
        lo_nfkb = pwm_log_odds(pwm_nfkb, bg)

        long_rows = []
        all_seq = []
        for label, seqs in (("AI", seqs_ai), ("Human", seqs_hu)):
            for i, s in enumerate(seqs, 1):
                prof_p = pwm_profile_logodds(s, lo_ppar)
                prof_n = pwm_profile_logodds(s, lo_nfkb)
                for motif, prof in (("PPARg", prof_p), ("NFkB", prof_n)):
                    for strand, vec in (("fwd", prof.forward),
                                        ("rev", prof.reverse),
                                        ("best", prof.best)):
                        if vec.size:
                            for pos, val in enumerate(vec):
                                if np.isfinite(val):
                                    long_rows.append((label, i, motif, strand, pos, float(val)))
                all_seq.append((label, i, prof_p, prof_n, len(s)))

        df_long = pd.DataFrame(long_rows,
                               columns=["dataset", "seq_id", "motif", "strand", "pos", "logit"])
        df_long.to_csv(out / "profiles_long.csv.gz", index=False, compression="gzip")
        print("[save]", out / "profiles_long.csv.gz")

        # (wide NPZ export kept as-is from original script)

    # ---------------- supervised & unsupervised maps -----------------
    feat_cols = [
        "ppar_max_lo","ppar_topk_lo","ppar_cnt",
        "nfkb_max_lo","nfkb_topk_lo","nfkb_cnt",
        "spacing",
        "ppar_norm","nfkb_norm","design_norm",
        "length",
    ]
    X = df[feat_cols].copy()
    X["spacing"] = X["spacing"].fillna(X["spacing"].mean())
    y = (df["dataset"] == "AI").astype(int).values

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X.values)

    lda = LDA(n_components=1)
    lda1 = lda.fit_transform(Xz, y).reshape(-1)
    pc1 = PCA(n_components=1).fit_transform(Xz).reshape(-1)
    df_map = pd.DataFrame({"dataset": df["dataset"], "lda1": lda1, "pc1": pc1})
    fig, ax = plt.subplots(figsize=(7.6,6.2))
    for name in ("AI","Human"):
        m = df_map.dataset==name
        ax.scatter(df_map.loc[m,"lda1"], df_map.loc[m,"pc1"], s=35, alpha=0.8, label=f"{name} (n={m.sum()})")
    ax.set_xlabel("LDA1 (supervised separation)")
    ax.set_ylabel("PC1 (unsupervised variance)")
    ax.set_title("AI vs Human in PWM-engineered space")
    ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out/"lda_pc_scatter.png", dpi=220); plt.close()
    print("[fig]", out/"lda_pc_scatter.png")

    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    clf = LogisticRegression(max_iter=2000, solver="liblinear")
    y_proba = cross_val_predict(clf, Xz, y, cv=skf, method="predict_proba")[:,1]
    y_pred  = (y_proba >= 0.5).astype(int)

    roc_auc = plot_roc(y, y_proba, out/"roc_cv.png")
    cm = confusion_matrix(y, y_pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(5.6,5.2))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Human","AI"])
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion matrix (CV probs, 0.5 threshold)")
    plt.tight_layout(); plt.savefig(out/"confusion_matrix.png", dpi=220); plt.close()
    print("[fig]", out/"roc_cv.png")
    print("[fig]", out/"confusion_matrix.png")

    clf_full = LogisticRegression(max_iter=2000, solver="liblinear")
    clf_full.fit(Xz, y)
    coefs = pd.Series(clf_full.coef_.ravel(), index=feat_cols)
    fig, ax = plt.subplots(figsize=(7.6,4.2))
    c = coefs.reindex(coefs.abs().sort_values(ascending=False).index)[:12]
    ax.barh(range(len(c)), c.values)
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.index)
    ax.invert_yaxis()
    ax.set_xlabel("logistic weight (std. features)"); ax.set_title("Most-informative features")
    plt.tight_layout(); plt.savefig(out/"coef_importance.png", dpi=220); plt.close()
    print("[fig]", out/"coef_importance.png")

    acc = (y_pred == y).mean()
    summary = pd.DataFrame([dict(
        n_AI=int((df.dataset=="AI").sum()),
        n_Human=int((df.dataset=="Human").sum()),
        cv_folds=args.cv,
        roc_auc=float(roc_auc),
        accuracy=float(acc),
    )])
    summary.to_csv(out/"supervised_summary.csv", index=False)
    print("[save]", out/"supervised_summary.csv")

if __name__ == "__main__":
    main()
