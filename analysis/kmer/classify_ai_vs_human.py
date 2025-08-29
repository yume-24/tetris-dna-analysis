import argparse, os, itertools, numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

"""
run
python analysis/kmer/classify_ai_vs_human.py --k 4
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

def kmer_vocab(k):
    from itertools import product
    return ["".join(p) for p in product("ACGT", repeat=k)]

def kmer_vec(seq, k, vocab):
    cnt = Counter(seq[i:i+k] for i in range(len(seq)-k+1) if set(seq[i:i+k])<=set("ACGT"))
    total = max(1, sum(cnt.values()))
    return np.array([cnt[v]/total for v in vocab], float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--ai", default="outputs/raw/seqs_main_ai.fasta")
    ap.add_argument("--human", default="outputs/raw/seqs_main_human.fasta")
    ap.add_argument("--outdir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ai = read_fasta(args.ai); hu = read_fasta(args.human)
    vocab = kmer_vocab(args.k)

    X_ai = np.vstack([kmer_vec(s, args.k, vocab) for s in ai])
    X_hu = np.vstack([kmer_vec(s, args.k, vocab) for s in hu])
    X = np.vstack([X_ai, X_hu])
    y = np.array([1]*len(X_ai) + [0]*len(X_hu))

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=200)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    auc = cross_val_score(clf, Xs, y, cv=cv, scoring="roc_auc").mean()
    acc = cross_val_score(clf, Xs, y, cv=cv, scoring="accuracy").mean()

    clf.fit(Xs, y)
    coefs = pd.DataFrame({"kmer": vocab, "coef": clf.coef_[0]})
    coefs.to_csv(os.path.join(args.outdir, f"ai_human_k{args.k}_coefs.csv"), index=False)

    # top driving features
    top_ai = coefs.sort_values("coef", ascending=False).head(25)
    top_hu = coefs.sort_values("coef", ascending=True).head(25)
    top_ai.to_csv(os.path.join(args.outdir, f"ai_human_k{args.k}_topcoef_AI.csv"), index=False)
    top_hu.to_csv(os.path.join(args.outdir, f"ai_human_k{args.k}_topcoef_Human.csv"), index=False)

    # bar plot
    both = pd.concat([top_hu.assign(side="Human"), top_ai.assign(side="AI")])
    plt.figure(figsize=(8,6))
    idx = np.arange(len(both))
    plt.barh(idx, both["coef"], tick_label=both["kmer"])
    plt.axvline(0, lw=1)
    plt.gca().invert_yaxis()
    plt.title(f"Top {args.k}-mers separating AI (right) vs Human (left)\nAUC={auc:.3f}, Acc={acc:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"ai_human_k{args.k}_coef_bar.png"), dpi=140)
    plt.close()

    pd.DataFrame([{"k":args.k, "AUC":round(auc,4), "ACC":round(acc,4)}]).to_csv(
        os.path.join(args.outdir, f"ai_human_k{args.k}_clf_metrics.csv"), index=False)
    print("[OK] AUC=", auc, "ACC=", acc)

if __name__ == "__main__":
    main()
