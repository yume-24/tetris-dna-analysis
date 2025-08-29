#!/usr/bin/env python3
import argparse, csv, pathlib as P
import numpy as np, matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch


"""
python analysis/plots/ai_vs_human_dnabert6.py \
  --ai_fasta outputs/raw/seqs_main_ai.fasta \
  --human_fasta outputs/raw/seqs_main_human.fasta \
  --human_scores outputs/analysis/per_seq_human.csv \
  --metric score_raw --min 4 \
  --tsne \
  --outdir outputs/analysis/ai_vs_human_dnabert6

"""

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

def kmers(seq, k=6):
    s = seq.upper()
    return " ".join(s[i:i+k] for i in range(len(s)-k+1))

def embed_batch(texts, tok, model):
    embs = []
    with torch.no_grad():
        for t in texts:
            inputs = tok(t, return_tensors="pt", truncation=True)
            h = model(**inputs).last_hidden_state  # (1, T, H)
            embs.append(h.mean(dim=1).squeeze(0).cpu().numpy())  # mean-pool
    return np.vstack(embs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai_fasta", required=True)
    ap.add_argument("--human_fasta", required=True)
    ap.add_argument("--human_scores", required=True)  # CSV with metric column
    ap.add_argument("--metric", default="score_norm")
    ap.add_argument("--min", type=float, default=0.88)
    ap.add_argument("--outdir", default="outputs/analysis/dnabert6_embed")
    ap.add_argument("--tsne", action="store_true")
    args = ap.parse_args()

    out = P.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    ai = read_fasta(args.ai_fasta)
    hu = read_fasta(args.human_fasta)
    print(f"[load] AI={len(ai)}  Human(all)={len(hu)}")

    # load scores and filter humans
    import pandas as pd
    df = pd.read_csv(args.human_scores)
    if "seq_id" not in df.columns:
        print("[note] 'seq_id' missing; assuming rows align with Human FASTA order (1..N).")
        df["seq_id"] = np.arange(1, len(df)+1)
    keep = df[df[args.metric] >= args.min]["seq_id"].values
    keep = [i for i in keep if 1 <= i <= len(hu)]
    hu_f = [hu[i-1] for i in keep]
    print(f"[filter] metric='{args.metric}' >= {args.min}  → Human(filtered)={len(hu_f)}")

    # DNABERT (k=6) tokenizer & model (CPU)
    mname = "zhihan1996/DNA_bert_6"
    tok = AutoTokenizer.from_pretrained(mname)
    model = AutoModel.from_pretrained(mname).eval()

    ai_texts = [kmers(s, 6) for s in ai]
    hu_texts = [kmers(s, 6) for s in hu_f]

    EA = embed_batch(ai_texts, tok, model)
    EH = embed_batch(hu_texts, tok, model)

    # PCA
    X = np.vstack([EA, EH])
    Xc = X - X.mean(axis=0, keepdims=True)
    U,S,VT = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :2] * S[:2]
    Za, Zh = Z[:len(EA)], Z[len(EA):]

    fig, ax = plt.subplots(figsize=(7,6))
    ax.scatter(Za[:,0], Za[:,1], s=25, alpha=0.75, label=f"AI (n={len(EA)})")
    ax.scatter(Zh[:,0], Zh[:,1], s=25, alpha=0.75, label=f"Human(≥thr) (n={len(EH)})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("PCA on DNABERT(k=6) embeddings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"pca_dnABERT6.png", dpi=220)

    if args.tsne and X.shape[0] >= 10:
        try:
            from sklearn.manifold import TSNE
            Zt = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=30).fit_transform(X)
            Za, Zh = Zt[:len(EA)], Zt[len(EA):]
            fig, ax = plt.subplots(figsize=(7,6))
            ax.scatter(Za[:,0], Za[:,1], s=25, alpha=0.75, label=f"AI (n={len(EA)})")
            ax.scatter(Zh[:,0], Zh[:,1], s=25, alpha=0.75, label=f"Human(≥thr) (n={len(EH)})")
            ax.set_title("t-SNE on DNABERT(k=6) embeddings")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out/"tsne_dnABERT6.png", dpi=220)
        except Exception as e:
            print("[warn] t-SNE skipped:", e)

    print(f"[done] wrote figures to: {out}")

if __name__ == "__main__":
    main()
