# analysis/plots/filter_match_sets.py
import argparse, os
import numpy as np, pandas as pd
from pathlib import Path
"""
python analysis/plots/filter_match_sets.py \
  --ai_fasta   outputs/raw/seqs_main_ai.fasta \
  --ai_scores  outputs/analysis/per_seq_test.csv \
  --human_fasta   outputs/raw/seqs_main_human.fasta \
  --human_scores  outputs/analysis/per_seq_human.csv \
  --score_col score_raw \
  --human_min=4 --ai_min=-1e9 \
  --match equal_n --seed=7 \
  --outdir outputs/analysis/matched_raw4
"""

def read_fasta(path):
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s: continue
            if s.startswith(">"):
                if cur: seqs.append("".join(cur).upper()); cur = []
            else:
                cur.append(s)
    if cur: seqs.append("".join(cur).upper())
    return seqs

def write_fasta(path, seqs, tag):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w") as f:
        for i,s in enumerate(seqs,1):
            f.write(f">{tag}_{i}\n{s}\n")

SCORE_ALIASES = {
    "score_raw":"score_raw",
    "raw_score":"score_raw",
    "raw":"score_raw",
    "score_norm":"score_norm",
    "normalized":"score_norm",
    "norm":"score_norm",
    "score":"score_norm",
}

def load_scores(csv_path, wanted_col):
    df = pd.read_csv(csv_path)
    # normalize headers
    norm = {c:c.strip() for c in df.columns}
    df.rename(columns=norm, inplace=True)
    lc2orig = {c.lower(): c for c in df.columns}

    # resolve score column
    if wanted_col not in df.columns:
        # try case-insensitive + aliases
        cand = None
        for k,v in SCORE_ALIASES.items():
            if k.lower() in lc2orig and v == wanted_col:
                cand = lc2orig[k.lower()]
                break
        if cand is None:
            # last chance: direct case-insensitive match to wanted_col
            if wanted_col.lower() in lc2orig:
                cand = lc2orig[wanted_col.lower()]
        if cand is None:
            raise SystemExit(f"{csv_path}: cannot find score column '{wanted_col}'. "
                             f"Available: {list(df.columns)}")
        score_col = cand
    else:
        score_col = wanted_col

    # coerce numeric
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    # ensure seq_id (1-based) to align with FASTA order if needed
    if "seq_id" not in df.columns:
        df["seq_id"] = np.arange(1, len(df)+1, dtype=int)

    return df, score_col

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai_fasta",    required=True)
    ap.add_argument("--ai_scores",   required=True)
    ap.add_argument("--human_fasta", required=True)
    ap.add_argument("--human_scores",required=True)
    ap.add_argument("--score_col",   default="score_raw",
                    help="which column to threshold (e.g., score_raw or score_norm)")
    ap.add_argument("--human_min",   type=float, default=4.0)
    ap.add_argument("--ai_min",      type=float, default=-1e9)
    ap.add_argument("--match",       choices=["none","equal_n","quantile"], default="equal_n")
    ap.add_argument("--target_n",    type=int, default=None)
    ap.add_argument("--quantile_bins", type=int, default=10)
    ap.add_argument("--seed",        type=int, default=7)
    ap.add_argument("--outdir",      required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # Load FASTAs
    ai_seqs    = read_fasta(args.ai_fasta)
    human_seqs = read_fasta(args.human_fasta)

    # Load score CSVs (robust)
    ai_df,    score_col_ai = load_scores(args.ai_scores,   args.score_col)
    human_df, score_col_hu = load_scores(args.human_scores, args.score_col)

    # Align lengths if needed (in case CSVs longer than FASTA)
    ai_n    = min(len(ai_seqs),    len(ai_df))
    human_n = min(len(human_seqs), len(human_df))
    if ai_n < len(ai_df):    ai_df    = ai_df.iloc[:ai_n].copy()
    if human_n < len(human_df): human_df = human_df.iloc[:human_n].copy()

    # Threshold
    ai_mask    = ai_df[score_col_ai]    >= args.ai_min
    human_mask = human_df[score_col_hu] >= args.human_min
    ai_keep_idx    = ai_df.index[ai_mask].to_numpy()
    human_keep_idx = human_df.index[human_mask].to_numpy()

    print(f"[filter] AI >= {args.ai_min}: {len(ai_keep_idx)}   | Human >= {args.human_min}: {len(human_keep_idx)}")

    # Matching policy
    if args.match == "none":
        ai_idx    = ai_keep_idx
        human_idx = human_keep_idx
    elif args.match == "equal_n":
        m = min(len(ai_keep_idx), len(human_keep_idx))
        if args.target_n is not None:
            m = min(m, args.target_n)
        ai_idx    = rng.choice(ai_keep_idx,    size=m, replace=False) if m>0 else np.array([], int)
        human_idx = rng.choice(human_keep_idx, size=m, replace=False) if m>0 else np.array([], int)
    else:  # quantile
        # bin on the chosen score column for each side and sample equal counts per bin
        def stratified(indices, df, col, bins, n_per_bin):
            if len(indices)==0: return np.array([], int)
            vals = df.loc[indices, col].to_numpy()
            qs = np.quantile(vals, np.linspace(0,1,bins+1))
            out = []
            for b in range(bins):
                lo, hi = qs[b], qs[b+1]
                sel = indices[(vals>=lo) & (vals<=hi)] if b==bins-1 else indices[(vals>=lo) & (vals<hi)]
                if len(sel)>0:
                    take = min(n_per_bin, len(sel))
                    out.append(rng.choice(sel, size=take, replace=False))
            return np.concatenate(out) if out else np.array([], int)

        bins = args.quantile_bins
        nbin = min(len(ai_keep_idx), len(human_keep_idx), 100000) // max(bins,1)
        ai_idx    = stratified(ai_keep_idx,    ai_df,    score_col_ai, bins, nbin)
        human_idx = stratified(human_keep_idx, human_df, score_col_hu, bins, nbin)

    print(f"[match] AI={len(ai_idx)}  Human={len(human_idx)}")

    # Slice
    ai_out_df    = ai_df.iloc[ai_idx].copy()
    human_out_df = human_df.iloc[human_idx].copy()
    ai_out_seqs    = [ai_seqs[i] for i in ai_idx]
    human_out_seqs = [human_seqs[i] for i in human_idx]

    # Summaries
    def summary(df, col):
        x = df[col].astype(float)
        return dict(n=len(x), mean=np.nanmean(x), std=np.nanstd(x), min=np.nanmin(x), max=np.nanmax(x))
    s_ai = summary(ai_out_df, score_col_ai) if len(ai_out_df)>0 else dict(n=0, mean=np.nan, std=np.nan, min=np.nan, max=np.nan)
    s_hu = summary(human_out_df, score_col_hu) if len(human_out_df)>0 else dict(n=0, mean=np.nan, std=np.nan, min=np.nan, max=np.nan)
    print(f"[AI] {score_col_ai}  n={s_ai['n']}  mean={s_ai['mean']:.3f}  std={s_ai['std']:.3f}  min={s_ai['min']:.3f}  max={s_ai['max']:.3f}")
    print(f"[Human] {score_col_hu}  n={s_hu['n']}  mean={s_hu['mean']:.3f}  std={s_hu['std']:.3f}  min={s_hu['min']:.3f}  max={s_hu['max']:.3f}")

    # Write outputs
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    write_fasta(outdir/"ai_filtered.fasta", ai_out_seqs, "AI")
    write_fasta(outdir/"human_filtered.fasta", human_out_seqs, "Human")
    ai_out_df.to_csv(outdir/"ai_filtered.csv", index=False)
    human_out_df.to_csv(outdir/"human_filtered.csv", index=False)
    print("[OK] wrote:")
    print(" ", outdir/"ai_filtered.fasta")
    print(" ", outdir/"human_filtered.fasta")
    print(" ", outdir/"ai_filtered.csv")
    print(" ", outdir/"human_filtered.csv")

if __name__ == "__main__":
    main()
