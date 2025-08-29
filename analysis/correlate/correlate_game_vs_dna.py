# analysis/correlate_game_vs_dna.py  (Python 3.9 OK)
import argparse, io, os, re, sys
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import math
import numpy as np
import pandas as pd

"""
RUN THIS 
python analysis/correlate_game_vs_dna.py \
  --model model-mlp2.pt \
  --test tetris_test-5k.json \
  --human "tetris_data copy.json" \
  --N 2000 --L 200 --min_score 1 \
  --outdir outputs/analysis

"""

# make repo root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- import the same infer you’ve been using ---
def import_infer(which="sl"):
    if which == "sl":
        from train_model import infer
        return infer
    else:
        from train_rl_new import infer
        return infer

# --- parse the model's stdout ---
RX_RAW   = re.compile(r"Raw Board Score:\s*([-\d.eE]+)")
RX_NORM  = re.compile(r"Normalized:\s*([-\d.eE]+)")
RX_PPAR  = re.compile(r"PPAR", re.I)           # presence only (optional)
RX_NFKB  = re.compile(r"NF", re.I)             # presence only (optional)
RX_DES   = re.compile(r"Design:\s*([-\d.eE]+)")
RX_DNA   = re.compile(r"DNA string .*:\s*([ACGTNacgtn]+)")

def parse_infer_stdout(text: str):
    rows = []
    # The model prints repeated blocks. We’ll walk line-by-line and
    # collect fields until we see a DNA string (end of block).
    cur = {}
    for line in text.splitlines():
        m = RX_RAW.search(line);   cur["score_raw"]  = float(m.group(1))  if m else cur.get("score_raw")
        m = RX_NORM.search(line);  cur["score_norm"] = float(m.group(1))  if m else cur.get("score_norm")
        m = RX_DES.search(line);   cur["design"]     = float(m.group(1))  if m else cur.get("design")
        m = RX_DNA.search(line)
        if m:
            cur["seq"] = m.group(1).strip().upper()
            # only commit if we have both score and design
            if all(k in cur for k in ("score_raw","score_norm","design")):
                rows.append(cur)
            cur = {}
    return pd.DataFrame(rows)

# simple Spearman without SciPy
def spearmanr(x, y):
    x = np.asarray(x); y = np.asarray(y)
    # ranks with average for ties
    def rankdata(a):
        order = a.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a))
        # average ties
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=ranks)
        avg = sums / counts
        return avg[inv]
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0: return np.nan
    return np.corrcoef(rx, ry)[0,1]

def run_once(infer_fn, games_path, model_path, N, L):
    # Map common arg names
    import inspect
    sig = inspect.signature(infer_fn)
    kw = {}
    for cand in ("traces_path","traces","json_path","data_path","games_path","test_path"):
        if cand in sig.parameters: kw[cand] = games_path; break
    for cand in ("model_path","model","ckpt","checkpoint","weights"):
        if cand in sig.parameters: kw[cand] = model_path; break
    for cand in ("N","num_seqs","num_samples","n_samples","num"):
        if cand in sig.parameters: kw[cand] = N; break
    for cand in ("L","length","seq_len","seq_length"):
        if cand in sig.parameters: kw[cand] = L; break

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        infer_fn(**kw)
    out = buf.getvalue()
    df = parse_infer_stdout(out)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model-mlp2.pt")
    ap.add_argument("--test",  default="tetris_test-5k.json")
    ap.add_argument("--human", default="tetris_data copy.json")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--L", type=int, default=200)
    ap.add_argument("--which", choices=["sl","rl"], default="sl")
    ap.add_argument("--min_score", type=float, default=None, help="filter out games with raw score < this")
    ap.add_argument("--outdir", default="outputs/analysis")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    infer_fn = import_infer(args.which)

    print("[INFO] Running TEST (AI) …")
    df_test  = run_once(infer_fn, args.test,  args.model, args.N, args.L)
    print("[INFO] Running HUMAN …")
    df_human = run_once(infer_fn, args.human, args.model, args.N, args.L)

    # optional: drop zero/low-score humans
    if args.min_score is not None:
        df_test  = df_test[df_test["score_raw"]  >= args.min_score].copy()
        df_human = df_human[df_human["score_raw"] >= args.min_score].copy()

    # compute correlations
    def corr_block(df, label):
        if len(df)==0:
            return dict(dataset=label, n=0, pearson_raw_vs_design=np.nan,
                        spearman_raw_vs_design=np.nan,
                        pearson_norm_vs_design=np.nan,
                        spearman_norm_vs_design=np.nan)
        pr = np.corrcoef(df["score_raw"],  df["design"])[0,1] if df["score_raw"].std()>0 and df["design"].std()>0 else np.nan
        pn = np.corrcoef(df["score_norm"], df["design"])[0,1] if df["score_norm"].std()>0 and df["design"].std()>0 else np.nan
        sr = spearmanr(df["score_raw"].values,  df["design"].values)
        sn = spearmanr(df["score_norm"].values, df["design"].values)
        return dict(dataset=label, n=len(df),
                    pearson_raw_vs_design=pr, spearman_raw_vs_design=sr,
                    pearson_norm_vs_design=pn, spearman_norm_vs_design=sn)

    summary = pd.DataFrame([corr_block(df_test,"TEST_AI"), corr_block(df_human,"HUMAN")])
    summary_path = Path(args.outdir, "correlation_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("[OK] wrote", summary_path)

    # save per-seq tables too
    df_test.to_csv(Path(args.outdir, "per_seq_test.csv"), index=False)
    df_human.to_csv(Path(args.outdir, "per_seq_human.csv"), index=False)

    # quick scatter plots
    import matplotlib.pyplot as plt
    for name, df in [("TEST_AI", df_test), ("HUMAN", df_human)]:
        if len(df)==0: continue
        plt.figure()
        plt.scatter(df["score_norm"], df["design"], s=10, alpha=0.6)
        plt.xlabel("Normalized game score")
        plt.ylabel("DNA Design score")
        plt.title(f"{name}: score_norm vs design (n={len(df)})")
        plt.tight_layout()
        outp = Path(args.outdir, f"scatter_norm_vs_design_{name}.png")
        plt.savefig(outp, dpi=150)
        plt.close()
        print("[OK] wrote", outp)

if __name__ == "__main__":
    main()
