# analysis/score_correlation.py
import argparse, io, os, re, sys
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import math


"""
RUN THIS 

python analysis/score_correlation.py \
  --model model-mlp2.pt \
  --test tetris_test-5k.json \
  --human "tetris_data copy.json" \
  --N 2000 --L 200 --outdir outputs/correlations

"""


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# regex for lines like:
# Raw Board Score: 292    Normalized: 0.995    ... Design: 0.681
LINE = re.compile(
    r"Raw Board Score:\s*(?P<raw>\d+)\s+Normalized:\s*(?P<norm>[0-9.]+).*?Design:\s*(?P<design>[0-9.]+)",
    re.IGNORECASE
)

def import_infer(which="sl"):
    if which == "sl":
        from train_model import infer
        return infer
    else:
        from train_rl_new import infer
        return infer

def call_infer_capture(infer_fn, traces_path, model_path, N=None, L=None):
    import inspect
    sig = inspect.signature(infer_fn)
    params = sig.parameters
    kw = {}
    for cand in ("traces_path","traces","json_path","data_path","games_path","test_path"):
        if cand in params: kw[cand] = traces_path; break
    for cand in ("model_path","model","ckpt","checkpoint","weights"):
        if cand in params: kw[cand] = model_path; break
    for cand in ("N","num_seqs","num_samples","n_samples","num"):
        if cand in params and N is not None: kw[cand] = N; break
    for cand in ("L","length","seq_len","seq_length"):
        if cand in params and L is not None: kw[cand] = L; break

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        infer_fn(**kw)
    return buf.getvalue()

def parse_pairs(text, filter_zero=True):
    pairs = []  # (raw, norm, design)
    for line in text.splitlines():
        m = LINE.search(line)
        if m:
            raw = int(m.group("raw"))
            norm = float(m.group("norm"))
            design = float(m.group("design"))
            if (not filter_zero) or (raw != 0 and norm != 0.0):
                pairs.append((raw, norm, design))
    return pairs

def pearson(x, y):
    n = len(x)
    if n < 2: return float('nan')
    mx = sum(x)/n; my = sum(y)/n
    num = sum((xi-mx)*(yi-my) for xi,yi in zip(x,y))
    denx = math.sqrt(sum((xi-mx)**2 for xi in x))
    deny = math.sqrt(sum((yi-my)**2 for yi in y))
    return num/(denx*deny) if denx>0 and deny>0 else float('nan')

def ranks(vals):
    # average ranks for ties
    order = sorted((v,i) for i,v in enumerate(vals))
    ranks = [0]*len(vals)
    i = 0
    while i < len(order):
        j = i
        while j+1 < len(order) and order[j+1][0] == order[i][0]:
            j += 1
        avg_rank = (i + j)/2 + 1
        for k in range(i, j+1):
            ranks[order[k][1]] = avg_rank
        i = j+1
    return ranks

def spearman(x, y):
    rx = ranks(x); ry = ranks(y)
    return pearson(rx, ry)

def write_csv(path, rows, header=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if header:
            f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model-mlp2.pt")
    ap.add_argument("--test",  default="tetris_test-5k.json")
    ap.add_argument("--human", default="tetris_data copy.json")
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--L", type=int, default=200)
    ap.add_argument("--which", choices=["sl","rl"], default="sl")
    ap.add_argument("--outdir", default="outputs/correlations")
    ap.add_argument("--keep_zero", action="store_true",
                    help="Keep rows where game score is zero (human data often has zeros).")
    args = ap.parse_args()

    infer_fn = import_infer(args.which)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # TEST / AI
    print("[INFO] running infer on TEST (AI)…")
    text_ai = call_infer_capture(infer_fn, args.test, args.model, args.N, args.L)
    pairs_ai = parse_pairs(text_ai, filter_zero=(not args.keep_zero))
    write_csv(outdir/"pairs_test.csv", pairs_ai, header=["raw_score","normalized_score","design_score"])

    # HUMAN
    print("[INFO] running infer on HUMAN…")
    text_hu = call_infer_capture(infer_fn, args.human, args.model, args.N, args.L)
    pairs_hu = parse_pairs(text_hu, filter_zero=(not args.keep_zero))
    write_csv(outdir/"pairs_human.csv", pairs_hu, header=["raw_score","normalized_score","design_score"])

    # correlations
    def corr_rows(name, pairs):
        if not pairs: return [[name,0,"nan","nan","nan","nan"]]
        raw = [p[0] for p in pairs]
        norm = [p[1] for p in pairs]
        des = [p[2] for p in pairs]
        return [[
            name, len(pairs),
            pearson(raw, des), spearman(raw, des),
            pearson(norm, des), spearman(norm, des),
        ]]

    rows = [["dataset","n","pearson_raw_vs_design","spearman_raw_vs_design",
             "pearson_norm_vs_design","spearman_norm_vs_design"]]
    rows += corr_rows("TEST_AI", pairs_ai)
    rows += corr_rows("HUMAN", pairs_hu)
    write_csv(outdir/"summary.csv", rows)

    print("[OK] wrote", outdir)
