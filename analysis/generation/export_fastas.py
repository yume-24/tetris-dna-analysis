# analysis/export_fastas.py  (Python 3.9)
"""
runs the infer() model on AI games and human games --> dumps generated DNA sequences into fasta files.
run:
    python analysis/generation/export_fastas.py \
  --model model-mlp2.pt \
  --test tetris_test-5k.json \
  --human "tetris_data copy.json" \
  --N 2000 --L 200 --outdir outputs/raw

normalizes, writes fasta files(outputs/raw/seqs_main_ai.fasta, outputs/raw/seqs_main_human.fasta) , debugs
"""

import argparse, inspect, os, sys, io, re
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

# make sure we can import modules from the repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- DNA parsing helpers ------------------------------------------------------
# make CASE INSENSITIVE
DNA_LINE  = re.compile(r'^[ACGTRYSWKMBDHVN]+$', re.IGNORECASE)        # whole line is DNA
DNA_TOKEN = re.compile(r'[ACGTRYSWKMBDHVN]+', re.IGNORECASE)           # DNA tokens inside a line

def parse_sequences_from_stdout(text: str):
    """
    1) Prefer the 'All DNA strings generated:' block if present.
    2) Otherwise, collect any pure DNA-looking lines.
    3) Otherwise, greedily extract long DNA tokens from mixed lines.
    """
    seqs, in_block = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("All DNA strings generated"):
            in_block = True
            continue
        if in_block:
            if not s or s.startswith("Traceback"):
                break
            if DNA_LINE.match(s):
                seqs.append(s)
    if seqs:
        return seqs

    # Fallback 1: any line that looks like DNA
    for line in text.splitlines():
        s = line.strip()
        if DNA_LINE.match(s):
            seqs.append(s)
    if seqs:
        return seqs

    # Fallback 2: long DNA tokens from mixed lines
    tokens = []
    for line in text.splitlines():
        for tok in DNA_TOKEN.findall(line):
            tok = tok.upper()
            if len(tok) >= 30:
                tokens.append(tok)
    return tokens

def coerce_to_strings(objs):
    """
    Accepts many shapes:
      - list[str]
      - (scores, list[str])
      - objects with .sequence or .seq
      - or None (when infer() only prints)
    Returns: list[str]
    """
    if objs is None:
        return []

    if isinstance(objs, (list, tuple)) and objs and isinstance(objs[0], str):
        return list(objs)

    if isinstance(objs, (list, tuple)) and len(objs) == 2 and isinstance(objs[1], (list, tuple)):
        seqs = objs[1]
        if seqs and isinstance(seqs[0], str):
            return list(seqs)

    seqs = []
    try:
        for o in objs:
            s = getattr(o, "sequence", None) or getattr(o, "seq", None) or (o if isinstance(o, str) else None)
            if s is not None:
                seqs.append(str(s))
    except TypeError:
        pass
    return seqs

# --- import the right infer() -------------------------------------------------
def import_infer(which="sl"):
    if which == "sl":
        from train_model import infer
    else:
        from train_rl_new import infer
    return infer

# --- single canonical call_infer (captures stdout+stderr) ---------------------
def call_infer(infer_fn, traces, model, N, L):
    sig = inspect.signature(infer_fn)
    params = sig.parameters
    kw = {}

    for cand in ("traces_path","traces","json_path","data_path","games_path","test_path"):
        if cand in params: kw[cand] = traces; break
    for cand in ("model_path","model","ckpt","checkpoint","weights"):
        if cand in params: kw[cand] = model; break
    for cand in ("N","num_seqs","num_samples","n_samples","num"):
        if cand in params: kw[cand] = N; break
    for cand in ("L","length","seq_len","seq_length"):
        if cand in params: kw[cand] = L; break

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        res = infer_fn(**kw)

    seqs = coerce_to_strings(res)
    if not seqs:  # parse printed output
        out = buf.getvalue()
        seqs = parse_sequences_from_stdout(out)

    if not seqs:
        Path("outputs/debug").mkdir(parents=True, exist_ok=True)
        with open("outputs/debug/infer_stdout.txt", "w") as f:
            f.write(buf.getvalue())
        raise SystemExit("[export_fastas] infer() returned nothing I can parse into sequences. "
                         "Wrote raw output to outputs/debug/infer_stdout.txt")

    seqs = [s.strip().upper() for s in seqs if s]
    if N and len(seqs) > N:
        seqs = seqs[:N]
    return seqs

# --- I/O ----------------------------------------------------------------------
def write_fasta(path, seqs, tag):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, s in enumerate(seqs, 1):
            f.write(f">{tag}_{i}\n{s}\n")

# --- CLI ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model-cnn-sl.pt")
    ap.add_argument("--test",  default="tetris_test-5k.json")
    ap.add_argument("--human", default="tetris_data copy.json")
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--L", type=int, default=200)
    ap.add_argument("--outdir", default="outputs/raw")
    ap.add_argument("--which", choices=["sl","rl"], default="sl")
    args = ap.parse_args()

    infer_fn = import_infer(args.which)

    print("[INFO] exporting TEST (AI) sequences…")
    seqs_ai = call_infer(infer_fn, args.test, args.model, args.N, args.L)
    write_fasta(os.path.join(args.outdir, "seqs_main_ai.fasta"), seqs_ai, "MainAI")
    print("[OK] wrote", os.path.join(args.outdir, "seqs_main_ai.fasta"))

    print("[INFO] exporting HUMAN sequences…")
    seqs_hu = call_infer(infer_fn, args.human, args.model, args.N, args.L)
    write_fasta(os.path.join(args.outdir, "seqs_main_human.fasta"), seqs_hu, "MainHuman")
    print("[OK] wrote", os.path.join(args.outdir, "seqs_main_human.fasta"))

if __name__ == "__main__":
    main()

## just run this

# python analysis/export_fastas.py \
#   --model model-mlp2.pt \
#   --test tetris_test-5k.json \
#   --human "tetris_data copy.json" \
#   --N 2000 --L 200 --outdir outputs/raw