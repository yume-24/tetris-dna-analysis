# analysis/generation/export_fastas.py  (Python 3.9)
"""
Run the model's infer() on AI and Human games, dump DNA to FASTA.

Examples:
  python analysis/generation/export_fastas.py \
    --which sl_cnn \
    --model model-cnn-sl.pt \
    --test  tetris_test.json \
    --human gcloud_data.json \
    --N 5000 \
    --outdir outputs/raw

  # only keep Human entries with score >= 400 (change threshold as needed)
  python analysis/generation/export_fastas.py \
    --which sl_cnn \
    --model model-cnn-sl.pt \
    --test  tetris_test.json \
    --human gcloud_data.json \
    --human_min_score 400 --score_key score \
    --N 5000 \
    --outdir outputs/raw
"""
import argparse, inspect, sys, io, re, json, gzip
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, Iterable, Dict, Any, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- DNA parsing --------------------------------------------------------------
DNA_LINE  = re.compile(r'^[ACGTRYSWKMBDHVN]+$', re.IGNORECASE)
DNA_TOKEN = re.compile(r'[ACGTRYSWKMBDHVN]+', re.IGNORECASE)

def _open_any(path: str):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, "rt")
    return open(p, "r")

def _detect_and_load(path: str) -> Tuple[str, Any]:
    """
    Return (fmt, obj/descriptor):
      fmt in {"array","dict_of_lists","dict_of_dicts","jsonl"}
      - array: obj is list
      - dict_of_lists: obj is dict (values lists)
      - dict_of_dicts: obj is dict (values dicts)
      - jsonl: obj is the file path (we stream lines)
    """
    with _open_any(path) as f:
        first = f.read(1)
        f.seek(0)
        if first in ("[", "{"):
            obj = json.load(f)
            if isinstance(obj, list):
                return "array", obj
            if isinstance(obj, dict):
                has_list = any(isinstance(v, list) for v in obj.values())
                has_dict = any(isinstance(v, dict) for v in obj.values())
                if has_dict and not has_list:
                    return "dict_of_dicts", obj
                if has_list:
                    return "dict_of_lists", obj
                # fallback guess
                return "dict_of_dicts", obj
        # else JSONL
        return "jsonl", path

def json_len(path: str) -> Optional[int]:
    try:
        fmt, obj = _detect_and_load(path)
        if fmt == "array":
            return len(obj)
        if fmt == "dict_of_lists":
            return max((len(v) for v in obj.values() if isinstance(v, list)), default=None)
        if fmt == "dict_of_dicts":
            return len(obj)
        if fmt == "jsonl":
            with _open_any(obj) as f:
                return sum(1 for ln in f if ln.strip())
    except Exception:
        return None

def iter_rows(path: str) -> Iterable[Dict[str, Any]]:
    """Yield dict rows from any supported format."""
    fmt, obj = _detect_and_load(path)
    if fmt == "array":
        for r in obj:
            if isinstance(r, dict): yield r
    elif fmt == "dict_of_lists":
        lists = {k:v for k,v in obj.items() if isinstance(v, list)}
        if not lists: return
        L = max(len(v) for v in lists.values())
        keys = list(lists.keys())
        for i in range(L):
            row = {}
            for k in keys:
                li = lists[k]
                if i < len(li): row[k] = li[i]
            if row: yield row
    elif fmt == "dict_of_dicts":
        for _, v in obj.items():
            if isinstance(v, dict): yield v
    else:  # jsonl
        with _open_any(obj) as f:
            for ln in f:
                s = ln.strip()
                if not s: continue
                try:
                    r = json.loads(s)
                    if isinstance(r, dict): yield r
                except Exception:
                    continue

def normalize_for_model(src: str,
                        dst: str,
                        score_key: str = "score",
                        min_score: Optional[float] = None) -> Tuple[str, int]:
    """
    Create a normalized JSON array at `dst` with entries like:
      {"game_matrix": <2D list>, "score": <number>, ...}
    mapping common alternatives:
      - "board" -> "game_matrix"
    Applies an optional score filter.
    Returns (dst_path, kept_count).
    """
    out = []
    total = 0
    for row in iter_rows(src):
        total += 1
        # score filter
        try:
            s = float(row.get(score_key, -1e30))
        except Exception:
            continue
        if (min_score is not None) and (s < float(min_score)):
            continue
        # normalize matrix key
        gm = None
        if "game_matrix" in row and isinstance(row["game_matrix"], list):
            gm = row["game_matrix"]
        elif "board" in row and isinstance(row["board"], list):
            gm = row["board"]
        elif "matrix" in row and isinstance(row["matrix"], list):
            gm = row["matrix"]
        # skip if we didn't find a matrix
        if gm is None:
            continue
        entry = dict(row)  # keep other fields if present
        entry["game_matrix"] = gm
        entry["score"] = s
        out.append(entry)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f)
    kept = len(out)
    print(f"[normalize] {src}: total={total}  kept={kept}  -> {dst}")
    return dst, kept

def parse_sequences_from_stdout(text: str):
    seqs, in_block = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("All DNA strings generated"):
            in_block = True; continue
        if in_block:
            if not s or s.startswith("Traceback"): break
            if DNA_LINE.match(s): seqs.append(s)
    if seqs: return seqs
    for line in text.splitlines():
        s = line.strip()
        if DNA_LINE.match(s): seqs.append(s)
    if seqs: return seqs
    toks = []
    for line in text.splitlines():
        for tok in DNA_TOKEN.findall(line):
            tok = tok.upper()
            if len(tok) >= 30: toks.append(tok)
    return toks

def coerce_to_strings(objs):
    if objs is None: return []
    if isinstance(objs, (list, tuple)) and (not objs or isinstance(objs[0], str)):
        return list(objs)
    if isinstance(objs, (list, tuple)) and len(objs) == 2 and isinstance(objs[1], (list, tuple)):
        seqs = objs[1]
        if seqs and isinstance(seqs[0], str): return list(seqs)
    seqs = []
    try:
        for o in objs:
            s = getattr(o, "sequence", None) or getattr(o, "seq", None) or (o if isinstance(o, str) else None)
            if s is not None: seqs.append(str(s))
    except TypeError:
        pass
    return seqs

def import_infer(which: str, model_path: str):
    import importlib
    if which == "sl_cnn":
        tm = importlib.import_module("train_model_cnn")
    elif which == "sl":
        tm = importlib.import_module("train_model")
    elif which == "rl":
        tm = importlib.import_module("train_rl_new")
    else:
        raise ValueError(f"unknown --which {which}")
    if hasattr(tm, "STATE_PATH"):
        tm.STATE_PATH = model_path
    return tm.infer

def call_infer(infer_fn, traces_path: str, model_path: str, N: int, L: int):
    sig = inspect.signature(infer_fn)
    params = sig.parameters
    kw = {}

    # games/data path
    for cand in ("data_path","json_path","test_path","traces_path","games_path","traces"):
        if cand in params: kw[cand] = traces_path; break

    # number to generate
    for cand in (
        "n","N","num","count","limit","num_seqs","num_samples","n_samples",
        "num_to_generate","n_generate","n_outputs","n_out","samples","sample_n",
        "max_n","max_seqs","top_n","k"
    ):
        if cand in params: kw[cand] = N; break

    # model path (only if accepted; CNN uses STATE_PATH)
    for cand in ("model_path","model","ckpt","checkpoint","weights","state_path","state"):
        if cand in params: kw[cand] = model_path; break

    # length (often ignored by CNN)
    for cand in ("L","length","seq_len","seq_length"):
        if cand in params: kw[cand] = L; break

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        res = infer_fn(**kw)

    seqs = coerce_to_strings(res) or parse_sequences_from_stdout(buf.getvalue())
    seqs = [s.strip().upper() for s in seqs if s]
    if N and len(seqs) > N: seqs = seqs[:N]
    if not seqs:
        Path("outputs/debug").mkdir(parents=True, exist_ok=True)
        with open("outputs/debug/infer_stdout.txt","w") as f: f.write(buf.getvalue())
        raise SystemExit("[export_fastas] infer() returned nothing parseable. See outputs/debug/infer_stdout.txt")
    return seqs

def write_fasta(path, seqs, tag):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, s in enumerate(seqs, 1):
            f.write(f">{tag}_{i}\n{s}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["sl","sl_cnn","rl"], default="sl_cnn")
    ap.add_argument("--model", default="model-cnn-sl.pt")
    ap.add_argument("--test",  required=True, help="AI/Test games json/jsonl(.gz)")
    ap.add_argument("--human", required=True, help="Human games json/jsonl(.gz), supports dict-of-dicts (e.g., gcloud_data.json)")
    ap.add_argument("--N", type=int, default=2000, help="max sequences to export per split")
    ap.add_argument("--L", type=int, default=200)

    ap.add_argument("--human_min_score", type=float, default=None,
                    help="if set, filter Human rows on score_key >= this value before infer()")
    ap.add_argument("--score_key", default="score",
                    help="score field name in Human rows (default: score)")

    ap.add_argument("--outdir", default="outputs/raw")
    args = ap.parse_args()

    infer_fn = import_infer(args.which, args.model)
    print(f"[{args.which.upper()}] Using STATE_PATH={args.model}")

    ai_total = json_len(args.test)
    hu_total = json_len(args.human)
    print(f"[detect] AI count={ai_total if ai_total is not None else '?'} | "
          f"Human count={hu_total if hu_total is not None else '?'}")

    # Normalize AI test only if needed (usually already array w/ game_matrix)
    norm_ai_path = Path(args.outdir) / "ai_normalized.json"
    ai_norm_path, ai_norm_n = normalize_for_model(args.test, str(norm_ai_path))

    # Normalize Human (and optionally filter by score)
    norm_hu_path = Path(args.outdir) / ("human_normalized.json" if args.human_min_score is None
                                        else f"human_normalized_ge{args.human_min_score}.json")
    hu_norm_path, hu_norm_n = normalize_for_model(args.human, str(norm_hu_path),
                                                  score_key=args.score_key,
                                                  min_score=args.human_min_score)

    n_ai = min(args.N, ai_norm_n)
    n_hu = min(args.N, hu_norm_n)
    print(f"[counts] normalized: AI={ai_norm_n} | Human={hu_norm_n}  "
          f"→ exporting AI n={n_ai}, Human n={n_hu}")

    print("[INFO] exporting TEST (AI) sequences…")
    seqs_ai = call_infer(infer_fn, ai_norm_path, args.model, n_ai, args.L)
    write_fasta(Path(args.outdir) / "seqs_main_ai.fasta", seqs_ai, "MainAI")
    print("[OK] wrote", Path(args.outdir) / "seqs_main_ai.fasta")

    print("[INFO] exporting HUMAN sequences…")
    seqs_hu = call_infer(infer_fn, hu_norm_path, args.model, n_hu, args.L)
    write_fasta(Path(args.outdir) / "seqs_main_human.fasta", seqs_hu, "MainHuman")
    print("[OK] wrote", Path(args.outdir) / "seqs_main_human.fasta")
    print(f"[DONE] exported AI={len(seqs_ai)}  Human={len(seqs_hu)}")

if __name__ == "__main__":
    main()
