# analysis/correlate/correlate_game_vs_dna.py
"""
Correlate game score vs internal design score for AI (TEST) and optional HUMAN.

- AI/TEST is always produced by calling your model's infer() and parsing stdout.
- HUMAN:
    * If the file looks like the "gcloud" dict format
      { "<id>.json": {board, score, bioScore, dna}, ... }
      we read that directly (design_model := bioScore, score_norm := min–max of 'score').
    * Otherwise we also call infer() on the HUMAN JSON.

Outputs:
  per_seq_test.csv, per_seq_human.csv (if given),
  scatter_* PNGs, and correlation_summary.csv

Works on Python 3.9 (no PEP604 types).
"""

import argparse, io, inspect, json, re, sys
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# --- add near the top ---
import json, os, tempfile, pathlib as P

def _json_len(path):
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return len(obj)
    if isinstance(obj, dict):
        return len(obj.keys())
    return 0

def _flatten_gcloud_json(in_path: str) -> str:
    """
    Accepts a mapping like { "id.json": {board: .., score: .., dna: ..}, ... }
    and emits a list of entries the model expects: [{"game_matrix": .., "score": ..}, ...]
    Returns the path to a temp JSON file.
    """
    with open(in_path) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        # already good
        return in_path

    # mapping -> list
    out = []
    for _, v in obj.items():
        if not isinstance(v, dict):
            continue
        board = v.get("game_matrix", v.get("board"))
        score = v.get("score")
        if board is None or score is None:
            continue
        out.append({"game_matrix": board, "score": score})
    if not out:
        raise SystemExit(f"[ERR] gcloud_data had no usable entries: {in_path}")

    tmp = P.Path(tempfile.gettempdir()) / "gcloud_flat.json"
    with open(tmp, "w") as g:
        json.dump(out, g)
    return str(tmp)


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------- robust regex parsers for infer() stdout ----------
RX_SCORE_RAW   = re.compile(r"(?:Raw\s*Board\s*Score|score_raw|raw)\s*[:=]\s*([-\d.eE]+)", re.I)
RX_SCORE_NORM  = re.compile(r"(?:Normalized|score_norm|norm)\s*[:=]\s*([-\d.eE]+)", re.I)
RX_DESIGN      = re.compile(r"(?:DESIGN_MODEL|Design(?:\s*score)?)\s*[:=]\s*([-\d.eE]+)", re.I)
RX_DNA         = re.compile(r"^[ACGT]+$", re.I)

def parse_infer_stdout(text: str) -> pd.DataFrame:
    """
    Parse many entries from infer() stdout. We create a new row whenever we have
    BOTH score_norm and design_model. DNA lines (if printed) are attached to the
    most recent row, but are not required.
    """
    rows = []
    cur = {"score_raw": None, "score_norm": None, "design_model": None, "seq": None}

    def push_if_complete():
        if cur["score_norm"] is not None and cur["design_model"] is not None:
            rows.append(cur.copy())
            cur["score_raw"] = None
            cur["score_norm"] = None
            cur["design_model"] = None
            cur["seq"] = None

    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue

        # metrics first
        m = RX_SCORE_RAW.search(s)
        if m:
            cur["score_raw"] = float(m.group(1))
        m = RX_SCORE_NORM.search(s)
        if m:
            cur["score_norm"] = float(m.group(1))
        m = RX_DESIGN.search(s)
        if m:
            cur["design_model"] = float(m.group(1))

        # if metrics complete, emit a row immediately (handles logs with no DNA)
        if cur["score_norm"] is not None and cur["design_model"] is not None:
            # attach DNA if the next line prints it; otherwise we'll already emit here
            push_if_complete()
            continue

        # optional DNA line
        if RX_DNA.fullmatch(s):
            cur["seq"] = s.upper()
            # if metrics were already complete, this would have been pushed above.

    # final safety push
    push_if_complete()

    if not rows:
        raise ValueError("Could not parse any rows from infer() output.")
    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)

# ---------- import and call infer() ----------
def import_infer(which: str, model_path: str):
    import importlib
    if which == "sl_cnn":
        mod = importlib.import_module("train_model_cnn")
    elif which == "sl":
        mod = importlib.import_module("train_model")
    elif which == "rl":
        mod = importlib.import_module("train_rl_new")
    else:
        raise ValueError(f"unknown --which {which}")

    if hasattr(mod, "STATE_PATH"):
        mod.STATE_PATH = model_path
    return mod.infer

def run_infer(infer_fn, json_path: str, model_path: str, N: int, L: Optional[int] = None,
              debug_stdout_path: Optional[Path] = None) -> pd.DataFrame:
    sig = inspect.signature(infer_fn)
    params = sig.parameters
    kw = {}

    for cand in ("data_path","json_path","test_path","traces_path","games_path","traces"):
        if cand in params:
            kw[cand] = json_path
            break

    for cand in ("n","N","num","count","num_seqs","num_samples","n_samples","top_n","limit","k"):
        if cand in params:
            kw[cand] = int(N)
            break

    for cand in ("model_path","model","ckpt","checkpoint","weights","state_path","state"):
        if cand in params:
            kw[cand] = model_path
            break

    if L is not None:
        for cand in ("L","length","seq_len","seq_length"):
            if cand in params:
                kw[cand] = int(L)
                break

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        _ = infer_fn(**kw)

    text = buf.getvalue()
    if debug_stdout_path is not None:
        debug_stdout_path.parent.mkdir(parents=True, exist_ok=True)
        debug_stdout_path.write_text(text)

    df = parse_infer_stdout(text)
    return df

# ---------- “gcloud” JSON loader ----------
def is_gcloud_json(obj) -> bool:
    if isinstance(obj, dict) and obj:
        v = next(iter(obj.values()))
        return isinstance(v, dict) and ("score" in v or "bioScore" in v or "dna" in v)
    return False

def load_gcloud_human(path: str, N: Optional[int] = None) -> pd.DataFrame:
    with open(path) as f:
        obj = json.load(f)
    if not is_gcloud_json(obj):
        raise ValueError("File does not look like gcloud format (dict of entries).")

    rows = []
    for _, rec in obj.items():
        if not isinstance(rec, dict):
            continue
        dna  = rec.get("dna")
        score = rec.get("score")
        bios = rec.get("bioScore")
        if dna is None or score is None or bios is None:
            continue
        rows.append({
            "seq": str(dna).upper(),
            "score_raw": float(score),
            "design_model": float(bios)
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No usable human rows (need dna+score+bioScore).")

    smin, smax = df["score_raw"].min(), df["score_raw"].max()
    df["score_norm"] = (df["score_raw"] - smin) / max(1e-12, (smax - smin))

    if N is not None and len(df) > N:
        df = df.sort_values("score_raw", ascending=False).head(N).reset_index(drop=True)
    return df

# ---------- plotting ----------
def scatter_with_fit(df: pd.DataFrame, xcol: str, ycol: str, title: str, outpng: Path):
    if len(df) < 2:
        raise ValueError(f"Not enough points to plot: {len(df)}")

    x = df[xcol].values
    y = df[ycol].values
    n = len(df)

    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)

    m, b = np.polyfit(x, y, 1)
    xv = np.linspace(x.min(), x.max(), 200)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(x, y, s=25, alpha=0.8)
    ax.plot(xv, m*xv + b, lw=2)
    ax.set_xlabel(xcol.replace("_"," "))
    ax.set_ylabel(ycol.replace("_"," "))
    ax.set_title(title)
    ax.text(0.03, 0.97, f"Pearson r = {pr:.3f}\nSpearman ρ = {sr:.3f}\nn = {n}",
            transform=ax.transAxes, va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    outpng.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(outpng, dpi=220); plt.close()
    return dict(pearson_r=pr, pearson_p=pp, spearman_rho=sr, spearman_p=sp, n=n)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["sl_cnn","sl","rl"], default="sl_cnn")
    ap.add_argument("--model", default="model-cnn-sl.pt")
    ap.add_argument("--test", required=True)
    ap.add_argument("--human")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--L", type=int, default=200)
    ap.add_argument("--outdir", default="outputs/correlate_check")
    ap.add_argument("--do_permutation_check", action="store_true")
    ap.add_argument("--debug_stdout", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    infer_fn = import_infer(args.which, args.model)

    # --- AI (TEST) via infer() ---
    print("[info] running infer() on TEST/AI…")
    df_ai = run_infer(
        infer_fn, args.test, args.model, args.N, args.L,
        debug_stdout_path=(outdir/"infer_test_stdout.txt") if args.debug_stdout else None
    )
    print(f"[info] parsed TEST rows: {len(df_ai)}")
    df_ai.insert(0, "dataset", "TEST_AI")
    df_ai.to_csv(outdir / "per_seq_test.csv", index=False)

    res_ai = scatter_with_fit(df_ai, "score_norm", "design_model",
                              "TEST_AI: score_norm vs DESIGN_MODEL",
                              outdir / "scatter_testAI_scoreNorm_vs_designModel.png")
    all_res = [{"dataset":"TEST_AI", **res_ai}]

    # --- HUMAN (optional) ---
    if args.human:
        human_mode = "infer"
        try:
            with open(args.human) as f:
                obj = json.load(f)
            if is_gcloud_json(obj):
                human_mode = "gcloud"
        except Exception:
            pass

        if human_mode == "gcloud":
            print("[info] HUMAN detected as GCLOUD format (dict).")
            df_hu = load_gcloud_human(args.human, N=args.N)
            print(f"[info] parsed HUMAN rows: {len(df_hu)}")
            df_hu.insert(0, "dataset", "HUMAN_GCLOUD")
            df_hu.to_csv(outdir / "per_seq_human.csv", index=False)
            title = "HUMAN (gcloud): score_norm vs bioScore"
        else:
            print("[info] HUMAN treated as infer()-compatible JSON.")
            df_hu = run_infer(
                infer_fn, args.human, args.model, args.N, args.L,
                debug_stdout_path=(outdir/"infer_human_stdout.txt") if args.debug_stdout else None
            )
            print(f"[info] parsed HUMAN rows: {len(df_hu)}")
            df_hu.insert(0, "dataset", "HUMAN")
            df_hu.to_csv(outdir / "per_seq_human.csv", index=False)
            title = "HUMAN: score_norm vs DESIGN_MODEL"

        res_hu = scatter_with_fit(df_hu, "score_norm", "design_model",
                                  title,
                                  outdir / "scatter_human_scoreNorm_vs_designModel.png")
        all_res.append({"dataset":df_hu['dataset'].iloc[0], **res_hu})

        # Optional permutation control
        if args.do_permutation_check:
            for label, df_ in [("TEST_AI", df_ai), (df_hu['dataset'].iloc[0], df_hu)]:
                perm = df_.copy()
                perm["design_model"] = perm["design_model"].sample(frac=1.0, random_state=0).values
                r = scatter_with_fit(
                    perm, "score_norm", "design_model",
                    f"{label} (PERMUTED design): score_norm vs DESIGN",
                    outdir / f"scatter_{label.lower()}_perm_design.png"
                )
                r["dataset"] = f"{label}_PERMUTED"
                all_res.append(r)

    pd.DataFrame(all_res).to_csv(outdir / "correlation_summary.csv", index=False)
    print(f"[done] wrote outputs to {outdir}")

if __name__ == "__main__":
    main()
