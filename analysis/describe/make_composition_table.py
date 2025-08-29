# analysis/make_composition_table.py
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np



"""
run this 
python analysis/make_composition_table.py \
  --scores outputs/describe_run/per_seq_scores.csv \
  --outdir outputs/tables \
  --round 5

"""
def read_fasta_index_map(fasta_path):
    """Return dict: index(int) -> sequence(upper) from >TAG_IDX headers."""
    idx2seq, cur_id, buf = {}, None, []
    if not Path(fasta_path).exists():
        return idx2seq
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                if cur_id is not None and buf:
                    idx2seq[cur_id] = "".join(buf).strip().upper()
                m = re.search(r"(\d+)\s*$", line.strip())  # digits at end
                cur_id = int(m.group(1)) if m else None
                buf = []
            else:
                buf.append(line.strip())
        if cur_id is not None and buf:
            idx2seq[cur_id] = "".join(buf).strip().upper()
    return idx2seq

def guess_fasta_for_dataset(dataset):
    """Map dataset name to a FASTA in outputs/raw."""
    d = dataset.lower()
    candidates = list(Path("outputs/raw").glob("*.fasta"))
    name = None
    if "test" in d or "ai" in d: name = "main_ai"
    elif "human" in d:           name = "main_human"
    elif "baseline" in d:        name = "baseline"
    elif "random" in d:          name = "random"
    if name:
        for p in candidates:
            if name in p.name.lower():
                return p
    # last resort: return None
    return None

def seq_fracs(seq):
    s = (seq or "").upper()
    L = sum(ch in "ACGT" for ch in s) or (len(s) if s else 1)
    return pd.Series({
        "A_frac": s.count("A")/L if L else np.nan,
        "C_frac": s.count("C")/L if L else np.nan,
        "G_frac": s.count("G")/L if L else np.nan,
        "T_frac": s.count("T")/L if L else np.nan,
    })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="outputs/describe_run/per_seq_scores.csv",
                    help="per-seq table (must include dataset, seq_id)")
    ap.add_argument("--outdir", default="outputs/tables",
                    help="Where to write table_composition.csv")
    ap.add_argument("--round", type=int, default=5)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.scores)
    if not {"dataset","seq_id"}.issubset(df.columns):
        raise SystemExit("Expected columns 'dataset' and 'seq_id' in scores CSV.")

    # Extract numeric index from seq_id (e.g., MainAI_123 -> 123)
    idx = df["seq_id"].astype(str).str.extract(r"(\d+)$")[0].astype(float)
    if idx.isna().any():
        raise SystemExit("Could not extract numeric indices from seq_id (…_NNN).")
    df["seq_idx"] = idx.astype(int)

    # Build per-dataset index->sequence maps from FASTAs
    datasets = df["dataset"].unique().tolist()
    dset_to_idx2seq = {}
    for dset in datasets:
        fa = guess_fasta_for_dataset(dset)
        if fa is None:
            print(f"[WARN] No FASTA found for dataset='{dset}' in outputs/raw; skipping")
            continue
        dset_to_idx2seq[dset] = read_fasta_index_map(fa)
        if not dset_to_idx2seq[dset]:
            print(f"[WARN] FASTA '{fa}' read but no records parsed; skipping")

    # Derive composition for each row from (dataset, seq_idx)
    def comp_row(row):
        m = dset_to_idx2seq.get(row["dataset"], {})
        seq = m.get(row["seq_idx"], "")
        return seq_fracs(seq)

    comp = df.apply(comp_row, axis=1)
    df = pd.concat([df, comp], axis=1)
    if "GC_frac" not in df.columns:
        df["GC_frac"] = df["G_frac"] + df["C_frac"]

    # Aggregate mean/std/count per dataset
    cols = ["A_frac","C_frac","G_frac","T_frac","GC_frac"]
    ok_cols = [c for c in cols if c in df.columns]
    agg = df.groupby("dataset")[ok_cols].agg(["mean","std","count"])
    agg.columns = [f"{m}_{s}" for (m,s) in agg.columns]
    agg = agg.reset_index()

    # Round
    for c in agg.columns:
        if c != "dataset":
            agg[c] = agg[c].astype(float).round(args.round)

    out_path = outdir / "table_composition.csv"
    agg.to_csv(out_path, index=False)
    print("[OK] Wrote", out_path)

if __name__ == "__main__":
    main()
