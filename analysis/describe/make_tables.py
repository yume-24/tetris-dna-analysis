import argparse, os, re
import pandas as pd
import numpy as np


#do composition table separately
""""
run this 
python analysis/make_tables.py \
  --scores outputs/describe_run/per_seq_scores.csv \
  --outdir outputs/tables \
  --round 5

"""
# columns we usually want if present (others are ignored)
PREFERRED = [
    "entropy",
    "A_frac","C_frac","G_frac","T_frac","GC_frac",
    "pparg_score","nfkb_score","design_score",
    "length"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="outputs/analysis/per_seq_scores.csv",
                    help="CSV from describe_fastas.py")
    ap.add_argument("--outdir", default="outputs/analysis",
                    help="Where to write the tables")
    ap.add_argument("--round", type=int, default=4,
                    help="Decimal places for rounding")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.scores)

    # Normalize column names just in case
    df.columns = [c.strip() for c in df.columns]
    # ---- normalize common alternate column names (case-insensitive) ----
    alias = {
        # bio scores
        "design": "design_score",
        "design_norm": "design_score",
        "nfkb": "nfkb_score",
        "nfkb_score": "nfkb_score",
        "nfkbp": "nfkb_score",
        "nfκb": "nfkb_score",
        "pparg": "pparg_score",
        "pparγ": "pparg_score",
        # composition
        "a": "A_frac", "c": "C_frac", "g": "G_frac", "t": "T_frac",
        "gc": "GC_frac", "gc_percent": "GC_frac", "gc_pct": "GC_frac"
    }
    lc2orig = {c.lower(): c for c in df.columns}
    renames = {}
    for k, v in alias.items():
        if k in lc2orig and v not in df.columns:
            renames[lc2orig[k]] = v
    if renames:
        df = df.rename(columns=renames)
    # --------------------------------------------------------------------
    # ---- ensure A/C/G/T fractions exist; derive them if missing -----------------
    import numpy as np

    need_comp = not all(c in df.columns for c in ["A_frac", "C_frac", "G_frac", "T_frac"])
    if need_comp:
        # 1) From a sequence column if present
        seq_col = next((c for c in ["sequence", "seq", "dna", "dna_seq"] if c in df.columns), None)
        if seq_col:
            def fracs_from_seq(s):
                s = str(s).upper()
                # count only A/C/G/T; ignore N/others in denominator
                L = sum(ch in "ACGT" for ch in s) or len(s) or 1
                return pd.Series({
                    "A_frac": s.count("A") / L,
                    "C_frac": s.count("C") / L,
                    "G_frac": s.count("G") / L,
                    "T_frac": s.count("T") / L,
                })

            df[["A_frac", "C_frac", "G_frac", "T_frac"]] = df[seq_col].apply(fracs_from_seq)
        else:
            # 2) From count-like columns
            count_sets = [
                ("A_count", "C_count", "G_count", "T_count"),
                ("A", "C", "G", "T"),
                ("count_A", "count_C", "count_G", "count_T"),
                ("nA", "nC", "nG", "nT"),
            ]
            for (a, c, g, t) in count_sets:
                if all(col in df.columns for col in (a, c, g, t)):
                    S = df[[a, c, g, t]].sum(axis=1).replace(0, np.nan)
                    df["A_frac"] = df[a] / S
                    df["C_frac"] = df[c] / S
                    df["G_frac"] = df[g] / S
                    df["T_frac"] = df[t] / S
                    break

    # Always create GC_frac if we now have G_frac/C_frac
    if "G_frac" in df.columns and "C_frac" in df.columns and "GC_frac" not in df.columns:
        df["GC_frac"] = df["G_frac"] + df["C_frac"]
    # ---------------------------------------------------------------------------

    # Ensure numeric dtypes for summary columns
    for c in ["A_frac", "C_frac", "G_frac", "T_frac", "GC_frac",
              "entropy", "pparg_score", "nfkb_score", "design_score", "length"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


    # Keep only numeric cols we care about (and drop any k-mer columns)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if not re.match(r'^(kmer|k\d+_)', c, flags=re.I)]
    # If GC_frac not present, compute it
    if "GC_frac" not in df.columns and all(c in df.columns for c in ["G_frac","C_frac"]):
        df["GC_frac"] = df["G_frac"] + df["C_frac"]
        if "GC_frac" not in num_cols:
            num_cols.append("GC_frac")

    # Prioritize preferred order; include any other numeric columns at the end
    ordered = [c for c in PREFERRED if c in num_cols] + [c for c in num_cols if c not in PREFERRED]

    if "dataset" not in df.columns:
        raise SystemExit("Expected a 'dataset' column in per_seq_scores.csv")

    # Main summary (mean ± std) per dataset
    agg = df.groupby("dataset")[ordered].agg(["mean","std","count"])
    # Flatten MultiIndex columns like ('entropy','mean') -> 'entropy_mean'
    agg.columns = [f"{m}_{s}" for (m, s) in agg.columns]
    agg = agg.reset_index()

    # Round nicely
    for c in agg.columns:
        if c != "dataset":
            agg[c] = agg[c].astype(float).round(args.round)

    # Save main summary table
    summary_path = os.path.join(args.outdir, "table_summary_stats.csv")
    agg.to_csv(summary_path, index=False)

    # A smaller “bio scores only” table (just in case Ben wants a quick view)
    bio_cols = [c for c in ["pparg_score","nfkb_score","design_score"] if c in ordered]
    if bio_cols:
        bio = df.groupby("dataset")[bio_cols].agg(["mean","std","count"])
        bio.columns = [f"{m}_{s}" for (m, s) in bio.columns]
        bio = bio.reset_index()
        for c in bio.columns:
            if c != "dataset":
                bio[c] = bio[c].astype(float).round(args.round)
        bio_path = os.path.join(args.outdir, "table_bio_scores.csv")
        bio.to_csv(bio_path, index=False)

    # Composition-only table (A/C/G/T/GC)
    # Composition-only table (A/C/G/T/GC)
    comp_cols = [c for c in ["A_frac", "C_frac", "G_frac", "T_frac", "GC_frac"] if c in df.columns]
    if comp_cols:
        for c in comp_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        comp = df.groupby("dataset")[comp_cols].agg(["mean", "std", "count"])
        comp.columns = [f"{m}_{s}" for (m, s) in comp.columns]
        comp = comp.reset_index()
        for c in comp.columns:
            if c != "dataset":
                comp[c] = comp[c].astype(float).round(args.round)
        comp_path = os.path.join(args.outdir, "table_composition.csv")
        comp.to_csv(comp_path, index=False)

    print("[OK] Wrote:")
    print(" ", summary_path)
    if bio_cols:
        print(" ", bio_path)
    if comp_cols:
        print(" ", comp_path)

if __name__ == "__main__":
    main()
