
"""
run python analysis/kmer/compare_ai_human_enrichment.py --k 4
"""
# analysis/compare_ai_human_enrichment.py
import argparse, os, re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _pick(colnames, patterns):
    """Find first column matching any regex pattern (case-insensitive)."""
    for p in patterns:
        rx = re.compile(p, re.I)
        for c in colnames:
            if rx.search(c):
                return c
    return None

def load_effect_sizes(k, ai_csv, hu_csv):
    have_ai = os.path.exists(ai_csv)
    have_hu = os.path.exists(hu_csv)

    if have_ai and have_hu:
        ai = pd.read_csv(ai_csv).rename(columns={"log2fc": "log2fc_ai"})
        hu = pd.read_csv(hu_csv).rename(columns={"log2fc": "log2fc_hu"})
        df = pd.merge(ai[["kmer", "log2fc_ai"]],
                      hu[["kmer", "log2fc_hu"]],
                      on="kmer", how="inner")
        return df

    combo = f"outputs/analysis/kmer_enrichment_k{k}.csv"
    if not os.path.exists(combo):
        raise SystemExit(f"Couldn’t find {ai_csv} and {hu_csv}, and fallback {combo} is missing.")

    df = pd.read_csv(combo)
    cols = list(df.columns)

    # 1) If there's a dataset column, pivot it
    if any(c.lower() == "dataset" for c in cols):
        ds_col = [c for c in cols if c.lower() == "dataset"][0]
        kmer_col = _pick(cols, [r"^kmer$"])
        lfc_col  = _pick(cols, [r"^log2fc$", r"^lfc$", r"log2.?fold"])
        if not (kmer_col and lfc_col):
            raise SystemExit(f"{combo} needs 'kmer' and 'log2fc/lfc' columns if using 'dataset'.")
        lab = df[ds_col].astype(str).str.lower()
        ai_mask = lab.str.contains("test_ai|main_ai|^ai$")
        hu_mask = lab.str.contains("human|main_human|^hu$|^human$")
        ai_df = df.loc[ai_mask, [kmer_col, lfc_col]].drop_duplicates(kmer_col).rename(
            columns={kmer_col:"kmer", lfc_col:"log2fc_ai"})
        hu_df = df.loc[hu_mask, [kmer_col, lfc_col]].drop_duplicates(kmer_col).rename(
            columns={kmer_col:"kmer", lfc_col:"log2fc_hu"})
        merged = pd.merge(ai_df, hu_df, on="kmer", how="inner")
        if merged.empty:
            raise SystemExit("No overlapping k-mers between AI and Human in combined file.")
        return merged

    # 2) No dataset column — try wide format with two LFC columns
    kmer_col = _pick(cols, [r"^kmer$"])
    ai_lfc   = _pick(cols, [r"^log2fc_.*ai$", r"^ai_?log2fc$", r"ai.*log2fc", r"log2fc.*ai"])
    hu_lfc   = _pick(cols, [r"^log2fc_.*human$", r"^human_?log2fc$", r"human.*log2fc", r"log2fc.*human"])
    if kmer_col and ai_lfc and hu_lfc:
        out = df[[kmer_col, ai_lfc, hu_lfc]].rename(
            columns={kmer_col:"kmer", ai_lfc:"log2fc_ai", hu_lfc:"log2fc_hu"})
        return out

    # 3) Try generic “two groups” names (AI/Human, TEST_AI/HUMAN) with or without “log2fc” in the name
    ai_guess = _pick(cols, [r"^ai$", r"^test_ai$", r"^main_ai$"])
    hu_guess = _pick(cols, [r"^human$", r"^main_human$", r"^hu$"])
    if kmer_col and ai_guess and hu_guess:
        out = df[[kmer_col, ai_guess, hu_guess]].rename(
            columns={kmer_col:"kmer", ai_guess:"log2fc_ai", hu_guess:"log2fc_hu"})
        return out

    raise SystemExit(
        f"{combo} has no 'dataset' column and I couldn’t infer AI/Human LFC columns.\n"
        f"Columns found: {cols}\n"
        "Options:\n"
        "  • Regenerate separate files (kmer_enrichment_AI_k4.csv and ...Human_...) and rerun, or\n"
        "  • Update the combined CSV to include a 'dataset' column with rows for AI and Human."
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--ai_csv", default="outputs/analysis/kmer_enrichment_AI_k4.csv")
    ap.add_argument("--hu_csv", default="outputs/analysis/kmer_enrichment_Human_k4.csv")
    ap.add_argument("--outdir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_effect_sizes(args.k, args.ai_csv, args.hu_csv)
    df["delta"] = df["log2fc_ai"] - df["log2fc_hu"]

    r = float(np.corrcoef(df["log2fc_ai"], df["log2fc_hu"])[0, 1])
    plt.figure(figsize=(6, 6))
    plt.scatter(df["log2fc_ai"], df["log2fc_hu"], s=12, alpha=0.7)
    lim = float(np.nanmax(np.abs(df[["log2fc_ai", "log2fc_hu"]].values)))
    lim = max(lim, 1e-3)
    plt.plot([-lim, lim], [-lim, lim], lw=1)
    plt.xlim(-lim, lim); plt.ylim(-lim, lim)
    plt.xlabel("AI vs Random  log2FC")
    plt.ylabel("Human vs Random  log2FC")
    plt.title(f"Per-{args.k}-mer effect sizes (r={r:.2f})")
    scat = os.path.join(args.outdir, f"ai_vs_human_k{args.k}_scatter.png")
    plt.tight_layout(); plt.savefig(scat, dpi=160); plt.close()

    out = df.reindex(df["delta"].abs().sort_values(ascending=False).index)
    top = os.path.join(args.outdir, f"ai_human_divergent_kmers_k{args.k}.csv")
    out.head(50).to_csv(top, index=False)

    print("[OK] wrote:")
    print(" ", scat, "(exists:", os.path.exists(scat), ")")
    print(" ", top,  "(exists:", os.path.exists(top),  ")")

if __name__ == "__main__":
    main()
