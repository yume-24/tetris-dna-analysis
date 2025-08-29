import argparse, pathlib as P
import numpy as np, pandas as pd, matplotlib.pyplot as plt

ALPH = "ACGT"; IDX = {c:i for i,c in enumerate(ALPH)}
def entropy_bits(seq: str) -> float:
    if not seq: return 0.0
    counts = np.zeros(4, dtype=float)
    for ch in seq:
        i = IDX.get(ch, -1)
        if i >= 0: counts[i] += 1
    p = counts / max(1.0, counts.sum())
    p = p[p > 0]
    return float(-(p*np.log2(p)).sum())

def load_entropy_from_csv(csv_path: str, label: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    if "entropy" in df.columns:
        s = df["entropy"].astype(float)
    elif "seq" in df.columns:
        s = df["seq"].map(entropy_bits)
    elif "sequence" in df.columns:
        s = df["sequence"].map(entropy_bits)
    else:
        raise SystemExit(f"[ERR] {csv_path} has neither entropy nor sequence.")
    s.name = label
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai_csv", required=True, help="CSV with 'seq' or 'entropy'")
    ap.add_argument("--human_csv", required=True, help="CSV with 'seq' or 'entropy'")
    ap.add_argument("--outdir", default="outputs/analysis/entropy_overlay")
    ap.add_argument("--bins", type=int, default=40)
    args = ap.parse_args()

    outdir = P.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    e_ai = load_entropy_from_csv(args.ai_csv, "AI")
    e_hu = load_entropy_from_csv(args.human_csv, "Human")

    # overlay histogram
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(e_ai, bins=args.bins, histtype="step", density=True, label=f"AI (n={len(e_ai)})")
    ax.hist(e_hu, bins=args.bins, histtype="step", density=True, label=f"Human (n={len(e_hu)})")
    ax.set_xlabel("Per-sequence Shannon entropy (bits)")
    ax.set_ylabel("Density")
    ax.set_title("Entropy distribution: AI vs Human")
    ax.legend()
    fig.tight_layout(); fig.savefig(outdir/"entropy_overlay_hist.png", dpi=220); plt.close(fig)

    # violin (same axis)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.violinplot([e_ai, e_hu], showmeans=True, showextrema=True)
    ax.set_xticks([1,2], [f"AI (n={len(e_ai)})", f"Human (n={len(e_hu)})"])
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Entropy: AI vs Human")
    fig.tight_layout(); fig.savefig(outdir/"entropy_overlay_violin.png", dpi=220); plt.close(fig)

    # small stats table
    res = pd.DataFrame({
        "dataset": ["AI","Human"],
        "n": [len(e_ai), len(e_hu)],
        "mean": [e_ai.mean(), e_hu.mean()],
        "std":  [e_ai.std(ddof=1), e_hu.std(ddof=1)]
    })
    res.to_csv(outdir/"entropy_summary.csv", index=False)
    print(f"[ok] wrote {outdir}")

if __name__ == "__main__":
    main()
