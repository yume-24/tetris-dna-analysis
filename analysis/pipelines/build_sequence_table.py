# analysis/pipelines/build_sequence_table.py
from __future__ import annotations
import argparse, pathlib as P
import numpy as np, pandas as pd
from typing import Optional
from analysis.lib.bioutils import (
    read_fasta, entropy_bits, composition,
    parse_meme_pwm, design_logits, ensure_dir, sha1
)

def build_table(ai_fasta: str, human_fasta: str,
                pwm_ppar_path: str, pwm_nfkb_path: str,
                human_scores_csv: Optional[str] = None,
                score_col: str = "score_norm",
                human_min: Optional[float] = None,
                match_ai_to_human: bool = True,
                k_top: int = 1) -> pd.DataFrame:

    ai = read_fasta(ai_fasta)
    hu = read_fasta(human_fasta)

    # Optional human filtering using external scores file
    if human_scores_csv and human_min is not None:
        dfh = pd.read_csv(human_scores_csv)
        # keep 1..N mapping if no seq_id is present
        if "seq_id" not in dfh.columns:
            dfh["seq_id"] = np.arange(1, len(dfh)+1)
        keep_ids = dfh.loc[dfh[score_col] >= human_min, "seq_id"].astype(int).tolist()
        keep_ids = [i for i in keep_ids if 1 <= i <= len(hu)]
        hu = [hu[i-1] for i in keep_ids]

    if match_ai_to_human:
        ai = ai[:len(hu)]

    pwm_ppar = parse_meme_pwm(pwm_ppar_path)
    pwm_nfkb = parse_meme_pwm(pwm_nfkb_path)

    def rows_for(seqs, dataset):
        rows = []
        for i, s in enumerate(seqs, 1):
            pp, nf, d = design_logits(s, pwm_ppar, pwm_nfkb, k_top=k_top)
            comp = composition(s)
            rows.append({
                "dataset": dataset,
                "seq_id": i,
                "sequence": s,
                "length": len(s),
                "entropy": entropy_bits(s),
                **comp,
                "ppar_top1": pp[0],
                "nfkb_top1": nf[0],
                "design_delta": d
            })
        return rows

    data = rows_for(ai, "AI") + rows_for(hu, "Human")
    return pd.DataFrame(data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai_fasta", required=True)
    ap.add_argument("--human_fasta", required=True)
    ap.add_argument("--human_scores", default=None)
    ap.add_argument("--score_col", default="score_norm")
    ap.add_argument("--human_min", type=float, default=None)
    ap.add_argument("--match_ai_to_human", action="store_true")
    ap.add_argument("--ppar", default="motifs/MA0065.2.meme")
    ap.add_argument("--nfkb", default="motifs/MA0105.4.meme")
    ap.add_argument("--k_top", type=int, default=1)
    ap.add_argument("--out", default="outputs/cache/per_seq_table")
    args = ap.parse_args()

    df = build_table(
        args.ai_fasta, args.human_fasta,
        args.ppar, args.nfkb,
        human_scores_csv=args.human_scores,
        score_col=args.score_col,
        human_min=args.human_min,
        match_ai_to_human=args.match_ai_to_human,
        k_top=args.k_top
    )

    base = P.Path(args.out)
    ensure_dir(base.parent)
    df.to_parquet(str(base)+".parquet", index=False)
    df.to_csv(str(base)+".csv", index=False)
    print(f"[ok] wrote {base}.parquet and .csv  (n={len(df)})")

if __name__ == "__main__":
    main()
