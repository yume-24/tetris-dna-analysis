# analysis/pipelines/run_all.py
import argparse, subprocess, pathlib as P, sys

# at the top
import subprocess, sys
from pathlib import Path

def run(cmd):
    print("\n$"," ".join(map(str, cmd)))
    subprocess.check_call([str(x) for x in cmd])

def main():
    repo = Path(__file__).resolve().parents[2]   # repo root
    py = sys.executable                          # <<< use the current interpreter
    out = repo / "outputs/analysis"
    out.mkdir(parents=True, exist_ok=True)

    # ... keep the rest of your commands as-is ...


    ai_fa = out/"raw/seqs_main_ai.fasta"
    hu_fa = out/"raw/seqs_main_human.fasta"

    # 1) Export FASTAs (AI: tetris_test.json, Human: gcloud_data.json)
    run([py, repo/"analysis/generation/export_fastas.py",
         "--which","sl_cnn","--model","model-cnn-sl.pt",
         "--test","tetris_test.json","--human","gcloud_data.json",
         "--human_min_score","400","--score_key","score",
         "--N","5000","--outdir", out/"raw"])

    # 2) Correlate + save per-seq tables (used later)
    corr_out = out/"correlate_gcloud"
    run([py, repo/"analysis/correlate/correlate_game_vs_dna.py",
         "--which","sl_cnn","--model","model-cnn-sl.pt",
         "--test","tetris_test.json","--human","gcloud_data.json",
         "--N","1000","--outdir", corr_out])

    human_scores = corr_out/"per_seq_human.csv"
    ai_scores    = corr_out/"per_seq_test.csv"

    # 3) Describe (match_equal_n)
    run([py, repo/"analysis/describe/describe_ai_human.py",
         "--ai", ai_fa, "--human", hu_fa,
         "--outdir", out/"analysis/describe_run",
         "--match_equal_n"])

    # 4) Supervised PWM map (AI vs Human) — match AI N to Human N
    # AFTER (correct flag)
    run([py, repo / "analysis/plots/ai_vs_human_supervised_map.py",
         "--ai_fasta", ai_fa,
         "--human_fasta", hu_fa,
         "--ppar", repo / "motifs/MA0065.2.meme",
         "--nfkb", repo / "motifs/MA0105.4.meme",
         "--ai_match_human",  # <- use this
         "--seed", "7",  # optional: reproducible subsample
         "--outdir", out / "analysis/pwm_supervised_matched"])

    # 5) PWM logit embed (fit on AI; keep all Human; match AI N to Human N)
    run([py, repo / "analysis/plots/pwm_logit_embed.py",
         "--ai_fasta", out / "raw/seqs_main_ai.fasta",
         "--human_fasta", out / "raw/seqs_main_human.fasta",
         "--human_scores", out / "correlate_gcloud/per_seq_human.csv",
         "--metric", "score_norm", "--min", "-1",
         "--ppar", repo / "motifs/MA0065.2.meme",
         "--nfkb", repo / "motifs/MA0105.4.meme",
         "--k_top", "10", "--tsne",
         "--match_ai_to_human", "--seed", "7",
         "--outdir", out / "analysis/pwm_logits_matched"])

    # 6) Entropy correlations (shared axes) from the per-seq tables above
    # 6) Entropy correlations (CSV from corr_out; FASTAs from out/raw)
    run([py, repo / "analysis/plots/entropy_correlations.py",
         "--ai_csv", corr_out / "per_seq_test.csv",
         "--human_csv", corr_out / "per_seq_human.csv",
         "--ai_fasta", ai_fa,
         "--human_fasta", hu_fa,
         "--outdir", out / "entropy_corrs"])

    print("\n[OK] Pipeline complete.")

if __name__ == "__main__":
    main()
