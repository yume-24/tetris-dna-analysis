# Tetris → DNA Analysis

End-to-end pipeline that:
1) exports AI & human DNA FASTAs from game JSON,
2) computes per-sequence **game** and **design** metrics,
3) makes matched AI/Human diagnostics,
4) builds **PWM logit** embeddings and **supervised motif** maps,
5) correlates **entropy** with scores.

---

## Quick start

* must obtain .json and .pt files separately 

```bash
# Create & enter a virtualenv (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install deps (or install selectively if you prefer)
pip install -r requirements.txt  # if present
# otherwise at minimum:
pip install numpy pandas matplotlib scikit-learn scipy

# (If needed for model inference)
pip install torch  # choose the right build for your OS/CPU/GPU

# Run the whole pipeline
python analysis/main.py

```

## analysis/main.py
Exports FASTAs – analysis/generation/export_fastas.py
-Model: model-cnn-sl.pt
-AI: tetris_test.json
-Human: gcloud_data.json (filtered: score ≥ 400)
-➜ outputs/analysis/raw/seqs_main_ai.fasta
-outputs/analysis/raw/seqs_main_human.fasta

## Correlate game ↔ DNA – analysis/correlate/correlate_game_vs_dna.py
-Saves per-sequence tables + scatter plots
-➜ outputs/analysis/correlate_gcloud/per_seq_test.csv
-outputs/analysis/correlate_gcloud/per_seq_human.csv

##Describe (AI vs Human, matched N) – analysis/describe/describe_ai_human.py --match_equal_n
-Base composition, entropy, PWM summaries, k-mer PCA/t-SNE
-➜ outputs/analysis/describe_run/*

##Supervised PWM map (matched N) –analysis/plots/ai_vs_human_supervised_map.py --ai_match_human
-Sliding-window PWM profiles (PPARγ & NF-κB), simple supervisor (LDA/logreg), ROC/confusion
-➜ outputs/analysis/pwm_supervised_matched/*

## PWM logit embedding (matched N) – analysis/plots/pwm_logit_embed.py --match_ai_to_human --tsne
-Features = top-K window log-odds per motif (PPARγ | NF-κB)
-Plots: PCA, t-SNE, max-logit boxplots, designΔ = PPAR_top1 − NF-κB_top1
-➜ outputs/analysis/pwm_logits_matched/*

## Entropy correlations (shared axes) – analysis/plots/entropy_correlations.py
-Shannon entropy vs game and design (AI & Human)
-➜ outputs/analysis/entropy_corrs/*


---

you can run the pipeline using 
```bash
python analysis/main.py
```
or run individual steps using 
```bash
# 1) FASTA export
python analysis/generation/export_fastas.py \
  --which sl_cnn --model model-cnn-sl.pt \
  --test tetris_test.json \
  --human gcloud_data.json \
  --human_min_score 400 --score_key score \
  --N 5000 --outdir outputs/analysis/raw

# 2) Per-seq score tables
python analysis/correlate/correlate_game_vs_dna.py \
  --which sl_cnn --model model-cnn-sl.pt \
  --test tetris_test.json --human gcloud_data.json \
  --N 1000 --outdir outputs/analysis/correlate_gcloud

# 3) Describe (matched N)
python analysis/describe/describe_ai_human.py \
  --ai outputs/analysis/raw/seqs_main_ai.fasta \
  --human outputs/analysis/raw/seqs_main_human.fasta \
  --match_equal_n \
  --outdir outputs/analysis/describe_run

# 4) Supervised PWM map (matched N)
python analysis/plots/ai_vs_human_supervised_map.py \
  --ai_fasta outputs/analysis/raw/seqs_main_ai.fasta \
  --human_fasta outputs/analysis/raw/seqs_main_human.fasta \
  --ppar motifs/MA0065.2.meme --nfkb motifs/MA0105.4.meme \
  --ai_match_human --seed 7 \
  --outdir outputs/analysis/pwm_supervised_matched

# 5) PWM logit embedding (matched N)
python analysis/plots/pwm_logit_embed.py \
  --ai_fasta outputs/analysis/raw/seqs_main_ai.fasta \
  --human_fasta outputs/analysis/raw/seqs_main_human.fasta \
  --human_scores outputs/analysis/correlate_gcloud/per_seq_human.csv \
  --metric score_norm --min -1 \
  --ppar motifs/MA0065.2.meme --nfkb motifs/MA0105.4.meme \
  --k_top 10 --tsne --match_ai_to_human --seed 7 \
  --outdir outputs/analysis/pwm_logits_matched

# 6) Entropy correlations (shared axes)
python analysis/plots/entropy_correlations.py \
  --ai_csv outputs/analysis/correlate_gcloud/per_seq_test.csv \
  --human_csv outputs/analysis/correlate_gcloud/per_seq_human.csv \
  --ai_fasta outputs/analysis/raw/seqs_main_ai.fasta \
  --human_fasta outputs/analysis/raw/seqs_main_human.fasta \
  --outdir outputs/analysis/entropy_corrs
```










