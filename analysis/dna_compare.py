# analysis/dna_compare.py
import argparse, os, math, csv
from collections import Counter, defaultdict
from pathlib import Path



"""
csvs: 
-base frequencies
-per-sequence entropies
-top 3-mers per group

pngs: 
-base dist and mean entropy 

RUN THIS 
python analysis/dna_compare.py \
  --input Baseline=outputs/raw/seqs_baseline_random.fasta \
  --input Main_AI=outputs/raw/seqs_main_ai.fasta \
  --input Main_Human=outputs/raw/seqs_main_human.fasta \
  --outdir outputs/compare_run --k 3

"""


def read_fasta(path):
    name, seqs = None, []
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            if line.startswith(">"):
                name=line
                continue
            seqs.append(line.upper())
    return seqs

def entropy(seq):
    c = Counter(seq)
    n = len(seq)
    H = 0.0
    for b in "ACGT":
        p = c.get(b,0)/n
        if p>0: H -= p*math.log2(p)
    return H

def base_counts(seqs):
    cnt = Counter()
    N = 0
    for s in seqs:
        cnt.update(ch for ch in s if ch in "ACGT")
        N += sum(1 for ch in s if ch in "ACGT")
    return {b: cnt.get(b,0) for b in "ACGT"}, N

def kmer_counts(seqs, k=3):
    cnt = Counter()
    for s in seqs:
        for i in range(0, len(s)-k+1):
            kmer = s[i:i+k]
            if all(ch in "ACGT" for ch in kmer):
                cnt[kmer]+=1
    return cnt

def safe_plot_bar(labels, values, title, outpath):
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.bar(labels, values)
        plt.title(title)
        plt.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[plot skipped] {title}: {e}")
        return False

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True,
                    help="Label=path.fasta (can be passed multiple times)")
    ap.add_argument("--outdir", default="outputs/compare_run")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    groups = {}
    for spec in args.input:
        label, path = spec.split("=",1)
        groups[label] = read_fasta(path)

    # Write per-group metrics
    summary_rows = []
    for label, seqs in groups.items():
        # Base composition
        bc, total = base_counts(seqs)
        freqs = {b: (bc[b]/total if total else 0.0) for b in "ACGT"}

        # Entropy distribution
        ent = [entropy(s) for s in seqs]
        H_avg = sum(ent)/len(ent) if ent else 0.0

        # K-mer (top 20)
        km = kmer_counts(seqs, k=args.k)
        top_k = km.most_common(20)

        # Save CSVs
        with open(os.path.join(args.outdir, f"{label}_base_freqs.csv"), "w", newline="") as f:
            w=csv.writer(f); w.writerow(["Base","Frequency"])
            for b in "ACGT": w.writerow([b, freqs[b]])
        with open(os.path.join(args.outdir, f"{label}_entropy.csv"), "w", newline="") as f:
            w=csv.writer(f); w.writerow(["seq_index","entropy_bits"])
            for i,h in enumerate(ent): w.writerow([i+1, h])
        with open(os.path.join(args.outdir, f"{label}_kmer_top{min(20,len(km))}.csv"), "w", newline="") as f:
            w=csv.writer(f); w.writerow([f"{args.k}-mer","count"])
            for kmer,count in top_k: w.writerow([kmer,count])

        # Quick plots (if matplotlib present)
        safe_plot_bar(["A","C","G","T"], [freqs[b] for b in "ACGT"],
                      f"{label} base distribution", os.path.join(args.outdir, f"{label}_base_dist.png"))
        # tiny entropy summary plot (mean only)
        safe_plot_bar(["mean_entropy"], [H_avg],
                      f"{label} entropy (mean)", os.path.join(args.outdir, f"{label}_entropy_mean.png"))

        summary_rows.append([label, len(seqs), freqs["A"], freqs["C"], freqs["G"], freqs["T"], H_avg])

    # Combined summary
    with open(os.path.join(args.outdir, "summary.csv"), "w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["label","num_seqs","A","C","G","T","mean_entropy_bits"])
        for row in summary_rows: w.writerow(row)

    print("[OK] wrote metrics to", args.outdir)
