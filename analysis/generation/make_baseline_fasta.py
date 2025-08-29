# analysis/make_baseline_fasta.py
import argparse, os, random, itertools
from collections import defaultdict, Counter
from pathlib import Path




"""
RUN 
Train on the HUMAN set, 3rd-order Markov, generate 1000×200bp
python analysis/generation/make_baseline_fasta.py --train outputs/raw/seqs_main_human.fasta --k 3 --N 1000 --L 200 --out outputs/raw/seqs_baseline.fasta

"""
def read_fasta(path):
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur: seqs.append("".join(cur).strip().upper()); cur=[]
            else:
                cur.append(line.strip())
    if cur: seqs.append("".join(cur).strip().upper())
    # keep only ACGT
    clean = []
    for s in seqs:
        s = "".join(ch for ch in s if ch in "ACGTacgt").upper()
        if s: clean.append(s)
    return clean

def train_markov(seqs, k, alpha=1.0):
    """k>=1. Returns: dict[prefix]->dict[next_base]->prob"""
    bases = "ACGT"
    # count (prefix -> next)
    trans = defaultdict(Counter)
    for s in seqs:
        if len(s) < k: continue
        for i in range(len(s)-k):
            pref = s[i:i+k]
            nxt  = s[i+k]
            trans[pref][nxt]+=1
    # Laplace smoothing & normalize
    probs = {}
    for pref, c in trans.items():
        total = sum(c.values()) + alpha*len(bases)
        probs[pref] = {b:(c.get(b,0)+alpha)/total for b in bases}
    # initialize prefix distribution from observed k-mers
    kmer_counts = Counter()
    for s in seqs:
        for i in range(len(s)-k+1):
            kmer_counts[s[i:i+k]] += 1
    total_k = sum(kmer_counts.values()) + alpha*(4**k)
    init = {km:(kmer_counts.get(km,0)+alpha)/total_k
            for km in ("".join(p) for p in itertools.product("ACGT", repeat=k))}
    return probs, init

def sample_seq(k, probs, init, L, rng):
    prefix_choices, prefix_weights = zip(*init.items())
    # start with a k-mer
    pref = rng.choices(prefix_choices, weights=prefix_weights, k=1)[0]
    out = [c for c in pref]
    bases = "ACGT"
    for _ in range(L - k):
        p = probs.get(pref)
        if not p:  # unseen prefix → backoff: pick base by global freq
            p = {b:1.0/4 for b in bases}
        bs, ws = zip(*p.items())
        nxt = rng.choices(bs, weights=ws, k=1)[0]
        out.append(nxt)
        pref = "".join(out[-k:])
    return "".join(out)

def write_fasta(path, seqs, tag):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, s in enumerate(seqs, 1):
            f.write(f">{tag}_{i}\n{s}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+",
                    default=["outputs/raw/seqs_main_human.fasta"],
                    help="FASTA(s) to fit the baseline on")
    ap.add_argument("--k", type=int, default=3, help="Markov order")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--L", type=int, default=200)
    ap.add_argument("--out", default="outputs/raw/seqs_baseline.fasta")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    # collect training sequences
    pool = []
    for p in args.train:
        pool += read_fasta(p)
    if not pool:
        raise SystemExit("No training sequences found.")

    probs, init = train_markov(pool, args.k)
    rng = random.Random(args.seed)
    seqs = [sample_seq(args.k, probs, init, args.L, rng) for _ in range(args.N)]
    write_fasta(args.out, seqs, f"BaselineK{args.k}")
    print("[OK] wrote", args.out)

if __name__ == "__main__":
    main()
