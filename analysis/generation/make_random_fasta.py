# analysis/make_random_fasta.py
import random, argparse, os
from pathlib import Path

"""
RUN THIS 

python analysis/generation/make_random_fasta.py --N 2000 --L 200 --out outputs/raw/seqs_baseline_random.fasta

can also pass in --freqs random dna should have weighted base freqs 
"""

#generates random dna. weighted(optional)
def rand_seq(L, freqs=None):
    if not freqs:
        alphabet = ["A","C","G","T"]
        return "".join(random.choice(alphabet) for _ in range(L))
    # weighted
    letters, weights = zip(*freqs.items())
    return "".join(random.choices(letters, weights=weights, k=L))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--L", type=int, default=200)
    ap.add_argument("--out", default="outputs/raw/seqs_baseline_random.fasta")
    # optional: skew slightly to mimic GC ~50%
    ap.add_argument("--freqs", default="")
    args = ap.parse_args()

    freqs = None
    if args.freqs:
        # e.g. --freqs A:0.25,C:0.25,G:0.25,T:0.25
        freqs = {k: float(v) for k,v in (kv.split(":") for kv in args.freqs.split(","))}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for i in range(args.N):
            s = rand_seq(args.L, freqs)
            f.write(f">Baseline_{i+1}\n{s}\n")
    print("[OK] wrote", args.out)




"""
def rnd(N,L,p=(0.25,0.25,0.25,0.25),seed=0):
    rng=np.random.default_rng(seed); b=np.array(list("ATGC"))
    idx=rng.choice(4,(N,L),p=p); return ["".join(b[r]) for r in idx]
def write_fa(path,seqs,tag):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,"w") as f:
        for i,s in enumerate(seqs): f.write(f">{tag}_{i+1}\n{s}\n")
if __name__=="__main__":
    write_fa("outputs/raw/seqs_random.fasta", rnd(2000,200), "Random")

"""