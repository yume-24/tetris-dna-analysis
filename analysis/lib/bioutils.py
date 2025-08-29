# analysis/lib/bioutils.py  (Python 3.9)
from __future__ import annotations
import hashlib, os, pathlib as P
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd

# ----------------------- IO helpers -----------------------
def ensure_dir(p: P.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_fasta(path: str) -> List[str]:
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s: continue
            if s.startswith(">"):
                if cur: seqs.append("".join(cur).upper())
                cur = []
            else:
                cur.append(s)
    if cur: seqs.append("".join(cur).upper())
    return seqs

def write_fasta(path: str, seqs: List[str], tag: str = "seq") -> None:
    path = str(path)
    ensure_dir(P.Path(path).parent)
    with open(path, "w") as f:
        for i, s in enumerate(seqs, 1):
            f.write(f">{tag}_{i}\n{s}\n")

# ----------------------- sequence stats -----------------------
_ALPH = "ACGT"
_IDX = {c:i for i,c in enumerate(_ALPH)}
_RC = str.maketrans("ACGT","TGCA")
def revcomp(s: str) -> str: return s.translate(_RC)[::-1]

def entropy_bits(seq: str) -> float:
    # Shannon entropy of mono-nucleotide distribution
    if not seq:
        return 0.0
    counts = np.zeros(4, dtype=float)
    for ch in seq:
        i = _IDX.get(ch, -1)
        if i >= 0: counts[i] += 1.0
    p = counts / max(1.0, counts.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def composition(seq: str) -> Dict[str, float]:
    n = max(1, len(seq))
    c = dict(A=seq.count("A")/n, C=seq.count("C")/n,
             G=seq.count("G")/n, T=seq.count("T")/n)
    c["GC_frac"] = c["G"] + c["C"]
    return c

# ----------------------- MEME PWM & scanning -----------------------
def parse_meme_pwm(path: str) -> np.ndarray:
    rows, in_mat = [], False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.lower().startswith("letter-probability matrix"):
                in_mat = True; continue
            if in_mat:
                if (not s) or s[0].isalpha():
                    break
                vals = [float(x) for x in s.split()]
                if len(vals) >= 4: rows.append(vals[:4])
    if not rows:
        raise ValueError(f"No PWM read from {path}")
    pwm = np.array(rows, dtype=float) # shape (w,4) order A,C,G,T
    pwm = np.clip(pwm, 1e-9, 1.0)
    pwm /= pwm.sum(axis=1, keepdims=True)
    return pwm

def pwm_logodds_windows(seq: str, pwm: np.ndarray,
                        bg: Optional[np.ndarray] = None) -> np.ndarray:
    """Return max-of-strands log-odds for every window."""
    if bg is None:
        bg = np.array([0.25,0.25,0.25,0.25], dtype=float)
    L, w = len(seq), pwm.shape[0]
    if L < w:
        return np.array([], dtype=float)
    logp = np.log(pwm) - np.log(bg[None,:])  # (w,4)

    def scan(s: str) -> np.ndarray:
        scores = np.full(L-w+1, -1e30, dtype=float)
        for i in range(L-w+1):
            tot = 0.0; ok = True
            window = s[i:i+w]
            for j, ch in enumerate(window):
                a = _IDX.get(ch, -1)
                if a < 0: ok=False; break
                tot += logp[j, a]
            if ok: scores[i] = tot
        return scores

    fwd = scan(seq)
    rev = scan(revcomp(seq))
    return np.maximum(fwd, rev)

def topk_desc(arr: np.ndarray, k: int) -> np.ndarray:
    if arr.size == 0:
        return np.full(k, -1e30, dtype=float)
    srt = np.sort(arr)[::-1]
    if srt.size >= k:
        return srt[:k]
    out = np.full(k, srt[-1], dtype=float)
    out[:srt.size] = srt
    return out

def design_logits(seq: str, pwm_ppar: np.ndarray, pwm_nfkb: np.ndarray,
                  k_top: int = 1) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (ppar_topk, nfkb_topk, designΔ = top1_ppar - top1_nfkb)."""
    bg = np.array([0.25,0.25,0.25,0.25], dtype=float)
    ppar = topk_desc(pwm_logodds_windows(seq, pwm_ppar, bg), k_top)
    nfkb = topk_desc(pwm_logodds_windows(seq, pwm_nfkb, bg), k_top)
    delta = float(ppar[0] - nfkb[0])
    return ppar, nfkb, delta

# ----------------------- caching helpers -----------------------
def sha1(items: List[str]) -> str:
    h = hashlib.sha1()
    for it in items:
        h.update(str(it).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:12]
