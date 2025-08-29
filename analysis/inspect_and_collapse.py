# analysis/inspect_and_collapse.py
# Collapse rows to unique games by hashing the `game_matrix`, then filter by score.
# Writes two files:
#   - <prefix>_byhash_all.json     (all unique games)
#   - <prefix>_byhash_ge<min>.json (unique games with score >= min)
#
# Usage:
#   python analysis/inspect_and_collapse.py \
#     --in tetris_data.json --min 4 --out tetris_data
#
import argparse, json, hashlib
from typing import Any, List

def hash_game_matrix(mat: Any, decimals: int = 3) -> str:
    """
    Deterministically hash a nested list of floats/ints representing the game_matrix.
    We round to `decimals` to guard against tiny float jitters.
    """
    def _flatten(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                yield from _flatten(y)
        else:
            yield x

    # Heuristic: many matrices are list-of-rows. Build a string with row separators if possible.
    # First try to detect 2D; else just flatten 1D.
    if isinstance(mat, list) and mat and isinstance(mat[0], (list, tuple)):
        # 2D case
        rows = []
        for row in mat:
            if isinstance(row, (list, tuple)):
                rows.append(",".join(f"{round(float(v),decimals):.{decimals}f}" for v in row))
            else:
                rows.append(f"{round(float(row),decimals):.{decimals}f}")
        s = ";".join(rows)
    else:
        # 1D or unknown nesting: fully flatten
        s = ",".join(f"{round(float(v),decimals):.{decimals}f}" for v in _flatten(mat))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSON (list of {game_matrix, score, ...})")
    ap.add_argument("--min", type=int, default=4, help="Score threshold for the filtered output (default 4)")
    ap.add_argument("--out", dest="outprefix", required=True, help="Output file prefix, e.g. tetris_data")
    ap.add_argument("--score_key", default="score", help="Key name for score in JSON (default: score)")
    ap.add_argument("--decimals", type=int, default=3, help="Rounding for hashing (default 3)")
    args = ap.parse_args()

    with open(args.inp) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("Input JSON must be a list.")

    print(f"rows (raw): {len(data)}")
    if not data:
        raise SystemExit("No rows found.")

    # sanity peek
    row0 = data[0]
    print("keys example:", list(row0.keys()))

    # collapse by hash of game_matrix
    by_hash = {}
    dup_counts = {}
    score_key = args.score_key

    for d in data:
        if "game_matrix" not in d or score_key not in d:
            # skip malformed rows
            continue
        h = hash_game_matrix(d["game_matrix"], args.decimals)
        sc = d[score_key]
        # Keep the exemplar with the highest score under this hash
        if (h not in by_hash) or (sc > by_hash[h][score_key]):
            by_hash[h] = d
        dup_counts[h] = dup_counts.get(h, 0) + 1

    unique = list(by_hash.values())
    print(f"unique games (by matrix hash): {len(unique)}")

    ge = [g for g in unique if g.get(score_key, -10) >= args.min]
    print(f"unique games with {score_key} ≥ {args.min}: {len(ge)}")

    # write outputs
    all_path = f"{args.outprefix}_byhash_all.json"
    ge_path  = f"{args.outprefix}_byhash_ge{args.min}.json"
    with open(all_path, "w") as f:
        json.dump(unique, f)
    with open(ge_path, "w") as f:
        json.dump(ge, f)
    print("wrote:", all_path)
    print("wrote:", ge_path)

    # Optional: show top duplicate counts (helps understand why raw rows were huge)
    top_dups = sorted(dup_counts.values(), reverse=True)[:5]
    if top_dups and top_dups[0] > 1:
        print("top duplicate group sizes (by identical matrix):", top_dups)

if __name__ == "__main__":
    main()
