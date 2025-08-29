# analysis/check_json_formats.py
import json, argparse, random


"""
RUN THIS 
python analysis/check_json_formats.py --test tetris_test-5k.json --human "tetris_data copy.json"

"""

def load_first(path, k=5):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "games" in data: data = data["games"]
    return data, data[:k]

def summarize(label, data):
    n = len(data)
    zero = sum(1 for g in data if ("score" in g and g["score"]==0) or ("boardScore" in g and g["boardScore"]==0))
    # key histogram
    from collections import Counter
    keys = Counter()
    for g in data:
        for k in g.keys(): keys[k]+=1
    print(f"\n[{label}] n={n}, zero_score={zero}")
    print("Top keys:", keys.most_common(10))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test",  default="tetris_test-5k.json")
    ap.add_argument("--human", default="tetris_data copy.json")
    args = ap.parse_args()

    test_all,  test_head  = load_first(args.test)
    human_all, human_head = load_first(args.human)

    summarize("TEST",  test_all)
    summarize("HUMAN", human_all)

    print("\nSample TEST rows:")
    for g in test_head:  print(list(g.keys()))
    print("\nSample HUMAN rows:")
    for g in human_head: print(list(g.keys()))

if __name__ == "__main__":
    main()
