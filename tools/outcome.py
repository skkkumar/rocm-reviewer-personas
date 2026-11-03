#!/usr/bin/env python3
"""Look up how PRs touching the same files went: rounds, days to merge, who reviewed, top asks.

Usage:
  python outcome.py --diff pr.diff [--data ../data/outcomes.json]
  python outcome.py --files vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
"""
import argparse, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))

def files_from_diff(text):
    return re.findall(r"^diff --git a/\S+ b/(\S+)$", text, re.M)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff"); ap.add_argument("--files", nargs="*"); ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "outcomes.json"))
    a = ap.parse_args()
    data = json.load(open(a.data, encoding="utf-8", errors="replace"))
    files = a.files or files_from_diff(sys.stdin.read() if a.diff == "-" else open(a.diff, encoding="utf-8", errors="replace").read())
    seen = set()
    for f in files:
        base = os.path.basename(f)
        rec = data["by_file"].get(base)
        if not rec or base in seen: continue
        seen.add(base)
        print(f"\n## {base}  ({rec['n_prs']} PRs, {rec['merged']} merged, {rec['closed']} closed)")
        print(f"- Median days open->merge: {rec['median_days_to_merge']}   median human comments: {rec['median_comments']}   median review rounds: {rec['median_rounds']}")
        print("- Reviewers who showed up: " + ", ".join(f"{w} ({n})" for w, n in rec["reviewers"][:6]))
        print("- What they asked for most: " + ", ".join(f"{t} ({n})" for t, n in rec["themes"][:6]))
        if rec.get("examples"): print("- Example PRs: " + ", ".join(f"#{n} ({d}d, {s})" for n, d, s in rec["examples"][:5]))
    if not seen: print("No history for these files in the data set. Treat as a new area: expect tjtanaa and AndreasKaratzas.")

if __name__ == "__main__":
    main()
