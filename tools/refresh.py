#!/usr/bin/env python3
"""Refresh the persona data from GitHub. No token needed (scrapes the web pages).

Steps:
  1. list ROCm PRs      -> data/lists/*.json
  2. fetch review pages -> data/prs/<num>.json   (slow: ~1.5 s per PR; resumable)
  3. re-derive          -> data/quotes.json, data/derived.json, data/outcomes.json, data/drift.md

Usage:
  python refresh.py --since 2026-08-01            # full refresh for PRs created since a date
  python refresh.py --since 2026-09-01 --max-pages 10 --skip-fetch   # just re-derive from cached pages

Then open data/drift.md: it lists reviewer themes that moved by >5 points and new phrases that appeared in 3+ PRs
since the last refresh, so you can update reviewers/<handle>.md by hand (the hand-written sections) while the
"Data-derived profile" and "Real review comments" sections are regenerated automatically by derive_personas.py.
"""
import argparse, collections, datetime, glob, json, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); DATA = os.path.join(ROOT, "data")

def month_windows(since):
    """[(YYYY-MM-DD, YYYY-MM-DD), ...] calendar months from `since` through today."""
    d = datetime.date.fromisoformat(since); today = datetime.date.today(); out = []
    while d <= today:
        nxt = (d.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)
        out.append((d.isoformat(), min(nxt - datetime.timedelta(days=1), today).isoformat())); d = nxt
    return out

def sh(cmd): print("+", " ".join(cmd), file=sys.stderr); subprocess.run(cmd, check=True, cwd=HERE)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--since", required=True); ap.add_argument("--max-pages", type=int, default=40); ap.add_argument("--skip-fetch", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(DATA, "lists"), exist_ok=True); os.makedirs(os.path.join(DATA, "prs"), exist_ok=True)
    os.environ["ROCM_REVIEW_PRS"] = os.path.join(DATA, "prs")
    if not a.skip_fetch:
        # GitHub search returns at most 1,000 results (40 pages x 25) per query, and label:rocm has
        # several thousand PRs, so every query is split into calendar-month windows.
        queries = {}
        for lo, hi in month_windows(a.since):
            m = lo[:7]
            queries[f"merged-{m}"] = f"is:pr is:merged label:rocm merged:{lo}..{hi} sort:created-desc"
            queries[f"closed-{m}"] = f"is:pr is:closed is:unmerged label:rocm closed:{lo}..{hi} sort:created-desc"
            queries[f"open-{m}"] = f"is:pr is:open label:rocm created:{lo}..{hi} sort:created-desc"
        queries["open-older"] = f"is:pr is:open label:rocm created:<{a.since} sort:updated-desc"
        this_month = datetime.date.today().strftime("%Y-%m")
        nums = set()
        for name, q in queries.items():
            out = os.path.join(DATA, "lists", f"{name}.json")
            # Finished months don't change much; re-list only the current month, open PRs and missing windows.
            if not (os.path.exists(out) and this_month not in name and not name.startswith("open")):
                sh([sys.executable, "scrape_list_prs.py", q, out, "", str(a.max_pages)])
                if len(json.load(open(out, encoding="utf-8"))) >= 1000:
                    print(f"WARNING: {name} hit the 1,000-result cap; split this window further", file=sys.stderr)
            for k, v in json.load(open(out, encoding="utf-8")).items():
                if v["author"].endswith("[bot]"): continue
                if v["comments"] >= 2: nums.add(int(k))
        numfile = os.path.join(DATA, "nums.txt"); open(numfile, "w", encoding="utf-8").write("\n".join(map(str, sorted(nums, reverse=True))))
        print(f"{len(nums)} PRs to fetch (cached ones are skipped)", file=sys.stderr)
        sh([sys.executable, "scrape_fetch_prs.py", numfile])
    sh([sys.executable, "derive_personas.py"])
    sh([sys.executable, "build_html.py"])

if __name__ == "__main__":
    main()
