#!/usr/bin/env python3
"""List vllm-project/vllm PRs matching a GitHub search query -> {num: {...}} JSON.

  python scrape_list_prs.py "<query>" <out.json> [min_date] [max_pages] [date_key]
  e.g. python scrape_list_prs.py "is:pr is:merged label:rocm merged:2026-03-01..2026-03-31" lists/merged-2026-03.json

Primary method: the GitHub search API (https://api.github.com/search/issues). Works without a token at
10 requests/minute; set GITHUB_TOKEN (or GH_TOKEN) for 30/minute. Either way a single query returns at most
1,000 results, so callers (refresh.py) split queries into monthly windows.
Fallback: the old HTML scraper of github.com/<repo>/pulls (no API needed, but GitHub throttles it and markup changes).
"""
import json, os, re, subprocess, sys, time, urllib.error, urllib.parse, urllib.request

REPO = "vllm-project/vllm"
TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

def api_get(url):
    h = {"Accept": "application/vnd.github+json", "User-Agent": "rocm-reviewer-personas"}
    if TOKEN: h["Authorization"] = "Bearer " + TOKEN
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):  # rate limited: wait until the window resets
                reset = int(e.headers.get("x-ratelimit-reset", "0") or 0)
                wait = max(10, min(120, reset - int(time.time()) + 2)) if reset else 30 * (attempt + 1)
                print(f"rate limited, waiting {wait}s", file=sys.stderr); time.sleep(wait); continue
            if e.code == 422: return {"items": []}  # beyond the 1,000-result window
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("GitHub API kept failing for " + url)

def list_prs_api(q, max_pages=10):
    sort = re.search(r"sort:(\w+)-(asc|desc)", q)
    q2 = re.sub(r"\s*sort:\S+", "", q).strip()
    if "repo:" not in q2: q2 = f"repo:{REPO} " + q2
    res, pause = {}, (2.1 if TOKEN else 6.5)
    for p in range(1, min(max_pages, 10) + 1):
        params = {"q": q2, "per_page": 100, "page": p}
        if sort: params.update(sort=sort.group(1), order=sort.group(2))
        d = api_get("https://api.github.com/search/issues?" + urllib.parse.urlencode(params))
        items = d.get("items", [])
        for it in items:
            pr = it.get("pull_request") or {}
            res[it["number"]] = {"num": it["number"], "title": it["title"], "author": (it.get("user") or {}).get("login", ""),
                                 "assoc": it.get("author_association", ""), "created": it["created_at"][:10], "updated": it["updated_at"][:10],
                                 "merged": (pr.get("merged_at") or "")[:10], "closed": (it.get("closed_at") or "")[:10],
                                 "state": "MERGED" if pr.get("merged_at") else it["state"].upper(),
                                 "labels": [l["name"] for l in it.get("labels", [])], "comments": it.get("comments", 0), "query": q}
        print(q, "page", p, len(items), "total", d.get("total_count"), file=sys.stderr)
        if len(items) < 100: break
        time.sleep(pause)
    time.sleep(pause)
    return res

# ---------- HTML fallback ----------
def fetch(url):
    return subprocess.run(["curl", "-sL", "-A", "Mozilla/5.0", "--max-time", "60", url], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout

def list_prs_html(q, max_pages=40, min_date=None, key="updatedAt"):
    from bs4 import BeautifulSoup
    res = {}
    for p in range(1, max_pages + 1):
        url = f"https://github.com/{REPO}/pulls?q={urllib.parse.quote(q)}&page={p}"
        sc = None
        for attempt in range(4):  # GitHub sometimes serves a page without data when throttling
            sc = BeautifulSoup(fetch(url), "lxml").find("script", attrs={"data-target": "react-app.embeddedData"})
            if sc: break
            time.sleep(30 * (attempt + 1))
        if not sc: print("no data page", p, file=sys.stderr); break
        items = json.loads(sc.string)["payload"].get("repoPullsDashboardContentRoute", {}).get("results", [])
        if not items: break
        oldest = None
        for it in items:
            n = it["number"]
            res[n] = {"num": n, "title": it["title"], "author": it["author"]["displayLogin"] if it.get("author") else "",
                      "assoc": (it.get("author") or {}).get("association", "").replace("REPO_PULL_REQUEST_AUTHOR_ASSOCIATION_", ""),
                      "created": it["createdAt"][:10], "updated": it["updatedAt"][:10], "merged": (it.get("mergedAt") or "")[:10],
                      "closed": (it.get("closedAt") or "")[:10], "state": it["state"].replace("REPO_PULL_REQUEST_STATE_", ""),
                      "labels": [l["name"] for l in it.get("labels", [])], "comments": it.get("commentCount", 0), "query": q}
            dd = it[key][:10] if it.get(key) else ""
            oldest = dd if (oldest is None or dd < oldest) else oldest
        print(q, "page", p, len(items), "oldest", oldest, file=sys.stderr)
        if min_date and oldest and oldest < min_date: break
        time.sleep(2.0)
    return res

def list_prs(q, max_pages=40, min_date=None, key="updatedAt"):
    try:
        return list_prs_api(q, max_pages)
    except Exception as e:
        print(f"search API failed ({e}); falling back to HTML scraping", file=sys.stderr)
        return list_prs_html(q, max_pages, min_date, key)

if __name__ == "__main__":
    q = sys.argv[1]; out = sys.argv[2]; min_date = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
    maxp = int(sys.argv[4]) if len(sys.argv) > 4 else 40; key = sys.argv[5] if len(sys.argv) > 5 else "updatedAt"
    r = list_prs(q, maxp, min_date, key)
    json.dump(r, open(out, "w", encoding="utf-8"), indent=1); print(len(r))
