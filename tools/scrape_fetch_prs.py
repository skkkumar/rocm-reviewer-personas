import re, sys, json, time, os, subprocess, html as htmlmod
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

OUT = os.environ.get("ROCM_REVIEW_PRS", "data/prs")
os.makedirs(OUT, exist_ok=True)

def fetch(url, tries=4):
    for t in range(tries):
        r = subprocess.run(["curl","-sL","-A","Mozilla/5.0 (X11; Linux x86_64)","--max-time","90","-w","\n%{http_code}",url],capture_output=True,text=True,encoding="utf-8",errors="replace")
        body, _, code = r.stdout.rpartition("\n")
        if code == "200" and len(body) > 1000: return body
        time.sleep(5 * (t+1))
    return ""

def text_of(el):
    if el is None: return ""
    for q in el.select("blockquote"):
        q.replace_with("[quote] " + q.get_text(" ", strip=True)[:200] + " [/quote]")
    t = el.get_text("\n", strip=True)
    return re.sub(r"\n{2,}", "\n", t)

def parse_fragment(soup, items):
    for c in soup.select(".timeline-comment, .review-comment"):
        a = c.select_one(".author"); body = c.select_one(".comment-body")
        if not a or not body: continue
        ts = c.select_one("relative-time")
        cls = c.get("class", [])
        is_review = "review-comment" in cls
        item = {"author": text_of(a), "time": ts.get("datetime") if ts else "",
                "body": text_of(body)[:5000], "type": "review_comment" if is_review else "comment",
                "minimized": "minimized-comment" in cls and "unminimized-comment" not in cls}
        if is_review:
            cont = c.find_parent(class_=re.compile("js-resolvable-timeline-thread-container"))
            fpath = None
            if cont:
                item["resolved"] = cont.get("data-resolved") == "true"
                n = cont
                for _ in range(6):
                    n = n.parent
                    if n is None: break
                    fh = n.select_one("a.text-mono, .file-header a, [data-path]")
                    if fh: fpath = fh.get("data-path") or text_of(fh); break
                outd = cont.select_one(".outdated-comment-label, .Label--outdated") or cont.find(string=re.compile("Outdated"))
                item["outdated"] = bool(outd)
            item["file"] = fpath
            item["suggestion"] = bool(c.select_one(".js-suggested-change, .suggested-change, .blob-code-suggestion, .js-apply-suggestion"))
        items.append(item)

def parse_reviews(soup, reviews, meta):
    for ev in soup.select(".js-timeline-item, .TimelineItem"):
        txt = ev.get_text(" ", strip=True)[:200]
        a = ev.select_one(".author")
        if not a: continue
        if "approved these changes" in txt or "requested changes" in txt or "dismissed" in txt:
            ts = ev.select_one("relative-time")
            st = "approved" if "approved these changes" in txt else ("changes_requested" if "requested changes" in txt else "dismissed")
            rv = {"author": text_of(a), "state": st, "time": ts.get("datetime") if ts else ""}
            if rv not in reviews: reviews.append(rv)
        if "merged commit" in txt and "merged_by" not in meta:
            meta["merged_by"] = text_of(a)
        if "closed this" in txt and "closed_by" not in meta:
            meta["closed_by"] = text_of(a)
        m = re.search(r"added the (.+?) label", txt)
    return reviews

def get_pr(num):
    path = f"{OUT}/{num}.json"
    if os.path.exists(path): return
    url = f"https://github.com/vllm-project/vllm/pull/{num}"
    h = fetch(url)
    if not h: print("FAIL", num, file=sys.stderr); return
    soup = BeautifulSoup(h, "lxml")
    meta = {"num": num}
    t = soup.select_one("h1 span.markdown-title, .js-issue-title")
    meta["title"] = text_of(t)
    fc = soup.select_one("#files_tab_counter, [data-tab-item='files'] .Counter")
    meta["files_changed"] = text_of(fc)
    ds = soup.select_one(".diffstat")
    meta["diffstat"] = text_of(ds)
    items, reviews = [], []
    parse_fragment(soup, items); parse_reviews(soup, reviews, meta)
    # hidden items
    seen = set()
    for _ in range(12):
        m = re.search(r'action="(/vllm-project/vllm/pull/\d+/timeline_more_items\?[^"]+)"', h)
        if not m: break
        u = htmlmod.unescape(m.group(1))
        if u in seen: break
        seen.add(u)
        h = fetch("https://github.com" + u)
        if not h: break
        s2 = BeautifulSoup(h, "lxml")
        parse_fragment(s2, items); parse_reviews(s2, reviews, meta)
    # dedupe
    uniq, out = set(), []
    for it in items:
        k = (it["author"], it["time"], it["body"][:120])
        if k in uniq: continue
        uniq.add(k); out.append(it)
    out.sort(key=lambda x: x["time"] or "9")
    json.dump({"meta": meta, "comments": out, "reviews": reviews}, open(path, "w", encoding="utf-8"), indent=0)
    print(num, len(out), len(reviews), len(seen), file=sys.stderr)
    time.sleep(0.5)

# ---------- API mode (used when GITHUB_TOKEN / GH_TOKEN is set) ----------
# Same output schema as the HTML scraper, but exact: full file paths, outdated flags, review states and bodies.
# Costs 4+ requests per PR, so it needs a token (5,000 requests/hour); unauthenticated is only 60/hour.
TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

def api(path):
    import urllib.request, urllib.error
    out, url = [], "https://api.github.com/repos/vllm-project/vllm/" + path
    while url:
        hdr = {"Accept": "application/vnd.github+json", "User-Agent": "rocm-reviewer-personas"}
        if TOKEN: hdr["Authorization"] = "Bearer " + TOKEN
        req = urllib.request.Request(url, headers=hdr)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = json.loads(r.read().decode("utf-8")); link = r.headers.get("Link", "")
                break
            except urllib.error.HTTPError as e:
                if e.code in (403, 429):
                    reset = int(e.headers.get("x-ratelimit-reset", "0") or 0)
                    time.sleep(max(10, min(900, reset - int(time.time()) + 2)) if reset else 60); continue
                raise
        else:
            raise RuntimeError("GitHub API kept failing for " + url)
        if isinstance(data, dict): return data
        out += data
        m = re.search(r'<([^>]+)>;\s*rel="next"', link); url = m.group(1) if m else None
    return out

def get_pr_api(num):
    path = f"{OUT}/{num}.json"
    if os.path.exists(path): return
    try:
        pr = api(f"pulls/{num}")
        rc = api(f"pulls/{num}/comments?per_page=100"); ic = api(f"issues/{num}/comments?per_page=100"); rv = api(f"pulls/{num}/reviews?per_page=100")
    except Exception as e:
        print("FAIL", num, e, file=sys.stderr); return
    login = lambda u: (u or {}).get("login", "")
    meta = {"num": num, "title": pr.get("title", ""), "files_changed": str(pr.get("changed_files", "")),
            "diffstat": f"+{pr.get('additions', 0)} -{pr.get('deletions', 0)}"}
    if pr.get("merged_by"): meta["merged_by"] = login(pr["merged_by"])
    items = []
    for c in rc:
        items.append({"author": login(c.get("user")), "time": c.get("created_at", ""), "body": (c.get("body") or "")[:5000], "type": "review_comment",
                      "minimized": False, "file": c.get("path"), "outdated": c.get("position") is None, "resolved": None,
                      "suggestion": "```suggestion" in (c.get("body") or "")})
    for c in ic:
        items.append({"author": login(c.get("user")), "time": c.get("created_at", ""), "body": (c.get("body") or "")[:5000], "type": "comment", "minimized": False})
    reviews = []
    st = {"APPROVED": "approved", "CHANGES_REQUESTED": "changes_requested", "DISMISSED": "dismissed"}
    for r in rv:
        if r.get("body"):
            items.append({"author": login(r.get("user")), "time": r.get("submitted_at") or "", "body": r["body"][:5000], "type": "comment", "minimized": False})
        if r.get("state") in st:
            reviews.append({"author": login(r.get("user")), "state": st[r["state"]], "time": r.get("submitted_at") or ""})
    items.sort(key=lambda x: x["time"] or "9")
    json.dump({"meta": meta, "comments": items, "reviews": reviews}, open(path, "w", encoding="utf-8"), indent=0)
    print(num, len(items), len(reviews), "api", file=sys.stderr)

if __name__ == "__main__":
    nums = [int(x) for x in open(sys.argv[1], encoding="utf-8").read().split()]
    print("mode:", "GitHub API (token)" if TOKEN else "HTML pages (no token)", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(get_pr_api if TOKEN else get_pr, nums))
