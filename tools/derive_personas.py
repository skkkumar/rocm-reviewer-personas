#!/usr/bin/env python3
"""Rebuild quotes, derived profiles, outcomes and the drift report from data/prs/*.json + data/lists/*.json.

Writes:
  data/quotes.json      every substantive comment per reviewer
  data/derived.json     the computed numbers behind each profile
  data/outcomes.json    per-file history used by tools/outcome.py and the web pre-flight
  data/drift.md         reviewer theme shifts, last 60 days vs earlier
  reviewers/<handle>.md the two auto sections: '## Real review comments ...' and '## Data-derived profile ...'

Hand-written sections above the auto sections are left untouched. Inside '## Data-derived profile' a few
lines are hand-curated and are carried over from the existing file on every run (they cannot be computed):
  '- **Pet models / configs to pre-test:** ...', '- **Their own PRs look like:** ...',
  the bullets under '### Approval triggers' (before 'Last thing they asked ...'),
  the bullets under '### Things they let slide' (before 'Comments they themselves marked ...').
Edit those lines by hand in the persona file; they survive refreshes.
"""
import json, glob, os, re, collections, statistics, datetime
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); DATA = os.path.join(ROOT, "data"); REV = os.path.join(ROOT, "reviewers")
R = ["tjtanaa","AndreasKaratzas","dllehr-amd","shen-shanshan","simondanielsson","Rohan138","hongxiayang","fxmarty-amd","JartX","Fangzhou-Ai","gshtras","mawong-amd","kliuae"]
TH = {"accuracy": r"gsm8k|lm.?eval|accuracy", "e2e perf": r"end.?to.?end|e2e|sweep|isl|osl|ttft|tpot|throughput|benchmark", "blast radius": r"cuda|nvidia|other model|other platform|affect", "init not forward": r"hot.?path|forward|__init__|cache", "platform guard": r"skipif|guard|is_rocm|gfx950|on_gfx|cdna", "tests": r"test group|test-amd|unit test|parametri|toleran|seed", "reuse": r"re-?use|duplicat|centraliz", "regression": r"regress", "flags": r"env ?var|flag|--moe-backend", "aiter pin": r"aiter (version|v0|bump|pin)|in aiter", "comments cleanup": r"remove (this |the )?comment|too many comments", "graph/warmup/invariance": r"capture|warmup|graph|invarian", "rebase/precommit": r"rebase|pre-?commit|dco|sign.?off"}
BOTS = {"gemini-code-assist","github-actions","mergify","claude","depthfirst-app","github-project-automation","coderabbitai","vllm-agent","Copilot","chatgpt-codex-connector","cursor"}
# Model names as reviewers write them. Extend when new model families show up.
MODELS = r"DeepSeek-V\d(?:\.\d)?|DSv\d(?:\.\d)?|DSV\d(?:\.\d)?|DS\d\.\d|Kimi-K\d(?:\.\d)?|\bK\d\b|MiniMax-M\d(?:\.\d)?|GLM-\d(?:\.\d)?|Qwen\d(?:\.\d)?|gpt-oss|Llama(?:-\d(?:\.\d)?)?|Gemma\d?|Hy\d|\bMTP\b|Mixtral|Mistral"
CURATED_LINES = ("- **Pet models / configs to pre-test:**", "- **Their own PRs look like:**")

def load():
    lists = {}
    for f in glob.glob(os.path.join(DATA, "lists", "*.json")):
        for k, v in json.load(open(f, encoding="utf-8")).items(): lists[int(k)] = v
    prs = {}
    for p in glob.glob(os.path.join(DATA, "prs", "*.json")):
        d = json.load(open(p, encoding="utf-8")); prs[d["meta"]["num"]] = d
    if not prs:
        raise SystemExit("No cached PR pages in data/prs/. Run tools/refresh.py --since <date> (without --skip-fetch) first; "
                         "re-deriving from an empty cache would wipe the quotes and profiles in reviewers/*.md.")
    return lists, prs

def dt(s):
    try: return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception: return None

def day(s):
    return dt(s + "T00:00:00Z") if s else None

def is_human(a):
    return a and a not in BOTS and not a.endswith("[bot]")

def clean(b):
    return re.sub(r"\s+", " ", re.sub(r"\[quote\].*?\[/quote\]", "", b)).strip()

def med(xs, nd=1):
    return round(statistics.median(xs), nd) if xs else None

def curated(text):
    """Pull the hand-curated pieces out of an existing '## Data-derived profile' section."""
    auto = text.split("\n## Data-derived profile", 1)
    if len(auto) < 2: return {}
    lines = auto[1].splitlines(); out = {"lines": {}, "approval": [], "slide": []}
    for ln in lines:
        for key in CURATED_LINES:
            if ln.startswith(key): out["lines"][key] = ln
    def block(head, stop):
        got, on = [], False
        for ln in lines:
            if ln.startswith(head): on = True; continue
            if on and (ln.startswith(stop) or ln.startswith("### ")): break
            if on and ln.startswith("- "): got.append(ln)
        return got
    out["approval"] = block("### Approval triggers", "Last thing")
    out["slide"] = block("### Things they let slide", "Comments they themselves")
    return out

def outcomes(lists, prs):
    by = collections.defaultdict(list)
    for n, d in prs.items():
        L = lists.get(n, {}); author = L.get("author")
        human = [c for c in d["comments"] if is_human(c["author"]) and not c["body"].lower().startswith(("/ci", "/amd-ci"))]
        others = [c for c in human if c["author"] != author]
        files = {os.path.basename(c["file"]) for c in d["comments"] if c.get("file")}
        if not files: continue
        created, merged = day(L.get("created")), day(L.get("merged"))
        state = "merged" if L.get("merged") else ("open" if L.get("state", "").upper() == "OPEN" else "closed")
        rounds = len({c["time"][:10] for c in others if c["time"]})
        themes = collections.Counter(k for c in others for k, rx in TH.items() if re.search(rx, c["body"].lower()))
        rec = {"num": n, "state": state, "days": (merged - created).days if merged and created else None,
               "comments": len(human), "rounds": rounds, "reviewers": collections.Counter(c["author"] for c in others), "themes": themes}
        for f in files: by[f].append(rec)
    res = {}
    for f, rs in by.items():
        if len(rs) < 2: continue
        rv, th = collections.Counter(), collections.Counter()
        for r in rs: rv.update(r["reviewers"]); th.update(r["themes"])
        days = [r["days"] for r in rs if r["days"] is not None]
        ex = sorted(rs, key=lambda r: -r["num"])[:3]
        res[f] = {"n_prs": len(rs), "merged": sum(r["state"] == "merged" for r in rs), "closed": sum(r["state"] == "closed" for r in rs),
                  "median_days_to_merge": int(statistics.median(days)) if days else None, "median_comments": med([r["comments"] for r in rs]),
                  "median_rounds": med([r["rounds"] for r in rs]), "reviewers": rv.most_common(8), "themes": th.most_common(8),
                  "examples": [[r["num"], r["days"], r["state"]] for r in ex if r["state"] == "merged"]}
    return dict(sorted(res.items(), key=lambda kv: -kv[1]["n_prs"]))

def main():
    lists, prs = load()
    now = datetime.datetime.now(datetime.timezone.utc)
    quotes = {r: [] for r in R}; derived = {}; drift = []
    bad = re.compile(r"^(lgtm|done|thanks|thank you|ok|/ci|/amd-ci|yes|fixed|resolved|removed|updated|cc )", re.I)
    for r in R:
        first = []; to_appr = []; appr_merge = []; paths = collections.Counter(); dirs = collections.Counter(); ment = collections.Counter()
        models = collections.Counter(); recent = collections.Counter(); old = collections.Counter(); nrec = nold = 0
        blocks = []; lastask = []; nits = []
        own = [n for n, L in lists.items() if L.get("author") == r]
        for n, d in prs.items():
            L = lists.get(n, {})
            if L.get("author") == r: continue
            mine = [c for c in d["comments"] if c["author"] == r and not c["body"].lower().startswith(("/ci", "/amd-ci"))]
            for c in mine:
                b = clean(c["body"])
                if len(b) >= 70 and not bad.match(b) and "Purpose" not in b[:30]:
                    quotes[r].append({"pr": n, "title": L.get("title", "")[:80], "file": (c.get("file") or "").split("/")[-1], "type": c["type"], "text": b[:420], "time": c["time"]})
                if re.match(r"\s*nits?\b", c["body"], re.I): nits.append((n, b[:200]))
                if c.get("file"):
                    paths[os.path.basename(c["file"])] += 1
                    if not c["file"].startswith("..."): dirs["/".join(c["file"].split("/")[:3])] += 1
                for m in re.findall(r"@([A-Za-z0-9-]+)", c["body"]):
                    if m not in (r, L.get("author"), "claude"): ment[m] += 1
                for m in re.findall(MODELS, b): models[m] += 1
                t = dt(c["time"]) if c["time"] else None
                if t:
                    recent_c = (now - t).days <= 60
                    bucket = recent if recent_c else old
                    if recent_c: nrec += 1
                    else: nold += 1
                    for k, rx in TH.items():
                        if re.search(rx, c["body"].lower()): bucket[k] += 1
            times = [t for t in (dt(c["time"]) for c in mine if c["time"]) if t]
            apps = [dt(x["time"]) for x in d["reviews"] if x["author"] == r and x["state"] == "approved" and x["time"]]
            chg = [dt(x["time"]) for x in d["reviews"] if x["author"] == r and x["state"] == "changes_requested" and x["time"]]
            if L.get("created") and times: first.append((min(times) - day(L["created"])).total_seconds() / 86400)
            if times and apps: to_appr.append(max(0, (min(apps) - min(times)).total_seconds() / 86400))
            if apps and L.get("merged"): appr_merge.append(max(0, (day(L["merged"]) - min(apps)).total_seconds() / 86400))
            if apps and mine:
                before = [c for c in mine if c["time"] and dt(c["time"]) <= min(apps) + datetime.timedelta(minutes=2)]
                if before and len(before[-1]["body"]) > 25 and not before[-1]["body"].lower().startswith("lgtm"): lastask.append((n, re.sub(r"\s+", " ", before[-1]["body"])[:220]))
            if chg:
                for c in [c for c in mine if c["time"] and abs((dt(c["time"]) - min(chg)).total_seconds()) < 3 * 3600][:3]: blocks.append((n, L.get("title", "")[:60], re.sub(r"\s+", " ", c["body"])[:260]))
        shift = [(k, round(old[k] / max(nold, 1) * 100), round(recent[k] / max(nrec, 1) * 100)) for k in TH if nrec >= 15 and abs(recent[k] / max(nrec, 1) - old[k] / max(nold, 1)) >= 0.05]
        lastask.sort(key=lambda x: -x[0]); blocks.sort(key=lambda x: -x[0]); nits.sort(key=lambda x: -x[0])
        derived[r] = {"n_prs_commented": len(first), "first_response_days_median": med(first), "first_comment_to_approval_days_median": med(to_appr),
                      "approval_to_merge_days_median": med(appr_merge), "top_files": paths.most_common(8), "top_dirs": dirs.most_common(6),
                      "models_named": models.most_common(6), "escalates_to": ment.most_common(5), "own_prs": len(own),
                      "recent_shift": sorted(shift, key=lambda x: -abs(x[2] - x[1])), "recent_n": nrec, "old_n": nold,
                      "last_ask_before_approval": lastask[:8], "blocking_reasons": blocks[:8], "nits": nits[:6]}
        for k, b, a in derived[r]["recent_shift"]: drift.append(f"- {r}: {k} {b}% -> {a}% ({nrec} recent vs {nold} earlier comments)")
    out = outcomes(lists, prs)
    json.dump(quotes, open(os.path.join(DATA, "quotes.json"), "w", encoding="utf-8"), indent=0, ensure_ascii=False)
    json.dump(derived, open(os.path.join(DATA, "derived.json"), "w", encoding="utf-8"), indent=1, default=str, ensure_ascii=False)
    json.dump({"built": now.date().isoformat(), "n_prs": len(prs), "by_file": out}, open(os.path.join(DATA, "outcomes.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    open(os.path.join(DATA, "drift.md"), "w", encoding="utf-8").write(f"# Reviewer drift (last 60 days vs earlier, computed {now.date()})\n\n" + ("\n".join(drift) or "- no shifts over 5 points") + "\n")
    # rewrite auto sections
    for r in R:
        p = os.path.join(REV, r + ".md")
        if not os.path.exists(p): continue
        full = open(p, encoding="utf-8").read(); cur = curated(full)
        txt = full.split("\n\n## Real review comments")[0].split("\n\n## Data-derived profile")[0].rstrip()
        qs = sorted(quotes[r], key=lambda q: (-(q["type"] == "review_comment"), -len(q["text"])))
        cnt = collections.Counter(); sel = []
        for q in qs:
            if cnt[q["pr"]] >= 3: continue
            cnt[q["pr"]] += 1; sel.append(q)
            if len(sel) >= 30: break
        qtxt = "\n".join(f"- **#{q['pr']}** ({q['title']}) — {('`'+q['file']+'`') if q['file'] else 'PR thread'}: \"{q['text']}\"" for q in sel) or "- (no comments in the data set)"
        d = derived[r]; cl = cur.get("lines", {})
        prof = (f"- **PRs commented on (not own):** {d['n_prs_commented']}\n"
                f"- **Median time to first comment after PR open:** {d['first_response_days_median']} days\n"
                f"- **Median time from first comment to approval:** {d['first_comment_to_approval_days_median']} days\n"
                f"- **Median time from approval to merge:** {d['approval_to_merge_days_median']} days\n")
        if d["top_dirs"]: prof += "- **Where they comment (paths):** " + ", ".join(f"`{p_}` ({n})" for p_, n in d["top_dirs"]) + "\n"
        if d["top_files"]: prof += "- **Files they comment on most:** " + ", ".join(f"`{p_}` ({n})" for p_, n in d["top_files"][:6]) + "\n"
        if d["models_named"]: prof += "- **Models named in their comments:** " + ", ".join(f"{m} ({n})" for m, n in d["models_named"]) + "\n"
        if CURATED_LINES[0] in cl: prof += cl[CURATED_LINES[0]] + "\n"
        if d["escalates_to"]: prof += "- **Escalates to (people they @mention):** " + ", ".join(f"@{m} ({n})" for m, n in d["escalates_to"]) + ". Pre-tag the first one when your diff is outside this reviewer's area.\n"
        if CURATED_LINES[1] in cl: prof += cl[CURATED_LINES[1]] + "\n"
        prof += f"- **Own PRs in the data set:** {d['own_prs']}\n"
        if d["recent_shift"]: prof += "- **Recent shift (last 60 days vs earlier, share of their comments):** " + "; ".join(f"{k} {b}% -> {a}%" for k, b, a in d["recent_shift"][:4]) + f" (based on {d['recent_n']} recent vs {d['old_n']} earlier comments)\n"
        prof += "\n### Approval triggers (what flipped them to LGTM)\n\n"
        if cur.get("approval"): prof += "\n".join(cur["approval"]) + "\n\n"
        if d["last_ask_before_approval"]: prof += "Last thing they asked for right before approving, in real PRs:\n\n" + "\n".join(f"- #{n}: \"{b}\"" for n, b in d["last_ask_before_approval"]) + "\n"
        if d["blocking_reasons"]: prof += "\n### Changes-requested triggers (their comments at the moment they formally blocked)\n\n" + "\n".join(f"- #{n} ({t}): \"{b}\"" for n, t, b in d["blocking_reasons"]) + "\n"
        prof += "\n### Things they let slide\n\n"
        if cur.get("slide"): prof += "\n".join(cur["slide"]) + "\n\n"
        if d["nits"]: prof += "Comments they themselves marked as nits (do not block on these):\n\n" + "\n".join(f"- #{n}: \"{b}\"" for n, b in d["nits"]) + "\n"
        open(p, "w", encoding="utf-8", newline="\n").write(txt + "\n\n## Real review comments from this reviewer (verbatim, with PR numbers)\n\n" + qtxt + f"\n\n## Data-derived profile (computed from the PR set, {now.date()})\n\n" + prof)
    print(f"updated {len(R)} persona files from {len(prs)} PRs; outcomes for {len(out)} files; see data/drift.md")

if __name__ == "__main__":
    main()
