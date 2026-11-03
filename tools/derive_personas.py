#!/usr/bin/env python3
"""Rebuild quotes, derived profiles, outcomes and the drift report from data/prs/*.json.
Regenerates the two auto sections of each reviewers/<handle>.md:
  '## Real review comments from this reviewer' and '## Data-derived profile'.
Hand-written sections above them are left untouched.
"""
import json, glob, os, re, collections, statistics, datetime
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); DATA = os.path.join(ROOT, "data"); REV = os.path.join(ROOT, "reviewers")
R = ["tjtanaa","AndreasKaratzas","dllehr-amd","shen-shanshan","simondanielsson","Rohan138","hongxiayang","fxmarty-amd","JartX","Fangzhou-Ai","gshtras","mawong-amd","kliuae"]
TH = {"accuracy": r"gsm8k|lm.?eval|accuracy", "e2e perf": r"end.?to.?end|e2e|sweep|isl|osl|ttft|tpot|throughput|benchmark", "blast radius": r"cuda|nvidia|other model|other platform|affect", "init not forward": r"hot.?path|forward|__init__|cache", "platform guard": r"skipif|guard|is_rocm|gfx950|on_gfx|cdna", "tests": r"test group|test-amd|unit test|parametri|toleran|seed", "reuse": r"re-?use|duplicat|centraliz", "regression": r"regress", "flags": r"env ?var|flag|--moe-backend", "aiter pin": r"aiter (version|v0|bump|pin)|in aiter", "comments cleanup": r"remove (this |the )?comment|too many comments", "graph/warmup/invariance": r"capture|warmup|graph|invarian"}
BOTS = {"gemini-code-assist","github-actions","mergify","claude","depthfirst-app","github-project-automation","coderabbitai","vllm-agent","Copilot","chatgpt-codex-connector","cursor"}

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

def main():
    lists, prs = load()
    now = datetime.datetime.now(datetime.timezone.utc)
    quotes = {r: [] for r in R}; derived = {}; drift = []
    bad = re.compile(r"^(lgtm|done|thanks|thank you|ok|/ci|/amd-ci|yes|fixed|resolved|removed|updated|cc )", re.I)
    for r in R:
        first = []; to_appr = []; paths = collections.Counter(); dirs = collections.Counter(); ment = collections.Counter(); recent = collections.Counter(); old = collections.Counter(); nrec = nold = 0; blocks = []; lastask = []
        for n, d in prs.items():
            L = lists.get(n, {})
            if L.get("author") == r: continue
            mine = [c for c in d["comments"] if c["author"] == r and not c["body"].lower().startswith(("/ci", "/amd-ci"))]
            for c in mine:
                b = re.sub(r"\s+", " ", re.sub(r"\[quote\].*?\[/quote\]", "", c["body"])).strip()
                if len(b) >= 70 and not bad.match(b) and "Purpose" not in b[:30]:
                    quotes[r].append({"pr": n, "title": L.get("title", "")[:80], "file": (c.get("file") or "").split("/")[-1], "type": c["type"], "text": b[:420]})
                if c.get("file"): paths[os.path.basename(c["file"])] += 1; dirs["/".join(c["file"].split("/")[:3])] += 1
                for m in re.findall(r"@([A-Za-z0-9-]+)", c["body"]):
                    if m not in (r, L.get("author"), "claude"): ment[m] += 1
                t = dt(c["time"]) if c["time"] else None
                if t:
                    bucket = recent if (now - t).days <= 60 else old
                    if (now - t).days <= 60: nrec += 1
                    else: nold += 1
                    for k, rx in TH.items():
                        if re.search(rx, c["body"].lower()): bucket[k] += 1
            times = [dt(c["time"]) for c in mine if c["time"]]; times = [t for t in times if t]
            apps = [dt(x["time"]) for x in d["reviews"] if x["author"] == r and x["state"] == "approved" and x["time"]]
            chg = [dt(x["time"]) for x in d["reviews"] if x["author"] == r and x["state"] == "changes_requested" and x["time"]]
            if L.get("created") and times: first.append((min(times) - dt(L["created"] + "T00:00:00Z")).total_seconds() / 86400)
            if times and apps: to_appr.append(max(0, (min(apps) - min(times)).total_seconds() / 86400))
            if apps and mine:
                before = [c for c in mine if c["time"] and dt(c["time"]) <= min(apps) + datetime.timedelta(minutes=2)]
                if before and len(before[-1]["body"]) > 25 and not before[-1]["body"].lower().startswith("lgtm"): lastask.append((n, re.sub(r"\s+", " ", before[-1]["body"])[:220]))
            if chg:
                for c in [c for c in mine if c["time"] and abs((dt(c["time"]) - min(chg)).total_seconds()) < 3 * 3600][:3]: blocks.append((n, L.get("title", "")[:60], re.sub(r"\s+", " ", c["body"])[:260]))
        shift = [(k, round(old[k] / max(nold, 1) * 100), round(recent[k] / max(nrec, 1) * 100)) for k in TH if nrec >= 15 and abs(recent[k] / max(nrec, 1) - old[k] / max(nold, 1)) >= 0.05]
        derived[r] = {"first_response_days_median": round(statistics.median(first), 1) if first else None, "first_comment_to_approval_days_median": round(statistics.median(to_appr), 1) if to_appr else None, "n_prs_commented": len(first), "top_files": paths.most_common(8), "top_dirs": dirs.most_common(6), "escalates_to": ment.most_common(5), "recent_shift": sorted(shift, key=lambda x: -abs(x[2] - x[1])), "recent_n": nrec, "old_n": nold, "last_ask_before_approval": lastask[:8], "blocking_reasons": blocks[:8]}
        for k, b, a in derived[r]["recent_shift"]: drift.append(f"- {r}: {k} {b}% -> {a}%")
    json.dump(quotes, open(os.path.join(DATA, "quotes.json"), "w", encoding="utf-8"), indent=0)
    json.dump(derived, open(os.path.join(DATA, "derived.json"), "w", encoding="utf-8"), indent=1, default=str)
    open(os.path.join(DATA, "drift.md"), "w", encoding="utf-8").write("# Reviewer drift (last 60 days vs earlier)\n\n" + ("\n".join(drift) or "- no shifts over 5 points") + "\n")
    # rewrite auto sections
    for r in R:
        p = os.path.join(REV, r + ".md")
        if not os.path.exists(p): continue
        txt = open(p, encoding="utf-8").read().split("\n\n## Real review comments")[0].split("\n\n## Data-derived profile")[0].rstrip()
        qs = sorted(quotes[r], key=lambda q: (-(q["type"] == "review_comment"), -len(q["text"])))
        cnt = collections.Counter(); sel = []
        for q in qs:
            if cnt[q["pr"]] >= 3: continue
            cnt[q["pr"]] += 1; sel.append(q)
            if len(sel) >= 30: break
        qtxt = "\n".join(f"- **#{q['pr']}** ({q['title']}) — {('`'+q['file']+'`') if q['file'] else 'PR thread'}: \"{q['text']}\"" for q in sel) or "- (no comments in the data set)"
        d = derived[r]
        prof = f"- **PRs commented on (not own):** {d['n_prs_commented']}\n- **Median time to first comment:** {d['first_response_days_median']} days\n- **Median first comment to approval:** {d['first_comment_to_approval_days_median']} days\n"
        if d["top_dirs"]: prof += "- **Where they comment:** " + ", ".join(f"`{p_}` ({n})" for p_, n in d["top_dirs"]) + "\n"
        if d["escalates_to"]: prof += "- **Escalates to:** " + ", ".join(f"@{m} ({n})" for m, n in d["escalates_to"]) + "\n"
        if d["recent_shift"]: prof += "- **Recent shift:** " + "; ".join(f"{k} {b}% -> {a}%" for k, b, a in d["recent_shift"][:4]) + "\n"
        if d["last_ask_before_approval"]: prof += "\nLast ask before approving:\n\n" + "\n".join(f"- #{n}: \"{b}\"" for n, b in d["last_ask_before_approval"]) + "\n"
        if d["blocking_reasons"]: prof += "\n### Changes-requested triggers\n\n" + "\n".join(f"- #{n} ({t}): \"{b}\"" for n, t, b in d["blocking_reasons"]) + "\n"
        open(p, "w", encoding="utf-8").write(txt + "\n\n## Real review comments from this reviewer (verbatim, with PR numbers)\n\n" + qtxt + f"\n\n## Data-derived profile (computed {now.date()})\n\n" + prof)
    print("updated", len(R), "persona files; see data/drift.md")

if __name__ == "__main__":
    main()
