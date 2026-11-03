#!/usr/bin/env python3
"""Re-embed reviewers/*.md, _common.md, REPLY_PROMPT.md and data/outcomes.json into index.html.

Run after refresh.py / derive_personas.py so the web tool uses the same data as the CLI tools:
  python tools/build_html.py
"""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)

def read(*p): return open(os.path.join(ROOT, *p), encoding="utf-8").read()

def main():
    path = os.path.join(ROOT, "index.html"); html = read("index.html")
    pat = re.compile(r'(<script id="data" type="application/json">)(.*?)(</script>)', re.S)
    data = json.loads(pat.search(html).group(2))
    known = {r["handle"] for r in data["reviewers"]}
    for fn in sorted(os.listdir(os.path.join(ROOT, "reviewers"))):
        h = fn[:-3]
        if fn.endswith(".md") and not fn.startswith("_") and h not in known:
            fm = dict(re.findall(r"^(\w+):\s*(.+)$", read("reviewers", fn).split("\n---", 1)[0], re.M))
            data["reviewers"].append({"handle": h, "label": f"{fm.get('display_name', h)} - {fm.get('focus', '')}".strip(" -")})
            print("added new reviewer", h)
    for h in [r["handle"] for r in data["reviewers"]]:
        data["personas"][h] = read("reviewers", h + ".md")
    data["common"] = read("reviewers", "_common.md")
    m = re.search(r"```\n(.*?)```", read("REPLY_PROMPT.md"), re.S)
    if m: data["reply"] = "\n" + m.group(1)
    data["outcomes"] = json.loads(read("data", "outcomes.json"))["by_file"]
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\/")
    html = pat.sub(lambda mm: mm.group(1) + blob + mm.group(3), html, count=1)
    open(path, "w", encoding="utf-8", newline="\n").write(html)
    print(f"index.html: {len(data['personas'])} personas, {len(data['outcomes'])} files of outcome history")

if __name__ == "__main__":
    main()
