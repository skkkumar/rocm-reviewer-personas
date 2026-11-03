import re, sys, json, time, urllib.parse, subprocess
from bs4 import BeautifulSoup
def fetch(url):
    return subprocess.run(["curl","-sL","-A","Mozilla/5.0","--max-time","60",url],capture_output=True,text=True,encoding="utf-8",errors="replace").stdout
def list_prs(q, max_pages=40, min_date=None, key="updatedAt"):
    res = {}
    for p in range(1, max_pages+1):
        url = f"https://github.com/vllm-project/vllm/pulls?q={urllib.parse.quote(q)}&page={p}"
        h = fetch(url); s = BeautifulSoup(h, "lxml")
        sc = s.find("script", attrs={"data-target":"react-app.embeddedData"})
        if not sc: print("no data page",p,file=sys.stderr); break
        d = json.loads(sc.string)
        items = d["payload"].get("repoPullsDashboardContentRoute",{}).get("results",[])
        if not items: break
        oldest=None
        for it in items:
            n=it["number"]
            res[n]={"num":n,"title":it["title"],"author":it["author"]["displayLogin"] if it.get("author") else "",
                    "assoc":(it.get("author") or {}).get("association","").replace("REPO_PULL_REQUEST_AUTHOR_ASSOCIATION_",""),
                    "created":it["createdAt"][:10],"updated":it["updatedAt"][:10],"merged":(it.get("mergedAt") or "")[:10],
                    "closed":(it.get("closedAt") or "")[:10],"state":it["state"].replace("REPO_PULL_REQUEST_STATE_",""),
                    "labels":[l["name"] for l in it.get("labels",[])],"comments":it.get("commentCount",0),"query":q}
            dd=it[key][:10] if it.get(key) else ""
            oldest = dd if (oldest is None or dd<oldest) else oldest
        print(q,"page",p,len(items),"oldest",oldest,file=sys.stderr)
        if min_date and oldest and oldest<min_date: break
        time.sleep(1.0)
    return res
if __name__=="__main__":
    q=sys.argv[1]; out=sys.argv[2]; min_date=sys.argv[3] if len(sys.argv)>3 else None
    maxp=int(sys.argv[4]) if len(sys.argv)>4 else 40; key=sys.argv[5] if len(sys.argv)>5 else "updatedAt"
    r=list_prs(q,maxp,min_date,key); json.dump(r,open(out,"w", encoding="utf-8"),indent=1); print(len(r))
