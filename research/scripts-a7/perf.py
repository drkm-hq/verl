"""Step 3: open PRs claiming a speedup or memory reduction, with claimed numbers from the body."""

import json
import re
import sys

sys.path.insert(0, os.path.join(_REPO, "research/scripts-a7"))
from common import age_days, load, title_tags  # noqa: E402
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))  # the repository root

KW = re.compile(r"\b(speed|speeds|speedup|speed-up|faster|throughput|overhead|memory|OOM|fuse|fused|fusion|overlap|overlapping|cache|caching|zero-copy|zero copy|vectorize|vectorized|latency|idle|bubble|prefetch|peak)\b", re.I)
STRONG_TITLE = re.compile(r"\b(speed|speedup|faster|throughput|overhead|memory|OOM|fuse|fused|overlap|cache|zero-copy|vectorize|vectorized|latency|idle|bubble|prefetch|offload|reduce|cut|peak)\b", re.I)
NUM = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:x|×)\s*(?:faster|speedup|speed-up|throughput|less|lower|reduction)?"
    r"|\d+(?:\.\d+)?\s*%\s*(?:faster|slower|speedup|speed-up|reduction|less|lower|improvement|fewer|higher|more)?"
    r"|(?:from|→)?\s*\d+(?:\.\d+)?\s*(?:GB|GiB|MB|MiB)\b(?:\s*(?:->|→|to)\s*\d+(?:\.\d+)?\s*(?:GB|GiB|MB|MiB))?"
    r"|\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds|min)\b\s*(?:->|→|to)\s*\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds|min)\b"
    r"|\d+(?:\.\d+)?\s*(?:tok(?:en)?s?/s|TFLOPs?|MFU)\b)",
    re.I,
)

prs = load("prs_open.json")
rows = []
for p in prs:
    mods, typ, breaking, valid = title_tags(p["title"])
    body = p.get("body") or ""
    title = p["title"]
    is_perf_tag = "perf" in mods
    title_hit = bool(STRONG_TITLE.search(title))
    body_hits = KW.findall(body)
    score = (3 if is_perf_tag else 0) + (2 if title_hit else 0) + min(len(body_hits), 6) / 3
    if not (is_perf_tag or title_hit or len(body_hits) >= 4):
        continue
    # claimed numbers: look at sentences containing a keyword and a number
    claims = []
    for line in body.splitlines():
        if KW.search(line) and NUM.search(line):
            claims.append(re.sub(r"\s+", " ", line.strip())[:220])
    nums = NUM.findall(body)
    rows.append({
        "number": p["number"], "title": title, "age": age_days(p["createdAt"]), "draft": p["isDraft"],
        "author": p["author"]["login"], "perf_tag": is_perf_tag, "title_hit": title_hit,
        "body_kw": len(body_hits), "score": round(score, 1),
        "lines": sum(f["additions"] + f["deletions"] for f in p["files"]), "nfiles": len(p["files"]),
        "has_tests": any(f["path"].startswith("tests/") for f in p["files"]),
        "claims": claims[:6], "nums": [n.strip() for n in nums[:8]],
        "files": [f["path"] for f in p["files"] if f["path"].startswith("verl/")][:8],
    })

rows.sort(key=lambda r: (-r["score"], r["age"]))
print(f"# {len(rows)} open PRs with perf/memory claims (perf tag, perf words in title, or >=4 perf words in body)")
for r in rows:
    print(f"\n#{r['number']} ({r['age']}d{', draft' if r['draft'] else ''}, {r['author']}, {r['lines']} lines/{r['nfiles']} files, tests={'y' if r['has_tests'] else 'n'}) {r['title']}")
    print("   files:", ", ".join(r["files"][:5]))
    if r["nums"]:
        print("   numbers:", "; ".join(r["nums"]))
    for c in r["claims"][:4]:
        print("   >", c)

with open(os.path.join(_REPO, "research/scripts-a7/out_perf.json"), "w") as f:
    json.dump(rows, f, indent=1)
