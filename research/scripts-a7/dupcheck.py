"""Local duplicate check over the open-PR dump (no GitHub API).

Usage:
  python3 -I research/scripts-a7/dupcheck.py --kw "Karmarkar" --kw seqlen_balancing --path verl/utils/seqlen_balancing.py
  python3 -I research/scripts-a7/dupcheck.py --kw "zero advantage" --merged

--kw   : case-insensitive substring, OR-ed, matched against title + body
--path : file-path prefix, OR-ed, matched against the PR's changed files
--merged : also scan the merged dump (titles only; it covers 2026-07-16 onward)
A PR matches when it hits any --kw or any --path. Output is sorted by last-commit age (freshest first).
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import age_days, load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kw", action="append", default=[])
ap.add_argument("--path", action="append", default=[])
ap.add_argument("--merged", action="store_true")
ap.add_argument("--body-chars", type=int, default=0, help="print this many body chars per hit")
args = ap.parse_args()
if not args.kw and not args.path:
    ap.error("give at least one --kw or --path")

kws = [k.lower() for k in args.kw]
enriched = {}
try:
    with open(os.path.join(HERE, "out_open_enriched.json")) as f:
        enriched = json.load(f)
except FileNotFoundError:
    pass

hits = []
for p in load("prs_open.json"):
    text = (p["title"] + "\n" + (p.get("body") or "")).lower()
    kw_hit = [k for k in kws if k in text]
    path_hit = [f["path"] for f in p["files"] if any(f["path"].startswith(x) for x in args.path)]
    if not kw_hit and not path_hit:
        continue
    v = enriched.get(str(p["number"]), {})
    nodes = v.get("commits", {}).get("nodes") or []
    lc = age_days(nodes[0]["commit"]["committedDate"]) if nodes else None
    hits.append((lc if lc is not None else 9999, p, v, kw_hit, path_hit))

hits.sort(key=lambda h: h[0])
print(f"{len(hits)} open PRs match kw={args.kw} path={args.path}")
for lc, p, v, kw_hit, path_hit in hits:
    flags = []
    if p["isDraft"]:
        flags.append("DRAFT")
    if v.get("mergeable"):
        flags.append(v["mergeable"])
    if v.get("reviewDecision") and v["reviewDecision"] != "REVIEW_REQUIRED":
        flags.append(v["reviewDecision"])
    if lc != 9999 and lc > 60:
        flags.append("STALE")
    print(f"\n#{p['number']}  age {age_days(p['createdAt'])}d  last commit {'?' if lc == 9999 else str(lc) + 'd'}  {p['author']['login']} ({v.get('authorAssociation', '?')})  [{' '.join(flags)}]")
    print(f"   {p['title']}")
    if kw_hit:
        print(f"   kw: {kw_hit}")
    if path_hit:
        print(f"   files: {path_hit[:6]}")
    if args.body_chars:
        print("   " + (p.get("body") or "")[: args.body_chars].replace("\n", " "))

if args.merged:
    print("\n--- merged (2026-07-16 .. 2026-10-28), title match only ---")
    for m in load("prs_merged_recent.json"):
        if any(k in m["title"].lower() for k in kws):
            print(f"#{m['number']} merged {m['mergedAt'][:10]} {m['author']['login']}  {m['title']}")
