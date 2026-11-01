"""Step 1: cluster the 774 open PRs by title module tag and by top-level path bucket."""

import collections
import json
import statistics
import sys

sys.path.insert(0, os.path.join(_REPO, "research/scripts-a7"))
from common import age_days, load, path_bucket, pr_short, title_tags  # noqa: E402
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))  # the repository root

prs = load("prs_open.json")
out = {}

# --- by module tag (a PR with [a, b] counts in both) ---
by_tag = collections.defaultdict(list)
by_type = collections.Counter()
invalid = []
for p in prs:
    mods, typ, breaking, valid = title_tags(p["title"])
    if not valid:
        invalid.append(p)
    for m in mods or ["(untagged)"]:
        by_tag[m].append(p)
    by_type[typ or "(none)"] += 1

print("# Open PRs by module tag (title prefix); a multi-module PR counts in each")
print("tag | n | drafts | median age d | >60d stale | newest 3")
rows = []
for tag, lst in sorted(by_tag.items(), key=lambda kv: -len(kv[1])):
    ages = [age_days(p["createdAt"]) for p in lst]
    stale = sum(1 for p in lst if age_days(p["updatedAt"]) > 60)
    newest = sorted(lst, key=lambda p: p["createdAt"], reverse=True)[:3]
    rows.append((tag, len(lst), sum(p["isDraft"] for p in lst), int(statistics.median(ages)), stale))
    print(f"{tag} | {len(lst)} | {sum(p['isDraft'] for p in lst)} | {int(statistics.median(ages))} | {stale} | "
          + "; ".join(f"#{p['number']}" for p in newest))
print()
print("type counts:", dict(by_type))
print("invalid/untagged titles:", len(invalid), [p["number"] for p in invalid][:40])
print()

# --- by path bucket (a PR counts in each bucket it touches) ---
by_bucket = collections.defaultdict(list)
for p in prs:
    seen = set()
    for f in p["files"]:
        b = path_bucket(f["path"])
        if b not in seen:
            seen.add(b)
            by_bucket[b].append(p)

print("# Open PRs by path bucket touched")
print("bucket | n | drafts | median age d | stale>60d(no update) | newest 3")
for b, lst in sorted(by_bucket.items(), key=lambda kv: -len(kv[1])):
    ages = [age_days(p["createdAt"]) for p in lst]
    stale = sum(1 for p in lst if age_days(p["updatedAt"]) > 60)
    newest = sorted(lst, key=lambda p: p["createdAt"], reverse=True)[:3]
    print(f"{b} | {len(lst)} | {sum(p['isDraft'] for p in lst)} | {int(statistics.median(ages))} | {stale} | "
          + "; ".join(f"#{p['number']}" for p in newest))
print()

# --- age histogram ---
ages = [age_days(p["createdAt"]) for p in prs]
upd = [age_days(p["updatedAt"]) for p in prs]
bins = [(0, 7), (8, 30), (31, 60), (61, 120), (121, 240), (241, 10000)]
print("# Age histogram (created)")
for lo, hi in bins:
    print(f"{lo}-{hi}d: {sum(1 for a in ages if lo <= a <= hi)}")
print("# Last-update histogram")
for lo, hi in bins:
    print(f"{lo}-{hi}d: {sum(1 for a in upd if lo <= a <= hi)}")
print("median age", statistics.median(ages), "median since update", statistics.median(upd))
print()

# --- size distribution ---
sizes = [sum(f["additions"] + f["deletions"] for f in p["files"]) for p in prs]
print("size quantiles (lines changed):", [int(statistics.quantiles(sizes, n=10)[i]) for i in range(9)])
nf = [len(p["files"]) for p in prs]
print("files/PR quantiles:", [int(statistics.quantiles(nf, n=10)[i]) for i in range(9)])

# dump per-cluster PR lists for the report
detail = {}
for tag, lst in by_tag.items():
    detail[tag] = [pr_short(p) for p in sorted(lst, key=lambda p: p["createdAt"], reverse=True)]
with open(os.path.join(_REPO, "research/scripts-a7/out_clusters_by_tag.json"), "w") as f:
    json.dump(detail, f, indent=1)
detail = {}
for b, lst in by_bucket.items():
    detail[b] = [pr_short(p) for p in sorted(lst, key=lambda p: p["createdAt"], reverse=True)]
with open(os.path.join(_REPO, "research/scripts-a7/out_clusters_by_path.json"), "w") as f:
    json.dump(detail, f, indent=1)
