"""Step 4: merge dynamics from the merged dump + GraphQL enrichment; review/CI state of open PRs."""

import collections
import json
import re
import statistics
import sys

sys.path.insert(0, os.path.join(_REPO, "research/scripts-a7"))
from common import age_days, load, parse_dt, title_tags  # noqa: E402
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))  # the repository root

OUT = os.path.join(_REPO, "research/scripts-a7")
merged = load("prs_merged_recent.json")
with open(f"{OUT}/out_merged_enriched.json") as f:
    me = json.load(f)
with open(f"{OUT}/out_open_enriched.json") as f:
    oe = json.load(f)
open_prs = load("prs_open.json")

BOTS = {"gemini-code-assist", "github-actions", "copilot-pull-request-reviewer", "copilot", "codex", "coderabbitai", "dependabot"}


def q(xs, n=4):
    xs = sorted(xs)
    if len(xs) < 2:
        return xs
    return [round(v, 1) for v in statistics.quantiles(xs, n=n)]


print(f"# merged enriched: {len(me)}/{len(merged)}; open enriched: {len(oe)}/{len(open_prs)}")

# ---- time to merge by author association ----
ttm = collections.defaultdict(list)
for n, v in me.items():
    days = (parse_dt(v["mergedAt"]) - parse_dt(v["createdAt"])).total_seconds() / 86400
    ttm[v["authorAssociation"]].append(days)
    ttm["ALL"].append(days)
print("\n# time-to-merge (days) by authorAssociation: n, median, p25, p75, mean, share<=3d")
for k, xs in sorted(ttm.items(), key=lambda kv: -len(kv[1])):
    print(f"{k}: n={len(xs)} median={statistics.median(xs):.1f} q={q(xs)} mean={statistics.mean(xs):.1f} <=3d={sum(1 for x in xs if x<=3)/len(xs):.0%} >30d={sum(1 for x in xs if x>30)/len(xs):.0%}")

# ---- who merges ----
mb = collections.Counter(v["mergedBy"]["login"] if v.get("mergedBy") else "?" for v in me.values())
print("\n# mergedBy (all merged in window):", mb.most_common(20))
mb_month = collections.defaultdict(collections.Counter)
for v in me.values():
    mb_month[v["mergedAt"][:7]][v["mergedBy"]["login"] if v.get("mergedBy") else "?"] += 1
for mo in sorted(mb_month):
    print(mo, mb_month[mo].most_common(6))

# ---- reviewers ----
rv = collections.Counter()
rv_approve = collections.Counter()
for v in me.values():
    seen = set()
    for r in v["reviews"]["nodes"]:
        a = (r.get("author") or {}).get("login", "?")
        if a in BOTS or a.endswith("[bot]"):
            continue
        if a not in seen:
            rv[a] += 1
            seen.add(a)
        if r["state"] == "APPROVED":
            rv_approve[a] += 1
print("\n# human reviewers on merged PRs (distinct PRs reviewed):", rv.most_common(20))
print("# approvals:", rv_approve.most_common(12))
rv_open = collections.Counter()
for v in oe.values():
    seen = set()
    for r in v["reviews"]["nodes"]:
        a = (r.get("author") or {}).get("login", "?")
        if a in BOTS or a.endswith("[bot]"):
            continue
        if a not in seen:
            rv_open[a] += 1
            seen.add(a)
print("# human reviewers on OPEN PRs (distinct PRs):", rv_open.most_common(20))

# ---- author association of merged PRs and of open PRs ----
print("\n# merged authorAssociation:", collections.Counter(v["authorAssociation"] for v in me.values()).most_common())
print("# open authorAssociation:", collections.Counter(v["authorAssociation"] for v in oe.values()).most_common())

# ---- shape of merged PRs ----
lines = [v["additions"] + v["deletions"] for v in me.values()]
files = [v["changedFiles"] for v in me.values()]
tests = [any(f["path"].startswith("tests/") for f in v["files"]["nodes"]) for v in me.values()]
docs = [any(f["path"].startswith("docs/") for f in v["files"]["nodes"]) for v in me.values()]
print(f"\n# merged shape: lines median={statistics.median(lines)} q={q(lines)}; files median={statistics.median(files)} q={q(files)}; with tests/ {sum(tests)/len(tests):.0%}; with docs/ {sum(docs)/len(docs):.0%}")
reviews_n = [v["reviews"]["totalCount"] for v in me.values()]
comments_n = [v["comments"]["totalCount"] for v in me.values()]
commits_n = [v["commits"]["totalCount"] for v in me.values()]
print(f"# merged: reviews median={statistics.median(reviews_n)} q={q(reviews_n)}; comments median={statistics.median(comments_n)}; commits median={statistics.median(commits_n)} q={q(commits_n)}")

# by type
by_type = collections.defaultdict(list)
for n, v in me.items():
    mods, typ, br, valid = title_tags(v["title"])
    by_type[typ or "none"].append(v["additions"] + v["deletions"])
print("# merged lines by type:", {k: (len(x), statistics.median(x)) for k, x in by_type.items()})

# ---- merged perf PRs: anatomy ----
NUM = re.compile(r"\d+(?:\.\d+)?\s*(?:x|×|%|GB|GiB|MB|MiB|ms|s\b|tok/s|TFLOP)", re.I)
print("\n# merged [perf]-tagged or perf-worded PRs: anatomy")
for n, v in sorted(me.items(), key=lambda kv: kv[1]["mergedAt"], reverse=True):
    mods, typ, br, valid = title_tags(v["title"])
    t = v["title"].lower()
    if "perf" in mods or re.search(r"\b(speed|faster|throughput|overhead|memory|oom|fuse|overlap|cache|zero-copy|vectoriz|gc\b)", t):
        days = (parse_dt(v["mergedAt"]) - parse_dt(v["createdAt"])).total_seconds() / 86400
        tfiles = [f["path"] for f in v["files"]["nodes"] if f["path"].startswith("tests/")]
        nums = NUM.findall(v["body"] or "")
        human_rev = sorted({(r.get("author") or {}).get("login", "?") for r in v["reviews"]["nodes"]} - BOTS)
        print(f"#{n} merged {v['mergedAt'][:10]} after {days:.1f}d by {v['mergedBy']['login'] if v.get('mergedBy') else '?'} | {v['author']['login']} ({v['authorAssociation']}) | {v['additions']}+/{v['deletions']}- in {v['changedFiles']} files | tests: {len(tfiles)} | nums in body: {len(nums)} e.g. {nums[:5]} | reviewers: {human_rev} | {v['title'][:100]}")

# ---- open PR review/CI state ----
print("\n# open PRs: reviewDecision / CI / mergeable")
print("reviewDecision:", collections.Counter(v.get("reviewDecision") or "NONE" for v in oe.values()).most_common())
ci = collections.Counter()
for v in oe.values():
    nodes = v["commits"]["nodes"]
    st = (nodes[0]["commit"].get("statusCheckRollup") or {}).get("state") if nodes else None
    ci[st or "NONE"] += 1
print("statusCheckRollup:", ci.most_common())
print("mergeable:", collections.Counter(v.get("mergeable") for v in oe.values()).most_common())
# human review coverage by age bucket
bins = [(0, 7), (8, 30), (31, 60), (61, 120), (121, 240), (241, 10000)]
print("# open PRs by age bucket: n, with any human review, with APPROVED decision, with CI SUCCESS, mergeable CONFLICTING, last-commit>60d")
for lo, hi in bins:
    grp = [v for v in oe.values() if lo <= age_days(v["createdAt"]) <= hi]
    if not grp:
        continue
    hr = sum(1 for v in grp if any(((r.get("author") or {}).get("login", "?") not in BOTS) for r in v["reviews"]["nodes"]))
    ap = sum(1 for v in grp if v.get("reviewDecision") == "APPROVED")
    cs = sum(1 for v in grp if v["commits"]["nodes"] and (v["commits"]["nodes"][0]["commit"].get("statusCheckRollup") or {}).get("state") == "SUCCESS")
    cf = sum(1 for v in grp if v.get("mergeable") == "CONFLICTING")
    lc = sum(1 for v in grp if v["commits"]["nodes"] and age_days(v["commits"]["nodes"][0]["commit"]["committedDate"]) > 60)
    print(f"{lo}-{hi}d: n={len(grp)} human_review={hr} approved={ap} ci_success={cs} conflicting={cf} lastcommit>60d={lc}")

# last-commit age distribution overall
lc_ages = [age_days(v["commits"]["nodes"][0]["commit"]["committedDate"]) for v in oe.values() if v["commits"]["nodes"]]
print("# last-commit age quantiles:", q(lc_ages), "median", statistics.median(lc_ages), ">60d:", sum(1 for a in lc_ages if a > 60), ">120d:", sum(1 for a in lc_ages if a > 120))
