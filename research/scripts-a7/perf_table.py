"""Join out_perf.json with the GraphQL enrichment and print a markdown table for the report (section 3)."""

import json
import sys

sys.path.insert(0, os.path.join(_REPO, "research/scripts-a7"))
from common import age_days  # noqa: E402
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))  # the repository root

OUT = os.path.join(_REPO, "research/scripts-a7")
perf = json.load(open(f"{OUT}/out_perf.json"))
oe = json.load(open(f"{OUT}/out_open_enriched.json"))
BOTS = {"gemini-code-assist", "github-actions", "copilot-pull-request-reviewer", "copilot", "codex", "coderabbitai", "chatgpt-codex-connector"}

# curated: perf tag OR strong title hit with numbers OR explicitly known perf PRs
KEEP = set()
for r in perf:
    if r["perf_tag"] or (r["title_hit"] and r["nums"]) or r["number"] in {
        5778, 6990, 8435, 8338, 7733, 7685, 7686, 7292, 6940, 6832, 4435, 2605, 7809, 8074, 7277, 7621, 6122, 6533, 5338,
        3852, 5651, 8057, 8096, 8097, 8378, 6110, 1295, 7131, 7415, 5976, 7108, 5398, 2629, 7524, 3548, 5426,
    }:
        KEEP.add(r["number"])

rows = [r for r in perf if r["number"] in KEEP]
rows.sort(key=lambda r: r["age"])
print("| PR | age / last commit | author (assoc) | size | claim (from body) | reviews (human) | CI | mergeable |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    v = oe.get(str(r["number"]), {})
    lc = age_days(v["commits"]["nodes"][0]["commit"]["committedDate"]) if v.get("commits", {}).get("nodes") else "?"
    humans = sorted({(x.get("author") or {}).get("login", "?") for x in v.get("reviews", {}).get("nodes", [])} - BOTS - {r["author"]})
    ci = (v["commits"]["nodes"][0]["commit"].get("statusCheckRollup") or {}).get("state") if v.get("commits", {}).get("nodes") else None
    nums = "; ".join(r["nums"][:4]) if r["nums"] else "none stated"
    dec = v.get("reviewDecision") or ""
    rv = f"{dec.replace('REVIEW_REQUIRED', 'none').replace('CHANGES_REQUESTED', 'changes req.').replace('APPROVED', 'APPROVED')}"
    if humans:
        rv += " by " + ",".join(humans)
    print(f"| #{r['number']} {'(draft) ' if r['draft'] else ''}{r['title'][:95]} | {r['age']} d / {lc} d | {r['author']} ({v.get('authorAssociation','?')}) | {r['lines']} l / {r['nfiles']} f, tests {'y' if r['has_tests'] else 'n'} | {nums} | {rv}; {v.get('comments',{}).get('totalCount','?')} comments | {ci or '-'} | {v.get('mergeable','?')} |")
print(f"\n{len(rows)} PRs")
