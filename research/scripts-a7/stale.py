"""Step 5: stale open PRs (last commit > 60 d) touching perf-relevant files."""

import json
import re
import sys

sys.path.insert(0, os.path.join(_REPO, "research/scripts-a7"))
from common import age_days, load, title_tags  # noqa: E402
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))  # the repository root

OUT = os.path.join(_REPO, "research/scripts-a7")
with open(f"{OUT}/out_open_enriched.json") as f:
    oe = json.load(f)
with open(f"{OUT}/out_perf.json") as f:
    perf = {r["number"]: r for r in json.load(f)}
prs = load("prs_open.json")
BOTS = {"gemini-code-assist", "github-actions", "copilot-pull-request-reviewer", "copilot", "codex", "coderabbitai"}

PERF_PATHS = re.compile(
    r"^(verl/trainer/ppo/(ray_trainer|core_algos)\.py|verl/trainer/ppo/v1/|verl/protocol\.py|verl/utils/seqlen_balancing\.py|"
    r"verl/utils/torch_functional\.py|verl/utils/kernel/|verl/workers/actor/|verl/workers/fsdp_workers\.py|verl/workers/megatron_workers\.py|"
    r"verl/workers/engine/|verl/workers/rollout/|verl/experimental/agent_loop/|verl/experimental/fully_async_policy/|verl/checkpoint_engine/|"
    r"verl/utils/fsdp_utils\.py|verl/utils/megatron_utils\.py|verl/utils/megatron/|verl/models/|verl/workers/sharding_manager/|verl/workers/utils/|"
    r"verl/utils/activation_offload\.py|verl/utils/memory_utils\.py|verl/utils/fp8_utils\.py|verl/utils/qat/|verl/utils/vllm/)"
)

rows = []
for p in prs:
    v = oe.get(str(p["number"]))
    if not v or not v["commits"]["nodes"]:
        continue
    lc = age_days(v["commits"]["nodes"][0]["commit"]["committedDate"])
    if lc <= 60:
        continue
    touched = [f["path"] for f in p["files"] if PERF_PATHS.match(f["path"])]
    if not touched:
        continue
    mods, typ, br, valid = title_tags(p["title"])
    is_perf = p["number"] in perf or "perf" in mods
    human_rev = sorted({(r.get("author") or {}).get("login", "?") for r in v["reviews"]["nodes"]} - BOTS)
    rows.append({
        "number": p["number"], "title": p["title"], "age": age_days(p["createdAt"]), "last_commit": lc,
        "draft": p["isDraft"], "author": p["author"]["login"], "assoc": v["authorAssociation"],
        "perf_claim": is_perf, "lines": v["additions"] + v["deletions"], "files": v["changedFiles"],
        "mergeable": v.get("mergeable"), "decision": v.get("reviewDecision"), "comments": v["comments"]["totalCount"],
        "human_reviewers": human_rev, "touched": touched[:6],
        "nums": perf[p["number"]]["nums"][:5] if p["number"] in perf else [],
    })

rows.sort(key=lambda r: (not r["perf_claim"], r["last_commit"]))
print(f"# {len(rows)} open PRs with last commit > 60 d touching perf-relevant paths; {sum(r['perf_claim'] for r in rows)} carry a perf/memory claim")
print("\n## with perf/memory claim (sorted by last-commit age)")
for r in rows:
    if r["perf_claim"]:
        print(f"#{r['number']} | {r['age']}d old, last commit {r['last_commit']}d ago{', DRAFT' if r['draft'] else ''} | {r['author']} ({r['assoc']}) | {r['lines']} lines/{r['files']} files | {r['mergeable']} | decision={r['decision']} | comments={r['comments']} | reviewers={r['human_reviewers']} | nums={r['nums']} | {r['title'][:100]}")
        print("     touches:", ", ".join(r["touched"][:4]))
print("\n## without explicit perf claim but in perf-relevant paths (number | age | last commit | title)")
for r in rows:
    if not r["perf_claim"]:
        print(f"#{r['number']} | {r['age']}d | lc {r['last_commit']}d{' DRAFT' if r['draft'] else ''} | {r['mergeable']} | {r['title'][:90]}")

with open(f"{OUT}/out_stale.json", "w") as f:
    json.dump(rows, f, indent=1)
