"""Enrich open + merged PRs via batched GraphQL (read-only). Writes out_open_enriched.json / out_merged_enriched.json.

Batch of 40 PRs per query with aliases. On 403/429 sleep 60 and retry once.
"""

import json
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(_REPO, "research/scripts-a7"))
from common import load  # noqa: E402
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))  # the repository root

OUT = os.path.join(_REPO, "research/scripts-a7")

OPEN_FIELDS = """
  number title isDraft reviewDecision mergeable authorAssociation additions deletions changedFiles
  createdAt updatedAt
  comments { totalCount }
  reviews(first: 40) { totalCount nodes { author { login } state submittedAt } }
  reviewThreads { totalCount }
  commits(last: 1) { totalCount nodes { commit { committedDate statusCheckRollup { state } } } }
  labels(first: 10) { nodes { name } }
"""

MERGED_FIELDS = """
  number title createdAt mergedAt authorAssociation additions deletions changedFiles
  mergedBy { login }
  author { login }
  comments { totalCount }
  reviews(first: 40) { totalCount nodes { author { login } state } }
  reviewThreads { totalCount }
  commits { totalCount }
  files(first: 100) { nodes { path additions deletions } }
  body
"""


def run_query(numbers, fields):
    parts = []
    for n in numbers:
        parts.append(f"p{n}: pullRequest(number: {n}) {{ {fields} }}")
    q = "query { repository(owner: \"verl-project\", name: \"verl\") { " + " ".join(parts) + " } }"
    for attempt in range(2):
        r = subprocess.run(["gh", "api", "graphql", "-f", f"query={q}"], capture_output=True, text=True)
        if r.returncode == 0:
            data = json.loads(r.stdout)
            if "errors" in data and not data.get("data"):
                print("graphql errors", data["errors"][:2], file=sys.stderr)
                return {}
            repo = data["data"]["repository"]
            return {int(k[1:]): v for k, v in repo.items() if v}
        err = (r.stderr or "") + (r.stdout or "")
        print("attempt", attempt, "failed:", err[:300], file=sys.stderr)
        if "403" in err or "429" in err or "rate limit" in err.lower():
            time.sleep(60)
        else:
            time.sleep(5)
    return {}


def enrich(prs, fields, out_name, batch=40):
    result = {}
    nums = [p["number"] for p in prs]
    for i in range(0, len(nums), batch):
        chunk = nums[i:i + batch]
        got = run_query(chunk, fields)
        result.update({str(k): v for k, v in got.items()})
        print(f"{out_name}: {len(result)}/{len(nums)}", file=sys.stderr)
        with open(f"{OUT}/{out_name}", "w") as f:
            json.dump(result, f)
        time.sleep(1.0)
    return result


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("open", "both"):
        enrich(load("prs_open.json"), OPEN_FIELDS, "out_open_enriched.json")
    if which in ("merged", "both"):
        enrich(load("prs_merged_recent.json"), MERGED_FIELDS, "out_merged_enriched.json", batch=25)
