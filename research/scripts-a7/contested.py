"""Step 2: contested files -- open PRs touching the hot-path files in the clone."""

import collections
import fnmatch
import glob
import os
import sys

sys.path.insert(0, os.path.join(_REPO, "research/scripts-a7"))
from common import age_days, load  # noqa: E402

CLONE = os.path.join(_REPO, "verl")
HOT = [
    "verl/trainer/ppo/ray_trainer.py",
    "verl/trainer/ppo/core_algos.py",
    "verl/trainer/ppo/reward.py",
    "verl/trainer/ppo/metric_utils.py",
    "verl/trainer/main_ppo.py",
    "verl/trainer/config/*.yaml",
    "verl/trainer/ppo/v1/*.py",
    "verl/protocol.py",
    "verl/utils/seqlen_balancing.py",
    "verl/utils/torch_functional.py",
    "verl/utils/kernel/*",
    "verl/workers/actor/dp_actor.py",
    "verl/workers/actor/megatron_actor.py",
    "verl/workers/critic/dp_critic.py",
    "verl/workers/fsdp_workers.py",
    "verl/workers/megatron_workers.py",
    "verl/workers/engine_workers.py",
    "verl/workers/rollout/vllm_rollout/*.py",
    "verl/workers/rollout/sglang_rollout/*.py",
    "verl/workers/rollout/replica.py",
    "verl/workers/rollout/base.py",
    "verl/workers/rollout/utils.py",
    "verl/experimental/agent_loop/agent_loop.py",
    "verl/experimental/agent_loop/*.py",
    "verl/experimental/fully_async_policy/*.py",
    "verl/checkpoint_engine/*.py",
    "verl/workers/engine/fsdp/*",
    "verl/workers/engine/megatron/*",
    "verl/workers/engine/base.py",
    "verl/workers/utils/losses.py",
    "verl/workers/utils/*.py",
    "verl/workers/config/*.py",
    "verl/workers/sharding_manager/*.py",
    "verl/utils/checkpoint/*.py",
    "verl/utils/dataset/rl_dataset.py",
    "verl/utils/model.py",
    "verl/utils/fsdp_utils.py",
    "verl/utils/vllm/*.py",
    "verl/models/transformers/monkey_patch.py",
    "verl/models/transformers/*.py",
    "verl/single_controller/**/*.py",
]

prs = load("prs_open.json")
touch = collections.defaultdict(list)
for p in prs:
    for f in p["files"]:
        touch[f["path"]].append(p)

# expand hot patterns against the clone
hot_files = []
for pat in HOT:
    matches = sorted(glob.glob(os.path.join(CLONE, pat), recursive=True))
    for m in matches:
        if os.path.isfile(m):
            hot_files.append(os.path.relpath(m, CLONE))
hot_files = list(dict.fromkeys(hot_files))

print("# Contested files (open PRs touching each hot-path file)")
print("file | open PRs | drafts | oldest | newest | PRs (number:age d, * = draft)")
rows = []
for hf in hot_files:
    lst = touch.get(hf, [])
    if not lst:
        rows.append((0, hf, lst))
        continue
    rows.append((len(lst), hf, lst))
rows.sort(key=lambda r: (-r[0], r[1]))
for n, hf, lst in rows:
    if n == 0:
        print(f"{hf} | 0 | - | - | - | (quiet)")
        continue
    ages = [age_days(p["createdAt"]) for p in lst]
    items = ", ".join(f"#{p['number']}:{age_days(p['createdAt'])}d{'*' if p['isDraft'] else ''}"
                      for p in sorted(lst, key=lambda p: p["createdAt"], reverse=True))
    print(f"{hf} | {n} | {sum(p['isDraft'] for p in lst)} | {max(ages)}d | {min(ages)}d | {items}")

print()
print("# Most-touched files overall (any path), top 40")
for path, lst in sorted(touch.items(), key=lambda kv: -len(kv[1]))[:40]:
    print(f"{len(lst):3d} {path}  drafts={sum(p['isDraft'] for p in lst)}  stale>60d={sum(1 for p in lst if age_days(p['updatedAt'])>60)}")

# detailed listing per hot file for the report
import json  # noqa: E402
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))  # the repository root
detail = {}
for n, hf, lst in rows:
    detail[hf] = [f"#{p['number']} ({age_days(p['createdAt'])}d{', draft' if p['isDraft'] else ''}, upd {age_days(p['updatedAt'])}d ago, {p['author']['login']}) {p['title'][:100]}"
                  for p in sorted(lst, key=lambda p: p["createdAt"], reverse=True)]
with open(os.path.join(_REPO, "research/scripts-a7/out_contested.json"), "w") as f:
    json.dump(detail, f, indent=1)
