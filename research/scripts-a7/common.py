"""Shared helpers for the open-PR territory study (agent slice 7)."""

import json
import re
from datetime import datetime, timezone
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))  # the repository root

GH = os.path.join(_REPO, "gh")
NOW = datetime(2026, 10, 29, 2, 0, tzinfo=timezone.utc)  # dump pulled 2026-10-28 ~21:08 local (UTC-5)

MODULE_TAGS = {
    "fsdp", "megatron", "veomni", "sglang", "vllm", "trtllm", "rollout", "trainer",
    "tests", "training_utils", "recipe", "hardware", "deployment",
    "ray", "worker", "single_controller", "misc", "docker", "ci",
    "perf", "model", "algo", "env", "tool", "ckpt", "doc", "data", "cfg", "reward",
    "fully_async", "one_step_off",
}
TYPE_TAGS = {"feat", "fix", "refactor", "chore", "test"}

_title_re = re.compile(r"^(?:\[\d/[\dNn]\]\s*)?(?:\[BREAKING\]\s*)?\[([a-z_,\s]+)\]\s*(?:(feat|fix|refactor|chore|test):)?", re.I)


def load(name):
    with open(f"{GH}/{name}") as f:
        return json.load(f)


def parse_dt(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def age_days(s):
    return (NOW - parse_dt(s)).days


def title_tags(title):
    """Return (modules:list, type:str|None, breaking:bool, valid:bool)."""
    t = title.strip()
    breaking = bool(re.match(r"^(?:\[\d/[\dNn]\]\s*)?\[BREAKING\]", t, re.I))
    m = _title_re.match(t)
    if not m:
        return [], None, breaking, False
    mods = [x for x in re.findall(r"[a-z_]+", m.group(1).lower())]
    typ = m.group(2).lower() if m.group(2) else None
    valid = all(x in MODULE_TAGS for x in mods) and typ is not None
    return mods, typ, breaking, valid


PATH_BUCKETS = [
    ("verl/trainer/ppo/v1", "verl/trainer/ppo/v1"),
    ("verl/trainer/ppo", "verl/trainer/ppo"),
    ("verl/trainer", "verl/trainer (other)"),
    ("verl/workers/rollout/vllm_rollout", "verl/workers/rollout/vllm_rollout"),
    ("verl/workers/rollout/sglang_rollout", "verl/workers/rollout/sglang_rollout"),
    ("verl/workers/rollout", "verl/workers/rollout (other)"),
    ("verl/workers/actor", "verl/workers/actor"),
    ("verl/workers/critic", "verl/workers/critic"),
    ("verl/workers/engine/fsdp", "verl/workers/engine/fsdp"),
    ("verl/workers/engine/megatron", "verl/workers/engine/megatron"),
    ("verl/workers/engine/veomni", "verl/workers/engine/veomni"),
    ("verl/workers/engine", "verl/workers/engine (other)"),
    ("verl/workers/fsdp_workers.py", "verl/workers/fsdp_workers.py"),
    ("verl/workers/megatron_workers.py", "verl/workers/megatron_workers.py"),
    ("verl/workers/config", "verl/workers/config"),
    ("verl/workers/utils", "verl/workers/utils"),
    ("verl/workers/sharding_manager", "verl/workers/sharding_manager"),
    ("verl/workers/reward_manager", "verl/workers/reward_manager"),
    ("verl/workers", "verl/workers (other)"),
    ("verl/checkpoint_engine", "verl/checkpoint_engine"),
    ("verl/experimental/agent_loop", "verl/experimental/agent_loop"),
    ("verl/experimental/fully_async_policy", "verl/experimental/fully_async_policy"),
    ("verl/experimental", "verl/experimental (other)"),
    ("verl/utils/kernel", "verl/utils/kernel"),
    ("verl/utils/megatron", "verl/utils/megatron"),
    ("verl/utils/dataset", "verl/utils/dataset"),
    ("verl/utils/checkpoint", "verl/utils/checkpoint"),
    ("verl/utils/reward_score", "verl/utils/reward_score"),
    ("verl/utils/seqlen_balancing.py", "verl/utils/seqlen_balancing.py"),
    ("verl/utils/torch_functional.py", "verl/utils/torch_functional.py"),
    ("verl/utils", "verl/utils (other)"),
    ("verl/protocol.py", "verl/protocol.py"),
    ("verl/single_controller", "verl/single_controller"),
    ("verl/models", "verl/models"),
    ("verl/tools", "verl/tools"),
    ("verl/interactions", "verl/interactions"),
    ("verl/third_party", "verl/third_party"),
    ("verl/", "verl (other)"),
    ("tests/", "tests"),
    ("docs/", "docs"),
    ("recipe", "recipe"),
    ("examples/", "examples"),
    (".github/", ".github"),
    ("docker/", "docker"),
    ("scripts/", "scripts"),
]


def path_bucket(path):
    for prefix, name in PATH_BUCKETS:
        if path.startswith(prefix):
            return name
    return "other"


def pr_short(p):
    return f"#{p['number']} ({age_days(p['createdAt'])}d{', draft' if p['isDraft'] else ''}) {p['title'][:90]}"
