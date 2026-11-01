# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""`reward.custom_reward_function` for the `gsm8k-tool` bench preset (bench/README.md).

BENCH PLACEHOLDER: the P4 author's packages/p4/bench/gsm8k_tool_reward.py replaces this file in
bench/gsm8k_tool/ as soon as it lands (same contract). Score: verl's stock GSM8K rule (strict
``#### <number>``, what ``default_compute_score`` applies to ``openai/gsm8k``), so the CONTROL arm is
scored exactly as a stock run. Diagnostics (one float each; they reach ``reward_extra_info`` and so
the rollout dumps and the ``val-aux/openai/gsm8k/<key>/mean@n`` metrics) read the turn-level metadata
the tool agent loop puts in ``extra_info`` through ``tool_extra_fields``:

  turn_count        ``num_turns`` of the trajectory (user + assistant + tool turns + 1; 2 = no tool call)
  tool_calls        number of tool step rewards collected (``tool_rewards``, exported by every tree)
  tool_rewards_sum  their sum
  metadata_present  1.0 when ``turn_rewards``, ``generated_turn_spans`` and ``episode_done`` are all present
                    (the P4 export under rollout.multi_turn.turn_level_rewards=True; 0.0 on main / flag off)
  n_spans           number of generated-turn spans
  spans_ok          1.0 when the spans are ordered, non-empty, non-overlapping and index-aligned with turn_rewards
  turn_rewards_sum  sum of the per-turn rewards
  episode_done      1.0 when a tool ended the episode (ToolResponse.done)
"""

from typing import Any, Optional

from verl.utils.reward_score import gsm8k

TURN_KEYS = ("turn_rewards", "generated_turn_spans", "episode_done")


def _as_list(value: Any) -> Optional[list]:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _spans_ok(spans: Optional[list], rewards: Optional[list]) -> float:
    if spans is None:
        return 0.0
    try:
        pairs = [(int(s), int(e)) for s, e in spans]
    except (TypeError, ValueError):
        return 0.0
    if rewards is not None and len(rewards) != len(pairs):
        return 0.0
    prev_end = 0
    for start, end in pairs:
        if start < prev_end or end <= start:
            return 0.0
        prev_end = end
    return 1.0


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[dict] = None, **kwargs) -> dict:
    extra_info = dict(extra_info or {})
    score = float(gsm8k.compute_score(solution_str, ground_truth))  # method="strict": 1.0 / 0.0
    turn_rewards = _as_list(extra_info.get("turn_rewards"))
    spans = _as_list(extra_info.get("generated_turn_spans"))
    tool_rewards = _as_list(extra_info.get("tool_rewards")) or []
    num_turns = extra_info.get("num_turns")
    return {
        "score": score,
        "acc": score,
        "turn_count": float(num_turns) if num_turns is not None else -1.0,
        "tool_calls": float(len(tool_rewards)),
        "tool_rewards_sum": float(sum(float(x) for x in tool_rewards)),
        "metadata_present": 1.0 if all(k in extra_info for k in TURN_KEYS) else 0.0,
        "n_spans": float(len(spans)) if spans is not None else 0.0,
        "spans_ok": _spans_ok(spans, turn_rewards),
        "turn_rewards_sum": float(sum(float(x) for x in turn_rewards)) if turn_rewards else 0.0,
        "episode_done": 1.0 if bool(extra_info.get("episode_done", False)) else 0.0,
    }
