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
"""Outcome reward for the P4 GPU experiment (GSM8K through the tool agent loop), v2 (round 3).

Load it with ``reward.custom_reward_function.path=<this file> reward.custom_reward_function.name=compute_score``.
One scorer for every arm: the outcome is the correctness of the LAST answer the policy submitted through
``calc_gsm8k_reward`` (the last ``<tool_call>`` JSON payload in the decoded response), judged with the same rule the
tool itself uses (verl's flexible GSM8K scorer on ``#### <answer>``). An episode ended by ``ToolResponse.done``
(``done_when_correct``) stops right after a correct submission, so it is scored by that submission like any other.
A trajectory with no submission falls back to verl's strict ``#### <answer>`` rule on the response text
(``fallback_to_final_text``, default True; ``reward.custom_reward_function.reward_kwargs.fallback_to_final_text=false``
scores it 0). Diagnostics returned next to the score (logged as ``val-aux/openai/gsm8k/<key>/mean@N``):

- ``submissions``, ``distinct_submissions``, ``repeated_submission`` (1.0 when an identical answer was submitted
  twice: the reviewer's ``analyze_runs.py`` rule, per arm), ``scored_by_submission`` (1.0 unless the fallback scored)
- ``final_text_correct``: the strict rule on the response text, for reference
- ``turn_count``, ``turn_rewards_sum``, ``episode_done``, ``spans_ok``, ``metadata_present``, ``num_turns``: the
  p4a metadata as delivered to the callback (0 / absent-as-0 with the flag off)
"""

import json
import re

from verl.utils.reward_score import gsm8k

TURN_KEYS = ("generated_turn_spans", "turn_rewards", "episode_done")
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def submitted_answers(text: str) -> list[str]:
    """Every ``answer`` argument of a parsable ``<tool_call>`` payload, in order (malformed JSON is skipped)."""
    answers = []
    for raw in TOOL_CALL_RE.findall(text or ""):
        try:
            call = json.loads(raw)
        except ValueError:
            continue
        arguments = call.get("arguments") if isinstance(call, dict) else None
        if isinstance(arguments, dict) and "answer" in arguments:
            answers.append(str(arguments["answer"]))
    return answers


def submission_correct(answer: str, ground_truth: str) -> float:
    """The tool's own rule: ``#### <answer>`` through verl's flexible GSM8K scorer."""
    response = answer if answer.startswith("#### ") else "#### " + answer
    return float(gsm8k.compute_score(response, ground_truth, method="flexible", format_score=0.0, score=1.0))


def _spans_ok(spans) -> bool:
    previous_end = 0
    try:
        for start, end in spans:
            start, end = int(start), int(end)
            if start < previous_end or start >= end:
                return False
            previous_end = end
    except (TypeError, ValueError):
        return False
    return True


def compute_score(data_source, solution_str, ground_truth, extra_info=None, fallback_to_final_text=True, **kwargs):
    answers = submitted_answers(solution_str)
    final_text_correct = float(gsm8k.compute_score(solution_str, ground_truth))
    if answers:
        score, scored_by_submission = submission_correct(answers[-1], ground_truth), 1.0
    else:
        score, scored_by_submission = (final_text_correct if fallback_to_final_text else 0.0), 0.0

    extra_info = extra_info if isinstance(extra_info, dict) else {}
    present = all(key in extra_info for key in TURN_KEYS)
    spans = list(extra_info.get("generated_turn_spans") or [])
    turn_rewards = [float(reward) for reward in (extra_info.get("turn_rewards") or [])]
    return {
        "score": score,
        "acc": score,
        "submissions": float(len(answers)),
        "distinct_submissions": float(len(set(answers))),
        "repeated_submission": float(len(answers) > len(set(answers))),
        "scored_by_submission": scored_by_submission,
        "final_text_correct": final_text_correct,
        "turn_count": float(len(spans)),
        "turn_rewards_sum": float(sum(turn_rewards)),
        "episode_done": float(bool(extra_info.get("episode_done", False))),
        "spans_ok": float(present and bool(spans) and len(spans) == len(turn_rewards) and _spans_ok(spans)),
        "metadata_present": float(present),
        "num_turns": float(extra_info.get("num_turns") or 0),
    }
