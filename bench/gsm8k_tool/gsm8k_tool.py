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
"""GSM8K scoring tool for the `gsm8k-tool` bench preset (bench/README.md).

verl removed its reference ``verl/tools/gsm8k_tool.py`` in 5a506cc5 (#6302: "reference patterns now
live in docs/git history"). This is that class (`git show 5a506cc5^:verl/tools/gsm8k_tool.py`) with the
imports updated to the flat ``verl.tools`` layout and two bench knobs read from the tool config dict:

  step_reward: "verl" (default) keeps the reference step reward: 0.0 when the submitted answer scores
      higher than the previous submission, else -0.05 (penalty for a non-improving submission);
      "correctness" returns the submitted answer's score (1.0 / 0.0) as the step reward of EVERY call
      (review round 3 found this rewards resubmitting a correct answer: 923 / 1085 of 1319 validation
      episodes repeated an identical answer); "first_correct" pays +1.0 only for the first correct
      submission of an episode (tool instance) and 0.0 for every other call, correct or not.
  done_when_correct: when true and ``ToolResponse`` has a ``done`` field (the P4 trees), a correct
      submission ends the episode through ``ToolResponse.done``: the loop appends the tool response as
      the final observation and stops generating, so the outcome scorer must read the last submitted
      answer from the text (packages/p4/bench/gsm8k_tool_reward.py v2) instead of a trailing ``#### ``.

The dataset (examples/data_preprocess/gsm8k_tool_agent_loop.py) passes the ground truth through
``extra_info.tools_kwargs.calc_gsm8k_reward.create_kwargs``, which ``create`` reads. Shipped into the
source tree as ``bench_extras/gsm8k_tool.py`` (the driver's cwd is on every Ray worker's sys.path).
"""

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.reward_score import gsm8k
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

HAS_DONE = "done" in getattr(ToolResponse, "model_fields", {})  # P4a adds ToolResponse.done
STEP_REWARD_MODES = ("verl", "correctness", "first_correct")


class Gsm8kTool(BaseTool):
    """verl's demo tool for calculating the reward of gsm8k (see module docstring)."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}
        config = config or {}
        self.step_reward = str(config.get("step_reward", "verl"))
        if self.step_reward not in STEP_REWARD_MODES:
            raise ValueError(f"step_reward must be one of {STEP_REWARD_MODES}, got {self.step_reward!r}")
        self.done_when_correct = bool(config.get("done_when_correct", False))
        if self.done_when_correct and not HAS_DONE:
            logger.warning("done_when_correct requested but ToolResponse has no `done` field on this tree; ignored")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        if ground_truth is None:
            ground_truth = kwargs.get("create_kwargs", {}).get("ground_truth", None)
        self._instance_dict[instance_id] = {
            "response": "", "ground_truth": ground_truth, "reward": 0.0, "calls": 0, "solved": False
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        answer = parameters.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)
        state = self._instance_dict[instance_id]
        state["response"] = answer if answer.startswith("#### ") else "#### " + answer
        reward = await self.calc_reward(instance_id)
        if self.step_reward == "verl":
            # penalty for non improved answer submission (the reference implementation)
            tool_reward = 0.0 if reward > state["reward"] else -0.05
        elif self.step_reward == "first_correct":
            tool_reward = 1.0 if (reward >= 1.0 and not state["solved"]) else 0.0
        else:
            tool_reward = float(reward)
        state["reward"] = reward
        state["solved"] = state["solved"] or reward >= 1.0
        state["calls"] += 1
        text = f"Current parsed {answer=} {reward=}"
        if HAS_DONE:
            response = ToolResponse(text=text, done=bool(self.done_when_correct and reward >= 1.0))
        else:
            response = ToolResponse(text=text)
        return response, tool_reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        state = self._instance_dict[instance_id]
        return gsm8k.compute_score(state["response"], state["ground_truth"], method="flexible", format_score=0.0, score=1.0)

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
