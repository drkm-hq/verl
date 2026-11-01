# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""R2c: the real Qwen2.5-0.5B tokenizer, the loop's own Continuous Token merges and the real Hermes parser,
swept over every ``response_length`` from 1 to past the full episode: the exported spans always satisfy the strict
invariants, equal the runs of generated tokens, decode to the generated text (or its truncated prefix), and the
dense row lands on generated tokens only. Skipped when ``~/models/Qwen/Qwen2.5-0.5B`` is absent (HF offline)."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import HermesToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils import hf_tokenizer
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder
from verl.workers.rollout.replica import TokenOutput

MODEL_PATH = Path("~/models/Qwen/Qwen2.5-0.5B").expanduser()
pytestmark = pytest.mark.skipif(not MODEL_PATH.is_dir(), reason=f"{MODEL_PATH} is not available")

SCHEMA = OpenAIFunctionToolSchema.model_validate(
    {"type": "function", "function": {"name": "step", "description": "Advance one cycle.", "parameters": {
        "type": "object", "properties": {"k": {"type": "integer", "description": "cycle"}}, "required": ["k"]}}}
)
TURNS = [f'<tool_call>\n{{"name": "step", "arguments": {{"k": {k}}}}}\n</tool_call><|im_end|>' for k in (1, 2)]
TURNS.append("The part is placed.<|im_end|>")


class StepTool:
    def __init__(self, rewards, done_on_last):
        self.name, self.tool_schema, self.rewards, self.done_on_last, self.calls = "step", SCHEMA, rewards, done_on_last, 0

    async def create(self, create_kwargs=None):
        return "instance", ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        reward, self.calls = self.rewards[self.calls], self.calls + 1
        return ToolResponse(text=f"observation after cycle {parameters['k']}", done=self.done_on_last and self.calls == 2), reward, {}

    async def release(self, instance_id):
        pass


class ScriptedServer:
    def __init__(self, tokenizer):
        self.tokenizer, self.calls = tokenizer, 0

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        text, self.calls = TURNS[self.calls], self.calls + 1
        return TokenOutput(token_ids=self.tokenizer.encode(text, add_special_tokens=False), log_probs=None)


def make_loop(tokenizer, tool, response_length):
    loop = object.__new__(ToolAgentLoop)
    loop.rollout_config = SimpleNamespace(full_determinism=False, prompt_length=512, response_length=response_length)
    loop.tokenizer, loop.processor = tokenizer, None
    loop.continuous_token_builder = create_continuous_token_builder(tokenizer, hf_model_type="qwen2")
    loop.loop = asyncio.get_running_loop()
    loop.max_user_turns, loop.max_assistant_turns, loop.max_parallel_calls = None, 8, 1
    loop.max_tool_response_length, loop.tool_response_truncate_side = 256, "middle"
    loop.turn_level_rewards, loop.prompt_length, loop.response_length = True, 512, response_length
    loop.tools, loop.tool_schemas = {"step": tool}, [SCHEMA.model_dump(exclude_unset=True, exclude_none=True)]
    loop.tool_parser, loop.tool_parser_name = HermesToolParser(tokenizer), "hermes"
    loop.server_manager = ScriptedServer(tokenizer)

    async def process_multi_modal_info(messages):
        return {}

    loop.process_multi_modal_info = process_multi_modal_info
    loop._get_mm_processor_kwargs = lambda audios: None
    return loop


def mask_runs(mask):
    runs, start = [], None
    for i, m in enumerate(list(mask) + [0]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i))
            start = None
    return runs


@pytest.mark.parametrize("done_on_last", [False, True])
async def test_every_response_budget_exports_strictly_valid_spans_through_the_real_merges(done_on_last):
    tokenizer = hf_tokenizer(str(MODEL_PATH))
    full = await make_loop(tokenizer, StepTool([0.25, 1.0], done_on_last), 4096).run({}, raw_prompt=[{"role": "user", "content": "Place the part."}])
    full_len = len(full.response_ids)
    assert full.extra_fields["generated_turn_spans"] == mask_runs(full.response_mask) and full_len > 40
    boundaries = {"mid_generation": 0, "at_merge": 0, "complete": 0}
    for response_length in range(1, full_len + 3):
        tool = StepTool([0.25, 1.0], done_on_last)
        out = await make_loop(tokenizer, tool, response_length).run({}, raw_prompt=[{"role": "user", "content": "Place the part."}])
        spans, rewards, mask, n = out.extra_fields["generated_turn_spans"], out.extra_fields["turn_rewards"], out.response_mask, len(out.response_ids)
        assert 1 <= n <= response_length and len(mask) == n
        previous_end = 0
        for start, end in spans:
            assert type(start) is int and type(end) is int and previous_end <= start < end <= n, (response_length, spans, n)
            previous_end = end
        assert spans == mask_runs(mask) and len(spans) == len(rewards) == len(out.turn_rewards)
        for (start, end), text in zip(spans, TURNS, strict=False):
            assert out.response_ids[start:end] == tokenizer.encode(text, add_special_tokens=False)[: end - start]
        assert out.response_ids[: full_len][:n] == full.response_ids[:n], "a truncated trajectory is a prefix of the full one"
        out.reward_score = 2.0
        rm_scores = out.as_dict()["rm_scores"]
        assert torch.all(rm_scores[torch.tensor(mask) == 0] == 0) and rm_scores.sum().item() == pytest.approx(sum(rewards) + 2.0)
        cut = n == response_length and n < full_len  # the budget truncated the trajectory
        if cut and tool.calls < len(spans):  # cut mid-generation, before that turn's tool ran
            boundaries["mid_generation"] += 1
            assert rewards[-1] == 0.0 and spans[-1][1] == n
        elif out.extra_fields["episode_done"] and mask[-1] == 1:  # done, but the terminal merge hit the budget
            boundaries["at_merge"] += 1
            assert rewards[-1] == 1.0 and n == spans[-1][1]
        elif n == full_len:
            boundaries["complete"] += 1
    assert boundaries["mid_generation"] and boundaries["complete"] and (boundaries["at_merge"] > 0) is done_on_last, boundaries
