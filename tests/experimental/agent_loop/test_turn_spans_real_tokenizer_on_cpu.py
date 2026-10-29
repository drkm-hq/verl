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
"""Turn spans through the real Continuous Token merge path with a real tokenizer.

``ToolAgentLoop`` runs its own ``ct_build_initial_tokens`` / ``ct_merge_assistant_token`` /
``ct_merge_context_msg`` on the ``QwenContinuousTokenBuilder`` over ``Qwen/Qwen2.5-0.5B`` and the real
Hermes tool parser; only the LLM server and the tool are scripted. The junction under test: generation
stops at ``<|im_end|>`` (vLLM's stop token), so the context merge inserts the missing newline as a mask-0
boundary token before the tool response, and the next assistant span must start after the generation prompt.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import HermesToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils import hf_tokenizer
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder
from verl.workers.rollout.replica import TokenOutput

MODEL_PATH = Path("~/models/Qwen/Qwen2.5-0.5B").expanduser()
pytestmark = pytest.mark.skipif(not MODEL_PATH.is_dir(), reason=f"{MODEL_PATH} is not available")

TOOL_SCHEMA = OpenAIFunctionToolSchema.model_validate(
    {
        "type": "function",
        "function": {
            "name": "step",
            "description": "Advance the device by one cycle.",
            "parameters": {
                "type": "object",
                "properties": {"k": {"type": "integer", "description": "cycle index"}},
                "required": ["k"],
            },
        },
    }
)


def tool_call_turn(k: int) -> str:
    return f'<tool_call>\n{{"name": "step", "arguments": {{"k": {k}}}}}\n</tool_call><|im_end|>'


FINAL_TURN = "The part is placed.<|im_end|>"


class StepTool:
    def __init__(self, rewards: list[float], done_on_last: bool):
        self.name, self.tool_schema = "step", TOOL_SCHEMA
        self.rewards, self.done_on_last, self.calls = rewards, done_on_last, 0

    async def create(self, create_kwargs=None):
        return "instance", ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        reward = self.rewards[self.calls]
        self.calls += 1
        done = self.done_on_last and self.calls == len(self.rewards)
        return ToolResponse(text=f"observation after cycle {parameters['k']}", done=done), reward, {}

    async def release(self, instance_id):
        pass


class ScriptedServer:
    """Returns the real token ids of the scripted assistant turns, each ending with ``<|im_end|>``."""

    def __init__(self, tokenizer, turns: list[str]):
        self.tokenizer, self.turns, self.calls = tokenizer, turns, 0

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        text = self.turns[self.calls]
        self.calls += 1
        return TokenOutput(token_ids=self.tokenizer.encode(text, add_special_tokens=False), log_probs=None)


def make_loop(tokenizer, tool: StepTool, turns: list[str], *, response_length: int = 256) -> ToolAgentLoop:
    loop = object.__new__(ToolAgentLoop)
    loop.rollout_config = SimpleNamespace(full_determinism=False, prompt_length=512, response_length=response_length)
    loop.tokenizer, loop.processor = tokenizer, None
    loop.continuous_token_builder = create_continuous_token_builder(tokenizer, hf_model_type="qwen2")
    loop.loop = asyncio.get_running_loop()
    loop.max_user_turns, loop.max_assistant_turns, loop.max_parallel_calls = None, 8, 1
    loop.max_tool_response_length, loop.tool_response_truncate_side = 256, "middle"
    loop.turn_level_rewards = True
    loop.prompt_length, loop.response_length = 512, response_length
    loop.tools = {tool.name: tool}
    loop.tool_schemas = [TOOL_SCHEMA.model_dump(exclude_unset=True, exclude_none=True)]
    loop.tool_parser, loop.tool_parser_name = HermesToolParser(tokenizer), "hermes"
    loop.server_manager = ScriptedServer(tokenizer, turns)

    async def process_multi_modal_info(messages):
        return {}

    loop.process_multi_modal_info = process_multi_modal_info
    loop._get_mm_processor_kwargs = lambda audios: None
    return loop


def _mask_runs(mask: list[int]) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, m in enumerate(mask + [0]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i))
            start = None
    return runs


async def _run(tokenizer, tool, turns, **kwargs):
    loop = make_loop(tokenizer, tool, turns, **kwargs)
    return await loop.run({}, raw_prompt=[{"role": "user", "content": "Place the part."}])


@pytest.mark.asyncio
async def test_spans_follow_the_real_qwen_junction_edits():
    tokenizer = hf_tokenizer(str(MODEL_PATH))
    turns = [tool_call_turn(1), tool_call_turn(2), FINAL_TURN]
    output = await _run(tokenizer, StepTool([0.25, 1.0], done_on_last=False), turns)

    mask, spans = output.response_mask, output.extra_fields["generated_turn_spans"]
    assert spans == _mask_runs(mask) and len(spans) == 3, "spans are exactly the runs of generated tokens"
    for (start, end), text in zip(spans, turns, strict=True):
        assert output.response_ids[start:end] == tokenizer.encode(text, add_special_tokens=False)
    newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    for start, end in spans[:2]:  # the junction: <|im_end|> ends the span, the inserted newline is mask 0
        assert output.response_ids[end - 1] == im_end and output.response_ids[end] == newline and mask[end] == 0
    between = tokenizer.decode(output.response_ids[spans[0][1] : spans[1][0]])
    assert between.startswith("\n<|im_start|>user") and "observation after cycle 1" in between
    assert between.endswith("<|im_start|>assistant\n"), "the next span starts right after the generation prompt"
    assert output.extra_fields["turn_rewards"] == [0.25, 1.0, 0.0]
    assert output.extra_fields["episode_done"] is False

    output.reward_score = 1.0
    rm_scores = output.as_dict()["rm_scores"]
    expected = {spans[0][1] - 1: 0.25, spans[1][1] - 1: 1.0, spans[2][1] - 1: 1.0}  # each turn's <|im_end|>
    assert {i: v for i, v in enumerate(rm_scores.tolist()) if v} == expected


@pytest.mark.asyncio
async def test_done_keeps_the_terminal_observation_unless_the_context_merge_hits_the_budget():
    tokenizer = hf_tokenizer(str(MODEL_PATH))
    turns = [tool_call_turn(1), tool_call_turn(2)]
    output = await _run(tokenizer, StepTool([0.25, 1.0], done_on_last=True), turns)
    spans = output.extra_fields["generated_turn_spans"]
    assert output.extra_fields["episode_done"] is True and len(spans) == 2
    tail = tokenizer.decode(output.response_ids[spans[1][1] :])
    assert "observation after cycle 2" in tail, "done: the tool response stays as the final observation"
    assert set(output.response_mask[spans[1][1] :]) == {0}

    # the documented exception: the final context merge would reach response_length, so it is not applied
    cut = await _run(tokenizer, StepTool([0.25, 1.0], done_on_last=True), turns, response_length=spans[1][1] + 1)
    assert cut.extra_fields["generated_turn_spans"] == spans
    assert len(cut.response_ids) == spans[1][1], "no terminal observation in the trajectory"
    assert cut.extra_fields["turn_rewards"] == [0.25, 1.0] and cut.extra_fields["episode_done"] is True
