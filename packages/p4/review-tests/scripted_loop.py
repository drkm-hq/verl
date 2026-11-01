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
"""Review harness: the real ``ToolAgentLoop.run`` with scripted server / parser / CT merges.

Unlike the branch's own test harness, the context merge can *remove* trailing prefix tokens
(GLM-style ``removed_prefix_token_count``) and *insert* boundary tokens (Gemma-style
``inserted_token_ids``), so the span-clipping branch of ``_align_generated_turn_spans`` runs.
Works on main as well (main ignores the extra attributes), which the flag-off identity test uses.
"""

from types import SimpleNamespace

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse
from verl.utils.tokenizer.continuous_token import MergeResult
from verl.workers.rollout.replica import TokenOutput

GEN = 3  # generated tokens per assistant turn
OBS = 2  # tokens per tool observation


class StepTool:
    """k-th call returns reward ``rewards[k]``; ``done`` on the call index in ``done_on`` (0-based)."""

    def __init__(self, rewards, done_on=None, name="step"):
        self.name = name
        self.tool_schema = SimpleNamespace(
            model_dump=lambda **kw: {"type": "function", "function": {"name": name, "parameters": {}}}
        )
        self.rewards = list(rewards)
        self.done_on = set(done_on or [])
        self.calls = 0

    async def create(self, create_kwargs=None):
        return "inst", ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        k = self.calls
        self.calls += 1
        return ToolResponse(text=f"obs {k}", done=k in self.done_on), self.rewards[k], {}

    async def release(self, instance_id):
        pass


class Server:
    def __init__(self):
        self.calls = 0

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        self.calls += 1
        return TokenOutput(token_ids=[1000 * self.calls + i for i in range(GEN)], log_probs=None)


class Parser:
    """Turns 1..tool_call_turns emit ``calls_per_turn`` calls to ``tool_names``; later turns none."""

    stop_token_ids: list[int] = []

    def __init__(self, tool_call_turns, calls_per_turn=1, tool_names=("step",)):
        self.tool_call_turns, self.calls_per_turn, self.tool_names = tool_call_turns, calls_per_turn, tool_names
        self.turn = 0

    async def extract_tool_calls(self, response_ids, tools):
        self.turn += 1
        if self.turn <= self.tool_call_turns:
            names = [self.tool_names[i % len(self.tool_names)] for i in range(self.calls_per_turn)]
            return "", [FunctionCall(name=n, arguments=f'{{"k": {self.turn}}}') for n in names]
        return "final", []


def make_loop(
    tools,
    *,
    tool_call_turns,
    turn_level_rewards=True,
    max_assistant_turns=8,
    max_parallel_calls=1,
    response_length=64,
    removed_on_context=0,
    inserted_on_context=0,
    calls_per_turn=1,
):
    loop = object.__new__(ToolAgentLoop)
    loop.rollout_config = SimpleNamespace(full_determinism=False)
    loop.max_user_turns = None
    loop.max_assistant_turns = max_assistant_turns
    loop.max_parallel_calls = max_parallel_calls
    loop.max_tool_response_length = 256
    loop.tool_response_truncate_side = "middle"
    loop.turn_level_rewards = turn_level_rewards
    loop.prompt_length = 16
    loop.response_length = response_length
    loop.tools = {t.name: t for t in tools}
    loop.tool_schemas = [t.tool_schema.model_dump() for t in tools]
    loop.tool_parser = Parser(tool_call_turns, calls_per_turn, tuple(t.name for t in tools))
    loop.tool_parser_name = "hermes"
    loop.processor = None
    loop.server_manager = Server()

    async def process_multi_modal_info(messages):
        return {}

    async def ct_build_initial_tokens(messages, **kwargs):
        return [101, 102, 103]

    async def ct_merge_assistant_token(runtime_token_ids, assistant_token_ids, response_mask, *a, **kw):
        merge = MergeResult(
            token_ids=list(runtime_token_ids) + list(assistant_token_ids),
            appended_token_count=len(assistant_token_ids),
            kind="assistant",
        )
        return merge, list(response_mask) + [1] * len(assistant_token_ids), None

    async def ct_merge_context_msg(previous_messages, updated_messages, runtime_token_ids, response_mask, *a, **kw):
        # Same edits as ContinuousTokenBuilder.align_response_metadata applies to the mask.
        prefix = list(runtime_token_ids)
        mask = list(response_mask)
        if removed_on_context:
            prefix = prefix[:-removed_on_context]
            mask = mask[:-removed_on_context]
        inserted = [7000 + i for i in range(inserted_on_context)]
        observation = [500 + len(prefix), 501 + len(prefix)]
        merge = MergeResult(
            token_ids=prefix + inserted + observation,
            appended_token_count=len(observation),
            kind="context",
            inserted_token_ids=inserted,
            removed_prefix_token_count=removed_on_context,
        )
        return merge, mask + [0] * len(inserted) + [0] * OBS, None

    loop.process_multi_modal_info = process_multi_modal_info
    loop._get_mm_processor_kwargs = lambda audios: None
    loop._assert_mm_supported = lambda has_multi_modal: None
    loop.ct_build_initial_tokens = ct_build_initial_tokens
    loop.ct_merge_assistant_token = ct_merge_assistant_token
    loop.ct_merge_context_msg = ct_merge_context_msg
    return loop


async def run_episode(loop):
    return await loop.run({}, raw_prompt=[{"role": "user", "content": "go"}])


def mask_runs(mask):
    """Maximal runs of mask-1 tokens as [start, end) spans: what generated_turn_spans must equal."""
    runs, start = [], None
    for i, m in enumerate(list(mask) + [0]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i))
            start = None
    return runs
