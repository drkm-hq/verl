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
"""Run one scripted episode with whatever verl is on PYTHONPATH (main clone or the branch, flag off) and
dump the ``as_dict`` row plus a V0 ``_postprocess`` row to a torch file: ``python episode_dump.py OUT``."""

import asyncio
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from scripted_loop import StepTool, make_loop, run_episode  # noqa: E402

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopWorker, _InternalAgentLoopOutput


def main(out_path):
    import verl

    # No ``done``: main ignores it, so an identity comparison needs an episode that ends the pre-existing way
    # (the model stops calling tools on turn 4). One trailing generated token is removed per context merge.
    tool = StepTool([0.25, -0.5, 1.0], done_on=None)
    loop = make_loop([tool], tool_call_turns=3, turn_level_rewards=False, removed_on_context=1)
    output = asyncio.run(run_episode(loop))
    output.reward_score = 1.0
    row = output.as_dict()
    row["extra_fields"] = dict(row["extra_fields"])
    row["server_calls"] = loop.server_manager.calls
    row["verl_file"] = verl.__file__

    internal = _InternalAgentLoopOutput(
        prompt_ids=torch.tensor([[0, 7, 8]]),
        response_ids=torch.tensor([[11, 12, 13, 14, 15, 0]]),
        input_ids=torch.tensor([[0, 7, 8, 11, 12, 13, 14, 15, 0]]),
        position_ids=torch.arange(9).unsqueeze(0),
        response_mask=torch.tensor([[1, 1, 0, 1, 0, 0]]),
        attention_mask=torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 0]]),
        reward_score=1.0,
        num_turns=4,
        metrics=AgentLoopMetrics(),
        extra_fields={"turn_scores": [], "tool_rewards": [0.5]},
    )
    worker = SimpleNamespace(reward_loop_worker_handles=None)
    v0 = AgentLoopWorker._postprocess(worker, [internal])
    row["v0_batch"] = {k: v.clone() for k, v in v0.batch.items()}
    row["v0_non_tensor_keys"] = sorted(v0.non_tensor_batch.keys())
    row["v0_meta"] = v0.meta_info
    torch.save(row, out_path)


if __name__ == "__main__":
    main(sys.argv[1])
