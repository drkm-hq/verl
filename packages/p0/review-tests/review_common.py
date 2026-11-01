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
"""Shared helpers for the adversarial review tests of contrib/p0-loss-horizon.

Run with PYTHONPATH=<repo>/wt/p0 so ``verl`` resolves to the branch; the batch
construction mirrors tests/trainer/ppo/test_loss_horizon_microbatch_invariance_on_cpu.py (same lens, same seed).
"""

import os

import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict

import verl.trainer.config as trainer_config
from verl.utils import tensordict_utils as tu
from verl.workers.utils.losses import ppo_loss, value_loss

CONFIG_DIR = os.path.dirname(trainer_config.__file__)
MODEL_PATH = os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B")
RESPONSE_LENGTH = 16
PROMPT_LENS = [4, 3, 5, 2, 6, 3, 4, 5]
RESP_LENS = [12, 1, 3, 7, 12, 2, 9, 5]
BATCH = len(RESP_LENS)
WHOLE = [list(range(BATCH))]
UNEVEN = [[0], [1, 2], [3, 4, 5, 6, 7]]

BASE_OVERRIDES = [
    f"data.max_response_length={RESPONSE_LENGTH}",
    f"actor_rollout_ref.model.path={MODEL_PATH}",
    f"critic.model.path={MODEL_PATH}",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
    "critic.ppo_micro_batch_size_per_gpu=1",
]


def compose_config(*extra, config_name="ppo_trainer"):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=config_name, overrides=[*BASE_OVERRIDES, *extra])


def nested(tensors):
    return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)


def make_nested_batch(seed=0):
    """The V1 layout: nested rows plus dp_size / batch_num_tokens / global_batch_size as the engine assigns them."""
    g = torch.Generator().manual_seed(seed)

    def rand(n):
        return torch.randn(n, generator=g)

    prompts = [torch.randint(10, 100, (p,), generator=g) for p in PROMPT_LENS]
    responses = [torch.randint(10, 100, (r,), generator=g) for r in RESP_LENS]
    response_mask = [torch.ones(r, dtype=torch.int64) for r in RESP_LENS]
    response_mask[0][4:7] = 0
    data = TensorDict(
        {
            "prompts": nested(prompts),
            "responses": nested(responses),
            "response_mask": nested(response_mask),
            "old_log_probs": nested([-1.5 + 0.5 * rand(r) for r in RESP_LENS]),
            "advantages": nested([rand(r) for r in RESP_LENS]),
            "values": nested([rand(r) for r in RESP_LENS]),
            "returns": nested([rand(r) for r in RESP_LENS]),
        },
        batch_size=[BATCH],
    )
    num_tokens = int(sum(int(m.sum()) for m in response_mask))
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=num_tokens, global_batch_size=BATCH)
    seq_lens = torch.tensor([p + r for p, r in zip(PROMPT_LENS, RESP_LENS, strict=True)])
    model_output = -1.2 + 0.4 * rand(int(seq_lens.sum()))
    return data, model_output, seq_lens


def accumulate(kind, config, data, model_output, seq_lens, splits):
    """Sum of per-micro-batch losses and the accumulated gradient on the flat model output (engine semantics)."""
    leaf = model_output.clone().requires_grad_(True)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), seq_lens.cumsum(0)])
    total = torch.zeros(())
    metrics = None
    for rows in splits:
        micro_batch = tu.index_select_tensor_dict(data, torch.tensor(rows))
        pieces = torch.cat([leaf[offsets[i] : offsets[i + 1]] for i in rows])
        row_offsets = torch.cat([torch.zeros(1, dtype=torch.long), seq_lens[rows].cumsum(0)])
        output = torch.nested.nested_tensor_from_jagged(pieces, row_offsets)
        if kind == "actor":
            loss, metrics = ppo_loss(config, {"log_probs": output}, micro_batch)
        else:
            loss, metrics = value_loss(config, {"values": output}, micro_batch)
        loss.backward()
        total = total + loss.detach()
    return total, leaf.grad.clone(), metrics


def make_dense_batch(seed=0, width=RESPONSE_LENGTH):
    """The V0 layout after ``left_right_2_no_padding``: dense response-side fields padded to ``width`` (the agent
    loop pads to ``rollout.response_length``), ``max_response_len`` carried, and a flat unpadded model output."""
    data, model_output, seq_lens = make_nested_batch(seed)
    prompt_width = max(PROMPT_LENS)
    dense = {}
    for key in ("prompts",):
        rows = [torch.nn.functional.pad(t, (prompt_width - len(t), 0)) for t in data[key].unbind()]  # left pad
        dense[key] = torch.stack(rows)
    for key in ("responses", "response_mask", "old_log_probs", "advantages", "values", "returns"):
        rows = [torch.nn.functional.pad(t, (0, width - len(t))) for t in data[key].unbind()]
        dense[key] = torch.stack(rows)
    attention_mask = torch.zeros(BATCH, prompt_width + width, dtype=torch.int64)
    for i, (p, r) in enumerate(zip(PROMPT_LENS, RESP_LENS, strict=True)):
        attention_mask[i, prompt_width - p : prompt_width + r] = 1
    dense["attention_mask"] = attention_mask
    out = TensorDict(dense, batch_size=[BATCH])
    tu.assign_non_tensor(out, dp_size=1, batch_num_tokens=data["batch_num_tokens"], global_batch_size=BATCH)
    tu.assign_non_tensor_data(out, "max_response_len", width)
    return out, model_output, seq_lens
