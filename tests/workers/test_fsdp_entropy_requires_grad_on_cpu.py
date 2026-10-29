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
"""``entropy_requires_grad`` in ``FSDPEngineWithLMHead.prepare_model_outputs``.

When the entropy is only a metric (the update pass under ``actor.skip_old_log_prob_when_on_policy``
with ``entropy_coeff=0``) it is computed under ``torch.no_grad``: same values, no graph, no
checkpoint, and the in-place cross-entropy backward stays enabled on the remove-padding path.
The default (``True``) is today's behaviour.
"""

import types
from types import SimpleNamespace

import pytest
import torch

import verl.utils.torch_functional as verl_F
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine, FSDPEngineWithLMHead

V = 16
LENS = [5, 3]
_LOGPROBS_FROM_LOGITS = transformer_impl.logprobs_from_logits  # the real one, whatever a test patches in


def _engine(entropy_checkpointing=False):
    engine = SimpleNamespace(
        engine_config=SimpleNamespace(
            entropy_checkpointing=entropy_checkpointing,
            entropy_from_logits_with_chunking=False,
            entropy_from_logits_chunk_size=2048,
        ),
        compute_entropy_from_logits=verl_F.entropy_from_logits,
        use_ulysses_sp=False,
    )
    engine._gather_and_unpad_packed = types.MethodType(FSDPEngine._gather_and_unpad_packed, engine)
    return engine


def _inputs(use_remove_padding: bool, entropy_requires_grad: bool | None):
    torch.manual_seed(0)
    ids = [torch.randint(0, V, (n,)) for n in LENS]
    micro_batch = tu.get_tensordict(tensor_dict={"input_ids": torch.nested.as_nested_tensor(ids, layout=torch.jagged)})
    tu.assign_non_tensor(
        micro_batch, use_remove_padding=use_remove_padding, use_fused_kernels=False, calculate_entropy=True
    )
    if entropy_requires_grad is not None:
        tu.assign_non_tensor(micro_batch, entropy_requires_grad=entropy_requires_grad)
    rolled = torch.cat([torch.roll(x, -1) for x in ids])
    if use_remove_padding:
        total = sum(LENS)
        output = SimpleNamespace(logits=torch.randn(1, total, V, requires_grad=True))
        output_args = {
            "input_ids_rmpad_rolled": rolled,
            "temperature_rmpad": torch.ones(total),
            "pad_size": 0,
            "temperature_is_one": True,
        }
    else:
        output = SimpleNamespace(logits=torch.randn(len(LENS), max(LENS), V, requires_grad=True))
        output_args = {
            "temperature": torch.ones(len(LENS)),
            "input_ids_rmpad_rolled": rolled,
            "temperature_is_one": True,
        }
    return output, output_args, micro_batch


def _run(monkeypatch, use_remove_padding, entropy_requires_grad, entropy_checkpointing=False):
    calls = {"inplace_backward": [], "checkpoint": 0}

    def logprobs_from_logits(logits, labels, inplace_backward=True):
        calls["inplace_backward"].append(inplace_backward)
        return _LOGPROBS_FROM_LOGITS(logits, labels, inplace_backward=inplace_backward)

    def checkpoint(fn, *args, **kwargs):
        calls["checkpoint"] += 1
        return fn(*args)

    monkeypatch.setattr(transformer_impl, "logprobs_from_logits", logprobs_from_logits)
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", checkpoint)
    output, output_args, micro_batch = _inputs(use_remove_padding, entropy_requires_grad)
    model_output = FSDPEngineWithLMHead.prepare_model_outputs(
        _engine(entropy_checkpointing), output, output_args, micro_batch, logits_processor_func=None
    )
    return model_output, calls


@pytest.mark.parametrize("use_remove_padding", [True, False])
def test_no_grad_entropy_has_the_same_values_and_no_graph(monkeypatch, use_remove_padding):
    with_grad, calls_grad = _run(monkeypatch, use_remove_padding, entropy_requires_grad=True)
    no_grad, calls_no_grad = _run(monkeypatch, use_remove_padding, entropy_requires_grad=False)

    assert with_grad["entropy"].requires_grad is True
    assert no_grad["entropy"].requires_grad is False
    assert torch.equal(with_grad["entropy"].values(), no_grad["entropy"].values())
    assert with_grad["log_probs"].requires_grad and no_grad["log_probs"].requires_grad
    assert torch.equal(with_grad["log_probs"].values(), no_grad["log_probs"].values())
    if use_remove_padding:
        # the entropy graph is what forces the out-of-place cross-entropy backward
        assert calls_grad["inplace_backward"] == [False]
        assert calls_no_grad["inplace_backward"] == [True]
    else:
        # this path already computes the log-probs on a copy of the logits
        assert calls_grad["inplace_backward"] == calls_no_grad["inplace_backward"] == [True]


def test_default_is_entropy_with_grad(monkeypatch):
    model_output, calls = _run(monkeypatch, use_remove_padding=True, entropy_requires_grad=None)
    assert model_output["entropy"].requires_grad is True
    assert calls["inplace_backward"] == [False]


@pytest.mark.parametrize("use_remove_padding", [True, False])
def test_no_grad_entropy_skips_entropy_checkpointing(monkeypatch, use_remove_padding):
    _, with_grad = _run(monkeypatch, use_remove_padding, entropy_requires_grad=True, entropy_checkpointing=True)
    _, no_grad = _run(monkeypatch, use_remove_padding, entropy_requires_grad=False, entropy_checkpointing=True)
    assert with_grad["checkpoint"] == 1
    assert no_grad["checkpoint"] == 0
