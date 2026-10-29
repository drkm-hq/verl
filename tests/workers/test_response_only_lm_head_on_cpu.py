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

"""``response_only_lm_head``: the FSDP engine projects the LM head for loss rows only.

The eager ``use_remove_padding`` path runs the vocabulary projection, the log-prob and the
entropy math on every packed token, while every PPO consumer reads only the row before each
response token (:func:`verl.workers.utils.padding.no_padding_2_padding`,
:func:`verl.workers.utils.padding.response_from_nested`). With the flag on,
``prepare_model_inputs`` hands the model that row index as ``logits_to_keep`` and
``prepare_model_outputs`` scatters the K results back into the packed layout.

These tests run the real ``FSDPEngineWithLMHead.prepare_model_inputs`` /
``prepare_model_outputs`` on a bare engine instance (the pattern of
``test_packed_pad_to_length_on_cpu.py``) around a tiny random Qwen3 model and check that
what both consumers read, plus the gradients through the head and the embedding, are
bitwise identical to the full projection while the head only sees the loss rows.
"""

import logging
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.torch_functional import entropy_from_logits
from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import (
    FSDPEngineWithLMHead,
    FSDPEngineWithValueHead,
    _probe_logits_to_keep_kwarg,
)
from verl.workers.utils.padding import loss_row_index, no_padding_2_padding, response_from_nested

VOCAB, HIDDEN = 512, 64
PROMPT_LENS = [37, 5, 20, 11]
RESPONSE_LENS = [9, 14, 3, 6]


def _nested(rows):
    return torch.nested.as_nested_tensor([torch.tensor(r, dtype=torch.long) for r in rows], layout=torch.jagged)


def _make_micro_batch(temperature=1.0, *, with_responses=True, **non_tensor_overrides):
    """Four packed sequences with the fields the V1 trainer hands the actor / ref engine."""
    generator = torch.Generator().manual_seed(0)
    prompts, responses, input_ids, position_ids, loss_mask = [], [], [], [], []
    for prompt_len, response_len in zip(PROMPT_LENS, RESPONSE_LENS, strict=True):
        prompt = torch.randint(0, VOCAB, (prompt_len,), generator=generator).tolist()
        response = torch.randint(0, VOCAB, (response_len,), generator=generator).tolist()
        prompts.append(prompt)
        responses.append(response)
        input_ids.append(prompt + response)
        position_ids.append(list(range(prompt_len + response_len)))
        loss_mask.append([1] * response_len)
    tensors = {"input_ids": _nested(input_ids), "position_ids": _nested(position_ids), "loss_mask": _nested(loss_mask)}
    if with_responses:
        tensors.update(prompts=_nested(prompts), responses=_nested(responses), response_mask=_nested(loss_mask))
    micro_batch = TensorDict(tensors, batch_size=[len(PROMPT_LENS)])
    if isinstance(temperature, torch.Tensor):
        micro_batch["temperature"] = temperature  # per-sample (bsz,)
    else:
        tu.assign_non_tensor(micro_batch, temperature=temperature)  # scalar, as the V1 trainer passes it
    non_tensor = dict(
        use_remove_padding=True,
        use_fused_kernels=False,
        pad_mode=DatasetPadMode.NO_PADDING,
        pad_token_id=0,
        max_response_len=max(RESPONSE_LENS),
        max_response_length=max(RESPONSE_LENS),
        calculate_entropy=True,
    )
    non_tensor.update(non_tensor_overrides)
    tu.assign_non_tensor(micro_batch, **non_tensor)
    return micro_batch


def _make_engine(response_only_lm_head: bool, *, pad_to_length: bool = False, bucket: int = 64):
    """Bare instance: prepare_model_inputs / prepare_model_outputs only read the attributes set here."""
    engine = FSDPEngineWithLMHead.__new__(FSDPEngineWithLMHead)
    engine.pad_to_length = pad_to_length
    engine.pad_to_length_bucket = bucket
    engine.use_ulysses_sp = False
    engine.ulysses_sequence_parallel_size = 1
    engine.pass_packed_cu_seqlens = False
    engine.engine_config = SimpleNamespace(
        entropy_checkpointing=False, entropy_from_logits_with_chunking=False, entropy_from_logits_chunk_size=2048
    )
    engine.compute_entropy_from_logits = entropy_from_logits
    engine.response_only_lm_head = response_only_lm_head
    engine.logits_to_keep_kwarg = "logits_to_keep"
    return engine


def _make_model():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    return Qwen3ForCausalLM(config).float().eval()


def _forward_backward(engine, model, micro_batch):
    """Run the engine's input / output path around the model.

    Returns what the two consumers read plus the head and embedding gradients, the model inputs,
    the output args, the raw model output and the number of rows the LM head projected.
    """
    projected_rows = []
    handle = model.lm_head.register_forward_hook(lambda _m, inputs, _o: projected_rows.append(inputs[0].shape[1]))
    model.zero_grad(set_to_none=True)
    try:
        model_inputs, output_args = engine.prepare_model_inputs(micro_batch)
        output = model(**model_inputs, use_cache=False)
        model_output = engine.prepare_model_outputs(output, output_args, micro_batch, None)
    finally:
        handle.remove()
    log_probs = no_padding_2_padding(model_output["log_probs"], micro_batch)
    response_mask = torch.nested.to_padded_tensor(micro_batch["response_mask"], 0, output_size=log_probs.shape).bool()
    (log_probs * response_mask).sum().backward()
    read = {
        "log_probs": log_probs,
        "log_probs_nested": response_from_nested(model_output["log_probs"], micro_batch["response_mask"]).values(),
        "entropy": no_padding_2_padding(model_output["entropy"], micro_batch),
        "lm_head.weight.grad": model.lm_head.weight.grad.clone(),
        "embed_tokens.weight.grad": model.model.embed_tokens.weight.grad.clone(),
    }
    if "sum_pi_squared" in model_output:
        read["sum_pi_squared"] = no_padding_2_padding(model_output["sum_pi_squared"], micro_batch)
    return read, model_inputs, output_args, model_output, projected_rows[0]


class TestLossRowIndex:
    def test_matches_the_consumers_slicing_rule(self):
        micro_batch = _make_micro_batch()
        offsets = micro_batch["input_ids"].offsets().tolist()
        expected = [
            row
            for end, response_len in zip(offsets[1:], RESPONSE_LENS, strict=True)
            for row in range(end - response_len - 1, end - 1)
        ]
        index = loss_row_index(micro_batch)
        assert index.dtype == torch.long
        assert index.tolist() == expected
        # the rows the forward-only consumer reads are exactly the indexed rows of the packed stream
        packed = torch.arange(offsets[-1], dtype=torch.float32)
        nested = torch.nested.nested_tensor_from_jagged(packed, micro_batch["input_ids"].offsets())
        assert torch.equal(response_from_nested(nested, micro_batch["response_mask"]).values(), packed[index])

    def test_an_empty_prompt_is_rejected_by_name(self):
        micro_batch = TensorDict(
            {"input_ids": _nested([[1, 2, 3], [4, 5]]), "responses": _nested([[2, 3], [4, 5]])}, batch_size=[2]
        )
        with pytest.raises(ValueError, match=r"sequence 1 has prompt length 0 and response length 2"):
            loss_row_index(micro_batch)


class TestIdentityWithTheFullProjection:
    @pytest.mark.parametrize(
        "temperature, pad_to_length, overrides",
        [
            (1.0, False, {}),
            (torch.tensor([1.0, 0.7, 1.3, 1.0]), True, {}),
            (0.8, False, {"calculate_sum_pi_squared": True}),
        ],
        ids=["unit-temperature", "per-sample-temperature-pad-to-length", "sum-pi-squared"],
    )
    def test_consumers_and_gradients_are_bitwise_identical(self, temperature, pad_to_length, overrides):
        model = _make_model()
        micro_batch = _make_micro_batch(temperature, **overrides)
        full, full_inputs, _, _, full_rows = _forward_backward(
            _make_engine(False, pad_to_length=pad_to_length), model, micro_batch
        )
        kept, kept_inputs, kept_args, kept_output, kept_rows = _forward_backward(
            _make_engine(True, pad_to_length=pad_to_length), model, micro_batch
        )

        assert "logits_to_keep" not in full_inputs
        assert torch.equal(kept_inputs["logits_to_keep"], loss_row_index(micro_batch))
        assert torch.equal(kept_args["loss_row_index"], kept_inputs["logits_to_keep"])
        # the head projected the loss rows only (static pad included in the full count)
        assert (full_rows, kept_rows) == (full_inputs["input_ids"].shape[1], sum(RESPONSE_LENS))
        assert set(full) == set(kept)
        # row-wise maths on identical hidden states: bitwise
        for name in (n for n in full if not n.endswith(".grad")):
            assert torch.equal(full[name], kept[name]), (
                f"{name}: max |diff| {(full[name] - kept[name]).abs().max().item():.3e}"
            )
        # the weight-gradient GEMMs reduce over K rows instead of T rows: the same terms, but a different
        # blocking may round differently (up to 5.6e-7 relative on a 1-token layout), hence a tolerance
        torch.testing.assert_close(kept["lm_head.weight.grad"], full["lm_head.weight.grad"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            kept["embed_tokens.weight.grad"], full["embed_tokens.weight.grad"], rtol=1e-5, atol=1e-6
        )
        # the rows nobody reads are zero in the packed layout every consumer still sees
        packed_log_probs = kept_output["log_probs"].values()
        unread = torch.ones(packed_log_probs.shape[0], dtype=torch.bool)
        unread[kept_args["loss_row_index"]] = False
        assert packed_log_probs.shape[0] == micro_batch["input_ids"].offsets()[-1]
        assert (packed_log_probs[unread] == 0).all()


class TestTheFlagIsANoOpWhenIneligible:
    @pytest.mark.parametrize(
        "flag, batch_kwargs",
        [
            (False, {}),
            (True, {"use_fused_kernels": True}),
            (True, {"with_responses": False}),
            (True, {"distillation_use_topk": True}),
            (True, {"distillation_only": True}),
            (True, {"score_centering": True}),
        ],
        ids=[
            "flag-off",
            "fused-kernels",
            "sft-shaped-batch",
            "topk-distillation",
            "distillation-only",
            "score-centering",
        ],
    )
    def test_no_index_is_passed_or_recorded(self, flag, batch_kwargs):
        model_inputs, output_args = _make_engine(flag).prepare_model_inputs(_make_micro_batch(**batch_kwargs))
        assert "logits_to_keep" not in model_inputs
        assert "loss_row_index" not in output_args

    def test_an_eligible_engine_still_runs_the_full_path_for_an_ineligible_batch(self):
        model, micro_batch = _make_model(), _make_micro_batch(with_responses=False)
        engine = _make_engine(True)
        model_inputs, output_args = engine.prepare_model_inputs(micro_batch)
        model_output = engine.prepare_model_outputs(
            model(**model_inputs, use_cache=False), output_args, micro_batch, None
        )
        assert model_output["log_probs"].values().shape[0] == model_inputs["input_ids"].shape[1]

    def test_a_model_that_ignores_the_index_is_rejected(self):
        model, micro_batch = _make_model(), _make_micro_batch()
        engine = _make_engine(True)
        model_inputs, output_args = engine.prepare_model_inputs(micro_batch)
        model_inputs.pop("logits_to_keep")  # a wrapper that swallowed the kwarg: every row comes back
        with pytest.raises(RuntimeError, match="ignored the row index"):
            engine.prepare_model_outputs(model(**model_inputs, use_cache=False), output_args, micro_batch, None)


def _bare_engine(cls, **attrs):
    engine = cls.__new__(cls)
    engine.engine_config = SimpleNamespace(response_only_lm_head=True, strategy="fsdp2")
    engine.use_remove_padding = True
    engine.use_ulysses_sp = False
    engine.model_config = SimpleNamespace(use_fused_kernels=False)
    engine.logits_to_keep_kwarg = "logits_to_keep"
    for name, value in attrs.items():
        setattr(engine, name, value)
    return engine


class TestBuildTimeResolution:
    def test_probe_requires_the_named_parameter(self):
        def hf_forward(self, input_ids=None, logits_to_keep=0): ...

        def renamed_forward(self, input_ids=None, keep_rows=0): ...

        def patched_vlm_forward(self, input_ids=None, **kwargs): ...

        assert _probe_logits_to_keep_kwarg(SimpleNamespace(forward=hf_forward)) == "logits_to_keep"
        assert _probe_logits_to_keep_kwarg(SimpleNamespace(forward=renamed_forward)) is None
        assert _probe_logits_to_keep_kwarg(SimpleNamespace(forward=patched_vlm_forward)) is None
        assert _probe_logits_to_keep_kwarg(SimpleNamespace(forward=None)) is None
        assert _probe_logits_to_keep_kwarg(_make_model()) == "logits_to_keep"

    def test_a_text_causal_lm_on_the_default_layout_is_eligible(self, caplog):
        with caplog.at_level(logging.WARNING, logger=transformer_impl.logger.name):
            assert _bare_engine(FSDPEngineWithLMHead)._resolve_response_only_lm_head(_make_model()) is True
        assert "ignored" not in caplog.text

    def test_flag_off_resolves_false_without_a_warning(self, caplog):
        engine = _bare_engine(FSDPEngineWithLMHead)
        engine.engine_config.response_only_lm_head = False
        with caplog.at_level(logging.WARNING, logger=transformer_impl.logger.name):
            assert engine._resolve_response_only_lm_head(_make_model()) is False
        assert "ignored" not in caplog.text

    @pytest.mark.parametrize(
        "cls, attrs, module, reason",
        [
            (FSDPEngineWithValueHead, {}, None, "vocabulary projection"),
            (
                FSDPEngineWithLMHead,
                {"model_config": SimpleNamespace(use_fused_kernels=True)},
                None,
                "use_fused_kernels=True",
            ),
            (FSDPEngineWithLMHead, {"use_ulysses_sp": True}, None, "ulysses_sequence_parallel_size"),
            (
                FSDPEngineWithLMHead,
                {"engine_config": SimpleNamespace(response_only_lm_head=True, strategy="fsdp_turbo")},
                None,
                "fsdp_turbo",
            ),
            (
                FSDPEngineWithLMHead,
                {},
                SimpleNamespace(config=SimpleNamespace(vision_config={}), forward=lambda logits_to_keep=0: None),
                "vision-language",
            ),
            (FSDPEngineWithLMHead, {"logits_to_keep_kwarg": None}, None, "does not accept logits_to_keep"),
        ],
        ids=["value-head", "fused-kernels", "ulysses-sp", "fsdp-turbo", "vlm", "no-kwarg"],
    )
    def test_ineligible_layouts_warn_once_and_project_every_row(self, caplog, cls, attrs, module, reason):
        engine = _bare_engine(cls, **attrs)
        with caplog.at_level(logging.WARNING, logger=transformer_impl.logger.name):
            assert engine._resolve_response_only_lm_head(module or _make_model()) is False
        assert caplog.text.count("response_only_lm_head=True is ignored") == 1
        assert reason in caplog.text

    def test_other_ranks_resolve_the_same_without_logging(self, caplog):
        engine = _bare_engine(FSDPEngineWithLMHead, use_ulysses_sp=True, rank=1)
        with caplog.at_level(logging.WARNING, logger=transformer_impl.logger.name):
            assert engine._resolve_response_only_lm_head(_make_model()) is False
        assert "ignored" not in caplog.text

    def test_padded_inputs_with_the_flag_on_are_rejected(self):
        engine = _bare_engine(FSDPEngineWithLMHead, use_remove_padding=False)
        with pytest.raises(ValueError, match="use_remove_padding=True"):
            engine._resolve_response_only_lm_head(_make_model())
