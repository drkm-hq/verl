# Round-2 adversarial tests for contrib/p1-lmhead-rows @ d7d820aa (not part of the branch).
# Target: 724d3d5e, the logits_to_keep probe through a PEFT wrapper, the bare-**kwargs rule and the
# build-time guards seen through the wrapper. Run with PYTHONPATH=<worktree> and the CPU harness.
import logging
import os
import sys
import types
from inspect import signature
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # review-tests/_helpers.py
from _helpers import assert_identical, forward_backward, make_engine, make_micro_batch, make_model  # noqa: E402

from verl.workers.engine.fsdp.transformer_impl import _probe_logits_to_keep_kwarg  # noqa: E402


def bare_engine(*, flag=True, strategy="fsdp", rank=0):
    """A bare FSDPEngineWithLMHead with exactly the attributes _resolve_response_only_lm_head reads."""
    engine = make_engine(flag)
    engine.engine_config = SimpleNamespace(
        response_only_lm_head=flag,
        strategy=strategy,
        entropy_checkpointing=False,
        entropy_from_logits_with_chunking=False,
        entropy_from_logits_chunk_size=8,
    )
    engine.model_config = SimpleNamespace(use_fused_kernels=False)
    engine.use_remove_padding = True
    engine.rank = rank
    return engine


def lora(model, **kwargs):
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], **kwargs)
    return get_peft_model(model, config)


def bare_kwargs_forward(model):
    """Instance-level forward whose signature is (input_ids=None, **kwargs): it would swallow the index."""
    original = model.forward

    def forward(self, input_ids=None, **kwargs):
        return original(input_ids=input_ids, **kwargs)

    model.forward = types.MethodType(forward, model)
    return model


class TestProbeThroughPeft:
    def test_wrapper_resolves_through_the_base_model(self, caplog):
        peft = lora(make_model())
        assert isinstance(peft, PeftModel)
        print("PEFT forward names logits_to_keep itself:", "logits_to_keep" in signature(peft.forward).parameters)
        assert _probe_logits_to_keep_kwarg(peft) == "logits_to_keep"
        assert _probe_logits_to_keep_kwarg(peft.get_base_model()) == "logits_to_keep"
        engine = bare_engine()
        engine.logits_to_keep_kwarg = _probe_logits_to_keep_kwarg(peft)
        with caplog.at_level(logging.WARNING):
            assert engine._resolve_response_only_lm_head(peft) is True
        assert "response_only_lm_head" not in caplog.text

    def test_bare_kwargs_forward_resolves_off(self, caplog):
        model = bare_kwargs_forward(make_model())
        assert _probe_logits_to_keep_kwarg(model) is None
        engine = bare_engine()
        engine.logits_to_keep_kwarg = None
        with caplog.at_level(logging.WARNING):
            assert engine._resolve_response_only_lm_head(model) is False
        assert "does not accept logits_to_keep" in caplog.text

    def test_peft_wrapper_around_a_bare_kwargs_forward_resolves_off(self, caplog):
        peft = lora(bare_kwargs_forward(make_model()))
        assert _probe_logits_to_keep_kwarg(peft) is None, "the wrapper must not hide the base model's signature"
        engine = bare_engine()
        engine.logits_to_keep_kwarg = None
        with caplog.at_level(logging.WARNING):
            assert engine._resolve_response_only_lm_head(peft) is False
        assert "does not accept logits_to_keep" in caplog.text

    def test_vlm_guard_is_seen_through_the_wrapper(self, caplog):
        model = make_model()
        model.config.vision_config = SimpleNamespace(hidden_size=8)
        peft = lora(model)
        engine = bare_engine()
        engine.logits_to_keep_kwarg = _probe_logits_to_keep_kwarg(peft)
        with caplog.at_level(logging.WARNING):
            assert engine._resolve_response_only_lm_head(peft) is False
        assert "vision-language" in caplog.text

    def test_other_ranks_resolve_off_silently_through_the_wrapper(self, caplog):
        peft = lora(bare_kwargs_forward(make_model()))
        engine = bare_engine(rank=3)
        engine.logits_to_keep_kwarg = _probe_logits_to_keep_kwarg(peft)
        with caplog.at_level(logging.WARNING):
            assert engine._resolve_response_only_lm_head(peft) is False
        assert caplog.text == ""


class TestIndexThroughPeft:
    @pytest.mark.parametrize("lora_dropout", [0.0, 0.3])
    def test_resolved_on_the_wrapper_the_index_reaches_the_base_model(self, lora_dropout):
        """Resolver run AFTER the wrap (the build order the submitter asked about) and the identity through PEFT's
        ``**kwargs``; lora_dropout=0.3 in eval mode (the engine's forward-only pass) must not break it."""
        peft = lora(make_model(), lora_dropout=lora_dropout).eval()
        engine_on, engine_off = bare_engine(flag=True), bare_engine(flag=False)
        for engine in (engine_on, engine_off):
            engine.logits_to_keep_kwarg = _probe_logits_to_keep_kwarg(peft)
            engine.response_only_lm_head = engine._resolve_response_only_lm_head(peft)
        assert engine_on.response_only_lm_head is True and engine_off.response_only_lm_head is False
        mb = make_micro_batch([1, 7, 3], [5, 2, 9])

        def run(**kwargs):
            return peft(**kwargs, use_cache=False)

        full, _, _, _, rows_full = forward_backward(engine_off, peft, mb, run_model=run)
        kept, inputs, _, _, rows_kept = forward_backward(engine_on, peft, mb, run_model=run)
        assert "logits_to_keep" in inputs and rows_full == 27 and rows_kept == 16
        exact = {k: v for k, v in full.items() if not k.endswith(".grad")}
        assert_identical(exact, {k: kept[k] for k in exact})
        grads = {k: v for k, v in full.items() if k.endswith(".grad")}
        assert grads, "LoRA gradients must exist"
        assert_identical(grads, {k: kept[k] for k in grads}, exact=False, atol=1e-6)
        assert torch.isfinite(kept["log_probs"]).all()
