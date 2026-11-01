# Adversarial review tests (wrappers, build-time resolution, end-to-end forward_step) for contrib/p1-lmhead-rows.
import logging
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model

from _helpers import assert_identical, forward_backward, make_engine, make_micro_batch, make_model
from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead, _probe_logits_to_keep_kwarg
from verl.workers.utils.padding import response_from_nested

PROMPTS, RESPONSES = [37, 5, 20, 11], [9, 14, 3, 6]


class TestLoRA:
    def test_peft_wrapped_model_receives_the_index_and_is_identical(self):
        base = make_model()
        model = get_peft_model(
            base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
        )
        # the probe must run on the raw HF module: PEFT's forward only has **kwargs
        assert _probe_logits_to_keep_kwarg(model) is None
        assert _probe_logits_to_keep_kwarg(model.base_model.model) == "logits_to_keep"
        # LoRA params start at zero for B, so perturb them to make the adapter path non-trivial
        for name, p in model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(p, std=0.1)
        mb = make_micro_batch(PROMPTS, RESPONSES, temperature=torch.tensor([1.0, 0.7, 1.3, 1.0]))
        full, _, _, _, full_rows = forward_backward(make_engine(False), model, mb)
        kept, kept_inputs, _, _, kept_rows = forward_backward(make_engine(True), model, mb)
        assert "logits_to_keep" in kept_inputs
        assert (full_rows, kept_rows) == (sum(PROMPTS) + sum(RESPONSES), sum(RESPONSES))
        assert any(k.endswith("lora_A.default.weight.grad") for k in kept)
        assert_identical(full, kept)


class TestTiedEmbeddings:
    def test_tied_head_and_embedding_gradients_are_identical(self):
        model = make_model(tie_word_embeddings=True)
        assert model.get_output_embeddings().weight is model.get_input_embeddings().weight
        mb = make_micro_batch(PROMPTS, RESPONSES, temperature=0.9)
        full, _, _, _, _ = forward_backward(make_engine(False), model, mb)
        kept, _, _, _, rows = forward_backward(make_engine(True), model, mb)
        assert rows == sum(RESPONSES)
        assert_identical(full, kept)


class TestEntropyVariants:
    @pytest.mark.parametrize("checkpointing, chunking", [(True, False), (False, True), (True, True)])
    def test_entropy_checkpointing_and_chunking_are_identical(self, checkpointing, chunking):
        model = make_model()
        mb = make_micro_batch(PROMPTS, RESPONSES)
        full, _, _, _, _ = forward_backward(
            make_engine(False, entropy_checkpointing=checkpointing, chunking=chunking, chunk=8), model, mb
        )
        kept, _, _, _, _ = forward_backward(
            make_engine(True, entropy_checkpointing=checkpointing, chunking=chunking, chunk=8), model, mb
        )
        assert_identical(full, kept)


class TestForwardStepEndToEnd:
    def test_forward_step_with_the_real_module_call(self, monkeypatch):
        """forward_step: micro_batch.to(device) -> prepare_model_inputs -> self.module(**inputs) -> outputs."""
        monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")
        monkeypatch.setattr(transformer_impl, "get_device_name", lambda: "cpu")
        model = make_model()
        mb = make_micro_batch(PROMPTS, RESPONSES)
        outs = {}
        for flag in (False, True):
            engine = make_engine(flag)
            engine.module = model
            engine._autocast_dtype = torch.float32
            loss, out = engine.forward_step(mb, loss_function=None, forward_only=True)
            outs[flag] = response_from_nested(out["model_output"]["log_probs"], mb["response_mask"]).values()
            assert out["model_output"]["log_probs"].values().shape[0] == sum(PROMPTS) + sum(RESPONSES)
        assert torch.equal(outs[False], outs[True])


class TestBF16Characterization:
    def test_cpu_bf16_max_abs_dlogp(self):
        """Not an identity claim: records how far a K-row bf16 GEMM drifts from the T-row one on this CPU."""
        model = make_model(dtype=torch.bfloat16)
        mb = make_micro_batch(PROMPTS, RESPONSES)
        full, _, _, _, _ = forward_backward(make_engine(False), model, mb)
        kept, _, _, _, _ = forward_backward(make_engine(True), model, mb)
        diff = (full["log_probs"].float() - kept["log_probs"].float()).abs().max().item()
        print(f"\nbf16 CPU max |dlogp| = {diff:.3e}; entropy {(full['entropy'].float() - kept['entropy'].float()).abs().max().item():.3e}")
        assert diff <= 5e-2, diff  # loose: the point is the printed number


class TestBuildTimeClaims:
    def _engine(self, **engine_attrs):
        engine = FSDPEngineWithLMHead.__new__(FSDPEngineWithLMHead)
        engine.engine_config = SimpleNamespace(response_only_lm_head=True, strategy="fsdp2")
        engine.use_remove_padding = True
        engine.use_ulysses_sp = False
        engine.logits_to_keep_kwarg = "logits_to_keep"
        for k, v in engine_attrs.items():
            setattr(engine, k, v)
        return engine

    def test_docs_claim_a_start_up_warning_for_fused_kernels(self, caplog):
        """PR body / docs: 'keeps the full projection, with one warning at start-up ... for use_fused_kernels=True'.

        The build-time resolver never looks at model_config.use_fused_kernels, and the fused forward's
        signature (dense_common.py:84) DOES list logits_to_keep, so the flag resolves True silently and
        every micro-batch then no-ops silently. Expected to FAIL on the branch (documents the gap).
        """
        from verl.models.transformers.dense_common import forward_with_torch_backend

        fused = SimpleNamespace(config=SimpleNamespace(), forward=forward_with_torch_backend)
        engine = self._engine(model_config=SimpleNamespace(use_fused_kernels=True))
        engine.logits_to_keep_kwarg = _probe_logits_to_keep_kwarg(fused)
        with caplog.at_level(logging.WARNING, logger=transformer_impl.logger.name):
            resolved = engine._resolve_response_only_lm_head(fused)
        assert resolved is False, "flag resolved True although every batch will no-op (fused kernels)"
        assert "ignored" in caplog.text

    def test_probe_sees_the_fused_forward_as_accepting_the_kwarg(self):
        from verl.models.transformers.dense_common import forward_with_torch_backend, forward_with_triton_backend

        for fn in (forward_with_torch_backend, forward_with_triton_backend):
            assert _probe_logits_to_keep_kwarg(SimpleNamespace(forward=fn)) == "logits_to_keep"
