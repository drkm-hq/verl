# Re-verification (adversarial) tests for the review fixes on contrib/p1-lmhead-rows (HEAD b32c777e).
import logging
from types import SimpleNamespace

import pytest
import torch

from _helpers import assert_identical, forward_backward, make_engine, make_micro_batch, make_model
from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import (
    FSDPEngineWithLMHead,
    FSDPEngineWithValueHead,
    _probe_logits_to_keep_kwarg,
)
from verl.workers.utils.padding import loss_row_index, response_from_nested


def _brute_force_index(mb):
    offsets = mb["input_ids"].offsets().tolist()
    lens = mb["responses"].offsets().diff().tolist()
    return [row for end, r in zip(offsets[1:], lens, strict=True) for row in range(end - r - 1, end - 1)]


class TestSyncFreeSizing:
    def test_all_prompt_micro_batch_with_pad_to_length_falls_back_identically(self):
        """D5 fix: total comes from responses.values().shape[0]; here that is 0 and the pad is 16 rows."""
        model = make_model()
        mb = make_micro_batch([5, 8, 3], [0, 0, 0])
        index = loss_row_index(mb)
        assert index.dtype == torch.long and index.shape == (0,)
        off, on = make_engine(False, pad_to_length=True, bucket=32), make_engine(True, pad_to_length=True, bucket=32)
        inputs_on, args_on = on.prepare_model_inputs(mb)
        assert "logits_to_keep" not in inputs_on and "loss_row_index" not in args_on
        assert inputs_on["input_ids"].shape[1] == 32 and args_on["pad_size"] == 16
        out_off = off.prepare_model_outputs(model(**off.prepare_model_inputs(mb)[0], use_cache=False), args_on, mb, None)
        out_on = on.prepare_model_outputs(model(**inputs_on, use_cache=False), args_on, mb, None)
        assert torch.equal(out_off["log_probs"].values(), out_on["log_probs"].values())
        assert out_on["log_probs"].values().shape[0] == 16

    def test_sized_from_the_shape_matches_the_offsets_on_sliced_micro_batches(self):
        """A micro-batch cut out of a larger TensorDict must not see the parent's values buffer."""
        from verl.utils import tensordict_utils as tu

        parent = make_micro_batch([4, 6, 2, 9, 5, 3], [2, 0, 3, 1, 4, 2])
        cuts = [
            ("index_select list", tu.index_select_tensor_dict(parent, [0, 2, 5])),
            ("index_select tensor", tu.index_select_tensor_dict(parent, torch.tensor([4, 1]))),
            ("index_select contiguous", tu.index_select_tensor_dict(parent, [1, 2, 3])),
        ]
        cuts += [(f"chunk_tensordict[{i}]", c) for i, c in enumerate(tu.chunk_tensordict(parent, 3))]
        for name, sub in cuts:
            r = sub["responses"]
            assert r.values().shape[0] == int(r.offsets().diff().sum()), name
            assert sub["input_ids"].offsets()[0].item() == 0, name
            assert loss_row_index(sub).tolist() == _brute_force_index(sub), name

    def test_no_device_to_host_sync_in_the_index_path(self, monkeypatch):
        """Any .item()/.tolist()/bool()/int() on a tensor during index construction would be a sync on GPU."""
        mb = make_micro_batch([3, 7, 1, 12, 4], [1, 0, 5, 2, 0])

        def boom(self, *a, **k):
            raise RuntimeError("host sync")

        for name in ("item", "tolist", "__bool__", "__int__", "__index__", "__float__", "numpy"):
            monkeypatch.setattr(torch.Tensor, name, boom)
        engine = make_engine(True)
        index = engine._response_only_lm_head_index(mb, use_fused_kernels=False)
        monkeypatch.undo()
        assert index.tolist() == _brute_force_index(mb)
        assert torch.equal(index, loss_row_index(mb))


class TestBuildTimeGuardsAfterTheFixes:
    def _engine(self, cls=FSDPEngineWithLMHead, **attrs):
        engine = cls.__new__(cls)
        engine.engine_config = SimpleNamespace(response_only_lm_head=True, strategy="fsdp2")
        engine.use_remove_padding = True
        engine.use_ulysses_sp = False
        engine.model_config = SimpleNamespace(use_fused_kernels=False)
        engine.logits_to_keep_kwarg = "logits_to_keep"
        for k, v in attrs.items():
            setattr(engine, k, v)
        return engine

    def test_padded_inputs_raise_with_both_knobs_named(self):
        with pytest.raises(ValueError, match="use_remove_padding=True.*response_only_lm_head=False"):
            self._engine(use_remove_padding=False)._resolve_response_only_lm_head(make_model())

    def test_padded_inputs_do_not_raise_when_the_flag_is_off_or_on_the_critic(self, caplog):
        engine = self._engine(use_remove_padding=False)
        engine.engine_config.response_only_lm_head = False
        with caplog.at_level(logging.WARNING, logger=transformer_impl.logger.name):
            assert engine._resolve_response_only_lm_head(make_model()) is False
            critic = self._engine(FSDPEngineWithValueHead, use_remove_padding=False, rank=0)
            assert critic._resolve_response_only_lm_head(make_model()) is False
        assert caplog.text.count("ignored") == 1 and "vocabulary projection" in caplog.text

    def test_raise_precedes_the_fused_kernels_warning(self):
        engine = self._engine(use_remove_padding=False, model_config=SimpleNamespace(use_fused_kernels=True))
        with pytest.raises(ValueError):
            engine._resolve_response_only_lm_head(make_model())

    @pytest.mark.parametrize("rank, expected_warnings", [(0, 1), (7, 0), (None, 1)], ids=["rank0", "rank7", "no-rank-attr"])
    def test_warning_fires_once_on_rank_zero_only(self, caplog, rank, expected_warnings):
        attrs = {"model_config": SimpleNamespace(use_fused_kernels=True)}
        if rank is not None:
            attrs["rank"] = rank
        engine = self._engine(**attrs)
        with caplog.at_level(logging.WARNING, logger=transformer_impl.logger.name):
            assert engine._resolve_response_only_lm_head(make_model()) is False
            assert engine._resolve_response_only_lm_head(make_model()) is False  # a second call logs again
        assert caplog.text.count("response_only_lm_head=True is ignored") == 2 * expected_warnings
        if expected_warnings:
            assert "use_fused_kernels=True" in caplog.text

    def test_alias_is_gone_and_the_literal_key_is_used(self):
        assert _probe_logits_to_keep_kwarg(SimpleNamespace(forward=lambda slice_logits_to_keep=0: None)) is None
        engine = make_engine(True)
        engine.logits_to_keep_kwarg = "anything"  # attribute no longer steers the key
        inputs, args = engine.prepare_model_inputs(make_micro_batch([4, 3], [2, 1]))
        assert torch.equal(inputs["logits_to_keep"], args["loss_row_index"])


class TestDroppedAssertCharacterization:
    def test_empty_prompt_in_the_first_row(self):
        """After D5 the engine no longer asserts prompt_len > 0: record what an empty prompt does now."""
        model = make_model()
        mb = make_micro_batch([0, 3], [2, 2])
        index = loss_row_index(mb)
        assert index.tolist() == [-1, 0, 4, 5]  # -1: the index silently wraps in the HF slice (reviewer's first guess was [-1,0,3,4])
        engine = make_engine(True)
        inputs, args = engine.prepare_model_inputs(mb)
        out = model(**inputs, use_cache=False)
        with pytest.raises(IndexError, match="index -1 is out of bounds"):  # index_copy (on CUDA: device-side assert)
            engine.prepare_model_outputs(out, args, mb, None)

    def test_empty_prompt_in_a_middle_row_reads_the_same_wrong_row_as_the_full_path(self):
        model = make_model()
        mb = make_micro_batch([3, 0, 4], [2, 2, 1])
        off = make_engine(False)
        out_off = off.prepare_model_outputs(model(**off.prepare_model_inputs(mb)[0], use_cache=False), off.prepare_model_inputs(mb)[1], mb, None)
        on = make_engine(True)
        inputs, args = on.prepare_model_inputs(mb)
        out_on = on.prepare_model_outputs(model(**inputs, use_cache=False), args, mb, None)
        a = response_from_nested(out_off["log_probs"], mb["response_mask"]).values()
        b = response_from_nested(out_on["log_probs"], mb["response_mask"]).values()
        assert torch.equal(a, b)  # forward-only consumers agree (both read row 4, the previous sequence's last row)


def test_identity_still_holds_after_the_fixes():
    model = make_model()
    mb = make_micro_batch([37, 5, 20, 11], [9, 14, 3, 6], temperature=torch.tensor([1.0, 0.7, 1.3, 1.0]), calculate_sum_pi_squared=True)
    full, _, _, _, full_rows = forward_backward(make_engine(False, pad_to_length=True), model, mb)
    kept, _, _, _, kept_rows = forward_backward(make_engine(True, pad_to_length=True), model, mb)
    assert (full_rows, kept_rows) == (128, 32)
    assert_identical(full, kept, exact=False, atol=1e-6)
    for k in ("log_probs", "entropy", "sum_pi_squared"):
        assert torch.equal(full[k], kept[k])
