# Adversarial review tests for contrib/p2-oldlogprob-skip (row alignment under dynamic bsz).
"""The update pass's written-back rows must be in the original sample order even when dynamic batch size
re-chunks the mini-batch into uneven micro-batches (seqlen-balanced, reordered). Drives the REAL
FSDPEngine.forward_backward_batch -> prepare_micro_batches -> postprocess_batch_func path with a stub
forward_step whose per-token output is a unique function of the row, for a training pass
(return_model_output=True) and a forward-only pass with a different token budget."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

LENS = [
    9,
    3,
    7,
    12,
    5,
    11,
    4,
    8,
    6,
    10,
]  # ragged rows, so the balanced split is a real permutation


def _rows_fn(ids: torch.Tensor) -> torch.Tensor:
    # unique per row: token value plus position, no model needed to test the ordering
    return ids.float() * 1e-3 + torch.arange(ids.shape[0], dtype=torch.float32)


def _make_data(max_token_len, use_dynamic_bsz=True, micro_bsz=None):
    torch.manual_seed(1)
    ids = [torch.randint(1, 1000, (n,)) for n in LENS]
    data = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(ids, layout=torch.jagged),
            "loss_mask": torch.ones(len(LENS)),
        },
        batch_size=[len(LENS)],
    )
    tu.assign_non_tensor(
        data, use_dynamic_bsz=use_dynamic_bsz, max_token_len_per_gpu=max_token_len
    )
    if micro_bsz is not None:
        tu.assign_non_tensor(data, micro_batch_size_per_gpu=micro_bsz)
    return data, ids


def _engine(seen, forward_only):
    weight = torch.ones(1, requires_grad=True)

    def forward_step(micro_batch, loss_function, forward_only):
        rows = micro_batch["input_ids"].unbind()
        seen.append([r.shape[0] for r in rows])
        out = torch.nested.as_nested_tensor(
            [_rows_fn(r) for r in rows], layout=torch.jagged
        )
        loss = (weight * len(rows)).sum()
        return loss, {
            "model_output": {"log_probs": out},
            "loss": loss.detach().item(),
            "metrics": {},
        }

    return SimpleNamespace(
        ulysses_sequence_parallel_size=1,
        get_data_parallel_group=lambda: None,
        get_data_parallel_size=lambda: 1,
        _gradient_sync_context=lambda is_last_micro_batch: nullcontext(),
        forward_step=forward_step,
        scaler=None,
    )


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")


def _run(
    max_token_len,
    forward_only,
    return_model_output,
    use_dynamic_bsz=True,
    micro_bsz=None,
):
    data, ids = _make_data(max_token_len, use_dynamic_bsz, micro_bsz)
    if return_model_output:
        tu.assign_non_tensor(data, return_model_output=True)
    seen = []
    out = FSDPEngine.forward_backward_batch(
        _engine(seen, forward_only), data, loss_function=None, forward_only=forward_only
    )
    return out, ids, seen


def test_update_pass_rows_match_old_pass_rows_bitwise_in_original_order():
    old, ids, seen_old = _run(
        max_token_len=16, forward_only=True, return_model_output=False
    )
    upd, _, seen_upd = _run(
        max_token_len=24, forward_only=False, return_model_output=True
    )
    assert len(seen_old) > len(seen_upd) >= 2, (
        seen_old,
        seen_upd,
    )  # different, uneven micro-batching
    assert seen_upd != sorted(seen_upd)  # the balanced split really permutes rows
    expected = [_rows_fn(r) for r in ids]
    for name, out in (("old", old), ("update", upd)):
        rows = out["model_output"]["log_probs"].unbind()
        assert len(rows) == len(LENS), name
        for got, exp in zip(rows, expected, strict=True):
            assert torch.equal(got, exp), f"{name} pass rows are not in input order"


def test_training_pass_drops_model_output_without_the_flag():
    upd, _, _ = _run(max_token_len=24, forward_only=False, return_model_output=False)
    assert upd["model_output"] == {}


def test_static_micro_batching_keeps_order_too():
    upd, ids, seen = _run(
        max_token_len=64,
        forward_only=False,
        return_model_output=True,
        use_dynamic_bsz=False,
        micro_bsz=5,
    )
    assert seen == [[9, 3, 7, 12, 5], [11, 4, 8, 6, 10]]
    for got, exp in zip(
        upd["model_output"]["log_probs"].unbind(),
        [_rows_fn(r) for r in ids],
        strict=True,
    ):
        assert torch.equal(got, exp)
