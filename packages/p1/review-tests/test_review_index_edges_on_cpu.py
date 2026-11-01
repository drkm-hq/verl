# Adversarial review tests (index math, padding, degenerate lengths) for contrib/p1-lmhead-rows.
# Run from the worktree with PYTHONPATH=<worktree> and this directory on the path.
import random

import pytest
import torch

from _helpers import assert_identical, forward_backward, make_engine, make_micro_batch, make_model, nested
from verl.workers.utils.padding import loss_row_index, no_padding_2_padding, response_from_nested

EDGE_PROMPTS = [3, 7, 1, 12, 4]  # includes a 1-token prompt
EDGE_RESPONSES = [1, 0, 5, 2, 0]  # includes response_length 1 and two prompt-only rows (0)


def _brute_force_index(mb):
    offsets = mb["input_ids"].offsets().tolist()
    lens = mb["responses"].offsets().diff().tolist()
    return [row for end, r in zip(offsets[1:], lens, strict=True) for row in range(end - r - 1, end - 1)]


class TestDegenerateLengths:
    def test_response_length_zero_and_one_rows_are_identical_to_the_full_projection(self):
        model = make_model()
        mb = make_micro_batch(EDGE_PROMPTS, EDGE_RESPONSES, temperature=torch.tensor([1.0, 0.7, 1.3, 0.9, 1.0]))
        full, _, _, _, full_rows = forward_backward(make_engine(False), model, mb)
        kept, kept_inputs, kept_args, kept_output, kept_rows = forward_backward(make_engine(True), model, mb)
        assert kept_inputs["logits_to_keep"].tolist() == _brute_force_index(mb)
        assert (full_rows, kept_rows) == (sum(EDGE_PROMPTS) + sum(EDGE_RESPONSES), sum(EDGE_RESPONSES))
        assert_identical(full, kept)
        # prompt-only rows read nothing: their padded log-prob rows are all zero in BOTH paths
        for i, r in enumerate(EDGE_RESPONSES):
            if r == 0:
                assert (kept["log_probs"][i] == 0).all() and (full["log_probs"][i] == 0).all()

    def test_a_micro_batch_of_only_prompt_rows_keeps_the_full_projection_and_the_same_outputs(self):
        model = make_model()
        mb = make_micro_batch([5, 8, 3], [0, 0, 0])
        engine = make_engine(True)
        model_inputs, output_args = engine.prepare_model_inputs(mb)
        assert "logits_to_keep" not in model_inputs and "loss_row_index" not in output_args
        out = engine.prepare_model_outputs(model(**model_inputs, use_cache=False), output_args, mb, None)
        assert out["log_probs"].values().shape[0] == 16
        assert response_from_nested(out["log_probs"], mb["response_mask"]).values().numel() == 0

    def test_single_sequence_single_response_token(self):
        """M=1 vs M=2 GEMMs: forward rows stay bitwise equal, the weight gradients need a tolerance."""
        model = make_model()
        mb = make_micro_batch([1], [1], temperature=0.8)
        full, _, _, _, _ = forward_backward(make_engine(False), model, mb)
        kept, kept_inputs, _, _, rows = forward_backward(make_engine(True), model, mb)
        assert kept_inputs["logits_to_keep"].tolist() == [0] and rows == 1
        diffs = {k: (full[k] - kept[k]).abs().max().item() for k in full}
        print(f"\nsingle-token max|diff| per quantity: {diffs}")
        for k in ("log_probs", "log_probs_nested", "entropy", "entropy_nested"):
            assert diffs[k] == 0.0, k
        assert_identical(full, kept, exact=False, atol=1e-6)
        # the shipped test asserts torch.equal on the gradients; here that assertion would read:
        assert all(diffs[k] == 0.0 for k in ("head.grad", "embed.grad")), (
            f"gradients are NOT bitwise identical for a different GEMM M: {diffs}"
        )

    def test_gradient_bitwise_identity_over_random_layouts(self):
        """How often do the head / embedding gradients differ bitwise (same BLAS, different M)?"""
        rng = random.Random(7)
        mismatches, cases = [], 0
        for _ in range(12):
            bsz = rng.randint(1, 4)
            prompts = [rng.randint(1, 12) for _ in range(bsz)]
            responses = [rng.randint(0, 8) for _ in range(bsz)]
            if sum(responses) == 0:
                continue
            model = make_model(seed=rng.randint(0, 999))
            mb = make_micro_batch(prompts, responses, seed=rng.randint(0, 10**6))
            full, _, _, _, _ = forward_backward(make_engine(False), model, mb)
            kept, _, _, _, _ = forward_backward(make_engine(True), model, mb)
            cases += 1
            assert_identical(full, kept, exact=False, atol=1e-6)  # the sound assertion always holds
            for k in ("head.grad", "embed.grad"):
                if not torch.equal(full[k], kept[k]):
                    mismatches.append((prompts, responses, k, (full[k] - kept[k]).abs().max().item()))
            for k in ("log_probs", "entropy"):
                assert torch.equal(full[k], kept[k]), (prompts, responses, k)
        print(f"\n{len(mismatches)} of {2 * cases} gradient tensors differ bitwise: {mismatches}")


class TestStaticPadInteraction:
    @pytest.mark.parametrize(
        "bucket, prompts, responses",
        [
            (32, [10, 20, 30], [6, 20, 10]),  # T = 96 is an exact bucket multiple: pad 0
            (64, [10, 20, 30], [6, 20, 10]),  # pad 32
            (64, [40, 20, 60], [6, 1, 0]),  # pad 1; last sequence is prompt-only, so the tail has no loss rows
            (128, [2], [3]),  # tiny batch, huge pad (123 pad rows)
        ],
    )
    def test_pad_to_length_variants_are_identical(self, bucket, prompts, responses):
        model = make_model()
        temperature = torch.linspace(0.6, 1.4, len(prompts))
        mb = make_micro_batch(prompts, responses, temperature=temperature, calculate_sum_pi_squared=True)
        full, full_inputs, _, _, full_rows = forward_backward(make_engine(False, pad_to_length=True, bucket=bucket), model, mb)
        kept, kept_inputs, kept_args, kept_output, kept_rows = forward_backward(
            make_engine(True, pad_to_length=True, bucket=bucket), model, mb
        )
        T = sum(prompts) + sum(responses)
        padded = -(-T // bucket) * bucket
        assert full_inputs["input_ids"].shape[1] == padded == full_rows
        assert kept_rows == sum(responses)
        assert kept_inputs["logits_to_keep"].max().item() < T  # never points into the pad
        assert kept_output["log_probs"].values().shape[0] == T  # pad stripped
        assert_identical(full, kept)


class TestFuzzIndexAgainstConsumers:
    def test_random_layouts_match_both_consumers(self):
        rng = random.Random(1234)
        for _ in range(60):
            bsz = rng.randint(1, 6)
            prompts = [rng.randint(1, 9) for _ in range(bsz)]
            responses = [rng.choice([0, 0, 1, 1, 2, 3, 5, 7]) for _ in range(bsz)]
            mb = make_micro_batch(prompts, responses, seed=rng.randint(0, 10**6))
            index = loss_row_index(mb)
            assert index.dtype == torch.long and index.tolist() == _brute_force_index(mb)
            assert (index.diff() > 0).all() if index.numel() > 1 else True
            T = sum(prompts) + sum(responses)
            packed = torch.arange(T, dtype=torch.float32) + 1.0  # 1-based so 0 means "not read"
            packed_nested = torch.nested.nested_tensor_from_jagged(packed, mb["input_ids"].offsets())
            # what the forward-only consumer reads is exactly packed[index]
            assert torch.equal(response_from_nested(packed_nested, mb["response_mask"]).values(), packed[index])
            # what the loss consumer reads, after zeroing everything outside the index, is unchanged
            zeroed = torch.zeros_like(packed).index_copy(0, index, packed[index])
            zeroed_nested = torch.nested.nested_tensor_from_jagged(zeroed, mb["input_ids"].offsets())
            assert torch.equal(no_padding_2_padding(zeroed_nested, mb), no_padding_2_padding(packed_nested, mb))

    def test_prompt_length_zero_is_rejected_not_misread(self):
        mb = make_micro_batch([0, 3], [2, 2])
        with pytest.raises(AssertionError):
            loss_row_index(mb)
        # the consumer has the same guard, so the engine is not stricter than the loss
        values = torch.nested.nested_tensor_from_jagged(torch.arange(7.0), mb["input_ids"].offsets())
        with pytest.raises(AssertionError):
            no_padding_2_padding(values, mb)


def test_nested_helper_sanity():
    assert nested([[1, 2], [3]]).offsets().tolist() == [0, 2, 3]
