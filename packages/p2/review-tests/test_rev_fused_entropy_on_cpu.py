# Adversarial review tests for contrib/p2-oldlogprob-skip (fused-kernels branch).
"""use_fused_kernels=True: prepare_model_outputs takes log_probs/entropy straight from the fused op
(transformer_impl.py:1385-1392) and never looks at entropy_requires_grad. Two things must hold for the
skip: the entropy keeps its graph (so entropy_coeff != 0 is not silently zeroed) and log_probs keep
theirs. The flip side, documented in REVIEW.md: with entropy_coeff == 0 the fused path saves nothing."""

from types import SimpleNamespace

import pytest
import torch
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine, FSDPEngineWithLMHead

V, LENS = 16, [5, 3]


def _engine():
    engine = SimpleNamespace(
        engine_config=SimpleNamespace(
            entropy_checkpointing=False,
            entropy_from_logits_with_chunking=False,
            entropy_from_logits_chunk_size=2048,
        ),
        use_ulysses_sp=False,
    )
    import types

    engine._gather_and_unpad_packed = types.MethodType(
        FSDPEngine._gather_and_unpad_packed, engine
    )
    return engine


@pytest.mark.parametrize("entropy_requires_grad", [True, False])
def test_fused_rmpad_entropy_and_log_probs_keep_their_graph(entropy_requires_grad):
    torch.manual_seed(0)
    ids = [torch.randint(0, V, (n,)) for n in LENS]
    total = sum(LENS)
    micro_batch = tu.get_tensordict(
        tensor_dict={
            "input_ids": torch.nested.as_nested_tensor(ids, layout=torch.jagged)
        }
    )
    tu.assign_non_tensor(
        micro_batch,
        use_remove_padding=True,
        use_fused_kernels=True,
        calculate_entropy=True,
        entropy_requires_grad=entropy_requires_grad,
    )
    output = SimpleNamespace(
        log_probs=torch.randn(1, total, requires_grad=True),
        entropy=torch.rand(1, total, requires_grad=True),
    )
    output_args = {
        "input_ids_rmpad_rolled": torch.cat([torch.roll(x, -1) for x in ids]),
        "temperature_rmpad": torch.ones(total),
        "pad_size": 0,
        "temperature_is_one": True,
    }
    model_output = FSDPEngineWithLMHead.prepare_model_outputs(
        _engine(), output, output_args, micro_batch, None
    )
    assert model_output["log_probs"].requires_grad
    # the fused entropy is grad-attached whatever the flag says: no silent zeroing of an entropy bonus,
    # and also no memory saving under the skip on this path
    assert model_output["entropy"].requires_grad
    assert torch.equal(model_output["entropy"].values(), output.entropy.squeeze(0))
