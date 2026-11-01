"""K2 CPU proof: loss-row-only LM-head projection on the DEFAULT FSDP path (rmpad, eager).

What it does (CPU, fp32, transformers 5.16.1, torch 2.13, the campaign's .venv-cpu):
  1. Builds a tiny random Qwen3 model and a 4-sequence packed micro-batch with the same fields the
     V1 engine sees (nested input_ids / position_ids / prompts / responses / loss_mask / temperature).
  2. Runs the REAL FSDPEngineWithLMHead.prepare_model_inputs / prepare_model_outputs (bare instance, no
     process group, the pattern of tests/workers/test_packed_pad_to_length_on_cpu.py) with full logits.
  3. Runs the same model with `logits_to_keep=<loss rows>` (the tensor form HF supports at
     modeling_qwen3.py:494-495) and the proposed K2 output path (index labels/temperature by the kept
     rows, scatter K results back into the (T,) layout), then slices both with the two consumers verl has
     (no_padding_2_padding for the update, response_from_nested for the forward-only passes).
  4. Compares response log-probs, entropies and the gradients (lm_head + embed) between the two paths,
     and reports LM-head FLOPs (FlopCounterMode) and logits bytes.

Run:
  cd <repo> && env TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 \
    .venv-cpu/bin/python -I research/vet/K2/k2_cpu_proof.py
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

for _mod in (
    "veomni",
    "veomni.arguments",
    "veomni.distributed",
    "veomni.distributed.offloading",
    "veomni.distributed.torch_parallelize",
    "veomni.models",
    "veomni.models.auto",
    "veomni.optim",
    "veomni.utils",
    "veomni.utils.moe_router_replay",
    "veomni.utils.seqlen_pos_transform_utils",
    "veomni.models.checkpoint_tensor_loading",
    "veomni.models.checkpoint_tensor_loading.get_checkpoint_tensor_converter",
):
    sys.modules.setdefault(_mod, MagicMock())

import torch  # noqa: E402
import transformers  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from verl.utils import tensordict_utils as tu  # noqa: E402
from verl.utils.dataset.dataset_utils import DatasetPadMode  # noqa: E402
from verl.utils.torch_functional import entropy_from_logits, logprobs_from_logits  # noqa: E402
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead, _scale_logits_by_temperature  # noqa: E402
from verl.workers.utils.padding import no_padding_2_padding, response_from_nested  # noqa: E402

torch.manual_seed(0)
VOCAB, HIDDEN, LAYERS = 512, 64, 2
PROMPT_LENS = [37, 5, 20, 11]
RESP_LENS = [9, 14, 3, 6]


def _nested(rows):
    return torch.nested.as_nested_tensor([torch.tensor(r, dtype=torch.long) for r in rows], layout=torch.jagged)


def make_micro_batch(temperature):
    prompts, responses, input_ids, position_ids, loss_mask = [], [], [], [], []
    for p, r in zip(PROMPT_LENS, RESP_LENS, strict=True):
        pi = torch.randint(0, VOCAB, (p,)).tolist()
        ri = torch.randint(0, VOCAB, (r,)).tolist()
        prompts.append(pi)
        responses.append(ri)
        input_ids.append(pi + ri)
        position_ids.append(list(range(p + r)))
        loss_mask.append([1] * r)
    mb = TensorDict(
        {
            "input_ids": _nested(input_ids),
            "position_ids": _nested(position_ids),
            "prompts": _nested(prompts),
            "responses": _nested(responses),
            "loss_mask": _nested(loss_mask),
            "response_mask": _nested(loss_mask),
        },
        batch_size=[len(PROMPT_LENS)],
    )
    if isinstance(temperature, torch.Tensor):
        mb["temperature"] = temperature  # per-sample (bsz,)
    else:
        tu.assign_non_tensor(mb, temperature=temperature)  # scalar, as the V1 trainer's extra_info passes it
    tu.assign_non_tensor(
        mb,
        use_remove_padding=True,
        use_fused_kernels=False,
        pad_mode=DatasetPadMode.NO_PADDING,
        pad_token_id=0,
        max_response_len=max(RESP_LENS),
        max_response_length=max(RESP_LENS),
        calculate_entropy=True,
    )
    return mb


def make_engine(pad_to_length=False, bucket=64):
    """Bare instance: prepare_model_inputs/outputs only read the attributes set here."""
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
    return engine


def loss_row_index(micro_batch, total_len):
    """Proposed K2 helper: packed positions whose log-prob the loss consumes.

    Exactly the rows no_padding_2_padding (padding.py:127-138) and response_from_nested (padding.py:194-210)
    read: for sequence i with end offset e_i and response length r_i, rows [e_i - r_i - 1, e_i - 1).
    """
    offsets = micro_batch["input_ids"].offsets()
    resp_lens = micro_batch["responses"].offsets().diff()
    starts = offsets[1:] - resp_lens - 1
    k = int(resp_lens.sum())
    seq_id = torch.repeat_interleave(torch.arange(len(resp_lens)), resp_lens)
    excl_cumsum = torch.cumsum(resp_lens, 0) - resp_lens
    idx = starts[seq_id] + (torch.arange(k) - excl_cumsum[seq_id])
    assert idx.max() < total_len
    return idx


def k2_prepare_model_outputs(engine, output, output_args, micro_batch, keep_idx):
    """The proposed delta to prepare_model_outputs (rmpad, eager, no SP): index by kept rows, scatter back."""
    logits_k = output.logits.squeeze(0)  # (K, vocab)
    temperature_k = output_args["temperature_rmpad"][keep_idx]
    labels_k = output_args["input_ids_rmpad_rolled"][keep_idx]
    logits_k = _scale_logits_by_temperature(
        logits_k, temperature_k.unsqueeze(-1), is_unit_temperature=output_args["temperature_is_one"]
    )
    entropy_k = engine.compute_entropy_from_logits(logits_k)
    log_probs_k = logprobs_from_logits(logits=logits_k, labels=labels_k, inplace_backward=False)
    total = output_args["input_ids_rmpad_rolled"].shape[0]
    log_probs = log_probs_k.new_zeros(total).index_copy(0, keep_idx, log_probs_k)  # differentiable scatter
    entropy = entropy_k.new_zeros(total).index_copy(0, keep_idx, entropy_k)
    pad_size = output_args["pad_size"]
    log_probs = engine._gather_and_unpad_packed(log_probs, pad_size)
    entropy = engine._gather_and_unpad_packed(entropy, pad_size)
    cu_seqlens = micro_batch["input_ids"].offsets()
    return {
        "log_probs": torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens),
        "entropy": torch.nested.nested_tensor_from_jagged(entropy, cu_seqlens),
    }


def run_case(name, temperature, pad_to_length):
    print(f"\n=== case: {name} (temperature={temperature}, pad_to_length={pad_to_length}) ===")
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    model = Qwen3ForCausalLM(cfg).float().eval()
    mb = make_micro_batch(temperature)
    engine = make_engine(pad_to_length=pad_to_length)

    model_inputs, output_args = engine.prepare_model_inputs(mb)
    total_len = model_inputs["input_ids"].shape[1]
    keep_idx = loss_row_index(mb, total_len)
    k = keep_idx.numel()
    print(f"packed tokens T={total_len} (static pad {output_args['pad_size']}), loss rows K={k}, kept fraction {k / total_len:.3f}")

    # ---- full-vocab path (what main does today) ----
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as fc_full:
        out_full = model(**model_inputs, use_cache=False)
        mo_full = engine.prepare_model_outputs(out_full, output_args, mb, None)
    lp_full = no_padding_2_padding(mo_full["log_probs"], mb)
    ent_full = no_padding_2_padding(mo_full["entropy"], mb)
    lp_full_nested = response_from_nested(mo_full["log_probs"], mb["response_mask"])
    resp_mask = torch.nested.to_padded_tensor(mb["response_mask"], 0, output_size=lp_full.shape).bool()
    (lp_full * resp_mask).sum().backward()
    grad_head_full = model.lm_head.weight.grad.clone()
    grad_emb_full = model.model.embed_tokens.weight.grad.clone()
    logits_full_bytes = out_full.logits.numel() * out_full.logits.element_size()

    # ---- K2 path: logits_to_keep = loss rows ----
    model.zero_grad(set_to_none=True)
    k2_inputs = dict(model_inputs, logits_to_keep=keep_idx)
    with FlopCounterMode(display=False) as fc_k2:
        out_k2 = model(**k2_inputs, use_cache=False)
        mo_k2 = k2_prepare_model_outputs(engine, out_k2, output_args, mb, keep_idx)
    assert out_k2.logits.shape == (1, k, VOCAB), out_k2.logits.shape
    lp_k2 = no_padding_2_padding(mo_k2["log_probs"], mb)
    ent_k2 = no_padding_2_padding(mo_k2["entropy"], mb)
    lp_k2_nested = response_from_nested(mo_k2["log_probs"], mb["response_mask"])
    (lp_k2 * resp_mask).sum().backward()
    grad_head_k2 = model.lm_head.weight.grad.clone()
    grad_emb_k2 = model.model.embed_tokens.weight.grad.clone()
    logits_k2_bytes = out_k2.logits.numel() * out_k2.logits.element_size()

    # ---- compare ----
    def maxdiff(a, b):
        return (a - b).abs().max().item()

    print(f"response log-probs  max|diff| = {maxdiff(lp_full, lp_k2):.3e}   (allclose atol=1e-6: {torch.allclose(lp_full, lp_k2, atol=1e-6, rtol=1e-6)})")
    print(f"nested consumer     max|diff| = {maxdiff(lp_full_nested.values(), lp_k2_nested.values()):.3e}")
    print(f"response entropy    max|diff| = {maxdiff(ent_full, ent_k2):.3e}")
    print(f"lm_head.weight.grad max|diff| = {maxdiff(grad_head_full, grad_head_k2):.3e}  (|grad| max {grad_head_full.abs().max().item():.3e})")
    print(f"embed.weight.grad   max|diff| = {maxdiff(grad_emb_full, grad_emb_k2):.3e}  (|grad| max {grad_emb_full.abs().max().item():.3e})")
    torch.testing.assert_close(lp_full, lp_k2, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ent_full, ent_k2, atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(grad_head_full, grad_head_k2, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(grad_emb_full, grad_emb_k2, atol=1e-6, rtol=1e-5)

    head_full = fc_full.get_flop_counts()["Qwen3ForCausalLM.lm_head"]
    head_k2 = fc_k2.get_flop_counts()["Qwen3ForCausalLM.lm_head"]
    head_full = sum(head_full.values())
    head_k2 = sum(head_k2.values())
    tot_full, tot_k2 = fc_full.get_total_flops(), fc_k2.get_total_flops()
    print(f"forward FLOPs total  full={tot_full:.3e}  k2={tot_k2:.3e}  ({100 * (1 - tot_k2 / tot_full):.1f}% fewer)")
    print(f"lm_head FLOPs        full={head_full:.3e}  k2={head_k2:.3e}  ({100 * (1 - head_k2 / head_full):.1f}% fewer; expected {100 * (1 - k / total_len):.1f}%)")
    print(f"logits tensor bytes  full={logits_full_bytes}  k2={logits_k2_bytes}  ({100 * (1 - logits_k2_bytes / logits_full_bytes):.1f}% smaller)")
    print("PASS")


if __name__ == "__main__":
    print(f"torch {torch.__version__}  transformers {transformers.__version__}")
    run_case("unit temperature, no static pad", 1.0, False)
    run_case("per-sample temperature, pad_to_length bucket 64", torch.tensor([1.0, 0.7, 1.3, 1.0]), True)
    print("\nALL CASES PASS")
