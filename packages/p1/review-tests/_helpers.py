# Review helpers for the adversarial tests of contrib/p1-lmhead-rows (not part of the worktree).
# Mirrors tests/workers/test_response_only_lm_head_on_cpu.py's fixtures so every test here drives the
# REAL FSDPEngineWithLMHead.prepare_model_inputs / prepare_model_outputs / forward_step.
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.torch_functional import entropy_from_logits, entropy_from_logits_with_chunking
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.utils.padding import no_padding_2_padding, response_from_nested

VOCAB, HIDDEN = 512, 64


def nested(rows):
    return torch.nested.as_nested_tensor([torch.tensor(r, dtype=torch.long) for r in rows], layout=torch.jagged)


def make_micro_batch(prompt_lens, response_lens, temperature=1.0, *, with_responses=True, seed=0, **overrides):
    """Packed micro-batch shaped like the V1 trainer's actor / ref batches (NO_PADDING, nested)."""
    generator = torch.Generator().manual_seed(seed)
    prompts, responses, input_ids, position_ids, loss_mask = [], [], [], [], []
    for p, r in zip(prompt_lens, response_lens, strict=True):
        prompt = torch.randint(0, VOCAB, (p,), generator=generator).tolist()
        response = torch.randint(0, VOCAB, (r,), generator=generator).tolist()
        prompts.append(prompt)
        responses.append(response)
        input_ids.append(prompt + response)
        position_ids.append(list(range(p + r)))
        loss_mask.append([1] * r)
    tensors = {"input_ids": nested(input_ids), "position_ids": nested(position_ids), "loss_mask": nested(loss_mask)}
    if with_responses:
        tensors.update(prompts=nested(prompts), responses=nested(responses), response_mask=nested(loss_mask))
    mb = TensorDict(tensors, batch_size=[len(prompt_lens)])
    if isinstance(temperature, torch.Tensor):
        mb["temperature"] = temperature
    else:
        tu.assign_non_tensor(mb, temperature=temperature)
    non_tensor = dict(
        use_remove_padding=True,
        use_fused_kernels=False,
        pad_mode=DatasetPadMode.NO_PADDING,
        pad_token_id=0,
        max_response_len=max(response_lens),
        max_response_length=max(response_lens),
        calculate_entropy=True,
    )
    non_tensor.update(overrides)
    tu.assign_non_tensor(mb, **non_tensor)
    return mb


def make_engine(flag, *, pad_to_length=False, bucket=64, entropy_checkpointing=False, chunking=False, chunk=8):
    engine = FSDPEngineWithLMHead.__new__(FSDPEngineWithLMHead)
    engine.pad_to_length = pad_to_length
    engine.pad_to_length_bucket = bucket
    engine.use_ulysses_sp = False
    engine.ulysses_sequence_parallel_size = 1
    engine.pass_packed_cu_seqlens = False
    engine.engine_config = SimpleNamespace(
        entropy_checkpointing=entropy_checkpointing,
        entropy_from_logits_with_chunking=chunking,
        entropy_from_logits_chunk_size=chunk,
    )
    engine.compute_entropy_from_logits = entropy_from_logits_with_chunking if chunking else entropy_from_logits
    engine.response_only_lm_head = flag
    engine.logits_to_keep_kwarg = "logits_to_keep"
    return engine


def make_model(*, tie_word_embeddings=False, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        tie_word_embeddings=tie_word_embeddings,
        attn_implementation="eager",
    )
    return Qwen3ForCausalLM(config).to(dtype).eval()


def lm_head_of(model):
    return model.get_output_embeddings()


def forward_backward(engine, model, mb, *, run_model=None):
    """Real input/output path around ``model``; returns what the consumers read + grads + row count."""
    head = lm_head_of(model)
    rows = []
    handle = head.register_forward_hook(lambda _m, inputs, _o: rows.append(inputs[0].shape[1]))
    model.zero_grad(set_to_none=True)
    try:
        model_inputs, output_args = engine.prepare_model_inputs(mb)
        output = (run_model or (lambda **kw: model(**kw, use_cache=False)))(**model_inputs)
        model_output = engine.prepare_model_outputs(output, output_args, mb, None)
    finally:
        handle.remove()
    log_probs = no_padding_2_padding(model_output["log_probs"], mb)
    response_mask = torch.nested.to_padded_tensor(mb["response_mask"], 0, output_size=log_probs.shape).bool()
    (log_probs * response_mask).sum().backward()
    read = {
        "log_probs": log_probs,
        "log_probs_nested": response_from_nested(model_output["log_probs"], mb["response_mask"]).values(),
        "entropy": no_padding_2_padding(model_output["entropy"], mb),
        "entropy_nested": response_from_nested(model_output["entropy"], mb["response_mask"]).values(),
    }
    if head.weight.grad is not None:  # frozen under LoRA
        read["head.grad"] = head.weight.grad.clone()
    if model.get_input_embeddings().weight.grad is not None:
        read["embed.grad"] = model.get_input_embeddings().weight.grad.clone()
    for name, p in model.named_parameters():
        if p.grad is not None and "lora" in name:
            read[f"{name}.grad"] = p.grad.clone()
    if "sum_pi_squared" in model_output:
        read["sum_pi_squared"] = no_padding_2_padding(model_output["sum_pi_squared"], mb)
    return read, model_inputs, output_args, model_output, (rows[0] if rows else None)


def assert_identical(full, kept, *, exact=True, atol=0.0):
    assert set(full) == set(kept), (set(full) ^ set(kept))
    for name, value in full.items():
        other = kept[name]
        if exact:
            assert torch.equal(value, other), f"{name}: max|diff|={(value - other).abs().max().item():.3e}"
        else:
            torch.testing.assert_close(value, other, atol=atol, rtol=0)
