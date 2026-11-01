# Round-2 adversarial tests for contrib/p1-lmhead-rows @ d7d820aa (not part of the branch).
# Target: the assertion rule of tests/workers/test_response_only_lm_head_on_gpu.py (2634019d, d7d820aa),
# exercised on CPU through the test module's own _check / _make_batch. Each test encodes a claim of that
# module's docstring; a failing test here is a predicted first-run failure of the GPU test.
import importlib.util
import os
import sys

import pytest
import torch

import verl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _helpers import make_engine, make_micro_batch, make_model, no_padding_2_padding  # noqa: E402

GPU_TEST = os.path.join(os.path.dirname(verl.__file__), "..", "tests", "workers", "test_response_only_lm_head_on_gpu.py")
LR = 1e-3  # the GPU test's AdamW lr


@pytest.fixture(scope="module")
def gpu():
    spec = importlib.util.spec_from_file_location("gpu_identity_test", GPU_TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bump_one_ulp(x16: torch.Tensor) -> torch.Tensor:
    """Every element one bf16 ulp further from zero."""
    return (x16.view(torch.int16) + 1).view(torch.bfloat16)


def test_frozen_batch_and_index(gpu):
    from verl.workers.utils.padding import loss_row_index

    batch = gpu._make_batch(151936)
    index = loss_row_index(batch)
    assert batch["input_ids"].values().shape[0] == 1589 and index.numel() == 498
    assert index[:40].tolist() == list(range(40))  # sequence 0: prompt 1, response 40


def test_check_tolerates_adamw_sign_noise_on_lora_B(gpu):
    """Docstring claim: AdamW's first step is ~lr*sign(g), so a sign flip at the noise floor 'is not an
    identity failure'. lora_B starts at zero, so param/ after the step IS that update: flip one element."""
    torch.manual_seed(0)
    key = "param/model.layers.0.self_attn.q_proj.lora_B.default.weight"
    ref = {key: -LR * torch.randn(896, 8).sign()}
    on = {key: ref[key].clone()}
    on[key][0, 0] *= -1
    gpu._check("lora", on, ref)


@pytest.mark.parametrize(
    "tail, label",
    [(None, "weight matrix alone, FSDP2 per-parameter"), (2.0, "same matrix flattened with a norm weight, FSDP1")],
)
def test_check_tolerates_one_sign_flip_in_a_trained_weight(gpu, tail, label):
    """Same claim for a trunk weight: W' = W - lr*sign(g); one element whose gradient sign flipped differs by 2*lr."""
    torch.manual_seed(0)
    w = (torch.randn(1024, 896) * 0.05).clamp(-0.15, 0.15).flatten()
    w[0] = 0.0
    if tail is not None:
        w = torch.cat([w, torch.full((16,), tail)])
    g = torch.randn_like(w).sign()
    key = "param/model.layers.0.mlp.down_proj.weight"
    ref, on = {key: w - LR * g}, {key: w - LR * g}
    on[key][0] = w[0] + LR * g[0]
    gpu._check(label, on, ref)


def test_entropy_tolerance_covers_one_bf16_ulp_of_a_top_logit(gpu):
    """Docstring claim: 'bf16 logits may round differently at the ulp level' and rtol=atol=1e-2 covers it.
    One row, a confident top token at logit 20 (bf16 ulp 0.125), that single logit one ulp higher."""
    torch.manual_seed(0)
    z = (torch.randn(151936) * 2.0).to(torch.bfloat16)
    z[0], z[1] = 20.0, 18.0
    flipped = z.clone()
    flipped[0] = bump_one_ulp(z[:1])[0]
    assert (flipped[0].float() - z[0].float()).item() == 0.125

    def stats(row):
        p = torch.softmax(row.float(), -1)
        return {
            "entropy": -(p * torch.log(p.clamp_min(1e-30))).sum(),
            "sum_pi_squared": (p * p).sum(),
            "log_probs": torch.log_softmax(row.float(), -1)[7],  # some label token
        }

    ref = {f"forward/{k}": v.reshape(1) for k, v in stats(z).items()}
    on = {f"forward/{k}": v.reshape(1) for k, v in stats(flipped).items()}
    for k in ref:
        print(f"{k}: ref={ref[k].item():.4f} delta={(on[k] - ref[k]).abs().item():.4f} tol={1e-2 + 1e-2 * abs(ref[k].item()):.4f}")
    gpu._check("one-ulp top logit", on, ref)


def test_ulp_noise_on_the_head_backward_flips_lora_B_signs(gpu):
    """A different cuBLAS split-K in dh = dlogits @ W (reduction over V) changes a fraction of the bf16 dh
    elements by one ulp. Inject exactly that on a tiny LoRA model and apply the GPU test's rule after one
    AdamW step: the claim is that the param/ comparison survives it."""
    from peft import LoraConfig, TaskType, get_peft_model

    torch.manual_seed(0)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = get_peft_model(make_model(), LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, target_modules=targets))
    engine = make_engine(False)
    mb = make_micro_batch([1, 30, 7, 52, 3, 16], [4, 12, 9, 6, 1, 25])
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    start = {n: p.detach().clone() for n, p in params.items()}

    def ulp_noise(g):
        g16 = g.to(torch.bfloat16)
        hit = torch.rand(g16.shape, generator=torch.Generator().manual_seed(1)) < 1e-2
        return torch.where(hit, bump_one_ulp(g16), g16).to(g.dtype)

    def run(perturb):
        for n, p in params.items():
            p.data.copy_(start[n])
            p.grad = None
        def forward_hook(_module, inputs, _output):  # returns None: the output must not be replaced
            if perturb:
                inputs[0].register_hook(ulp_noise)

        hook = model.get_output_embeddings().register_forward_hook(forward_hook)
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                model_inputs, output_args = engine.prepare_model_inputs(mb)
                model_output = engine.prepare_model_outputs(model(**model_inputs, use_cache=False), output_args, mb, None)
            log_probs = no_padding_2_padding(model_output["log_probs"], mb)
            mask = torch.nested.to_padded_tensor(mb["response_mask"], 0, output_size=log_probs.shape).bool()
            (log_probs * mask).sum().backward()
        finally:
            hook.remove()
        out = {f"grad/{n}": p.grad.detach().float().clone() for n, p in params.items()}
        torch.optim.AdamW(params.values(), lr=LR, weight_decay=0.01).step()
        out.update({f"param/{n}": p.detach().float().clone() for n, p in params.items()})
        return out

    ref, on = run(False), run(True)
    b_keys = [k for k in ref if k.startswith("grad/") and "lora_B" in k]
    flips = sum(((on[k].sign() != ref[k].sign()) & (ref[k] != 0)).sum().item() for k in b_keys)
    total = sum(ref[k].numel() for k in b_keys)
    print(f"lora_B gradient sign flips: {flips} of {total} elements")
    gpu._check("lora ulp-noise", on, ref)
