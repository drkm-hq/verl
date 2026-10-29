# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""GPU identity of ``response_only_lm_head`` on the real FSDP engine: flag on against flag off.

One frozen packed micro-batch (six sequences, prompt / response lengths from one token to a few
hundred, ``use_remove_padding=True``) goes through ``FSDPEngineWithLMHead`` built from the same
checkpoint and seed with ``fsdp_config.response_only_lm_head`` off and on. Compared after an fp32
upcast: (a) response log-probs, (b) entropy and (c) ``sum_pi_squared`` from a forward-only pass
and from the training forward, (d) every parameter gradient after one backward of the real PPO
policy loss with ``entropy_coeff > 0`` (the entropy is in the loss), (e) the parameters before and
after one optimizer step. Variants: plain, gradient checkpointing, LoRA (PEFT, rank 8), each on
``fsdp`` (FSDP1, the default strategy) and ``fsdp2``. Each case runs five engines:

1. bf16 (the default mixed precision: fp32 master weights, bf16 compute, flash-attn 2), flag off;
2. the same again, the determinism baseline: bitwise equal to 1 (flash-attn runs with
   ``FLASH_ATTENTION_DETERMINISTIC=1``), so every flag-on delta is the feature's;
3. bf16, flag on;
4. the fp32 reference (``mixed_precision`` param / reduce / buffer dtype fp32, so no autocast; sdpa
   attention on torch's math kernel, since flash-attn has no fp32; TF32 off), flag off;
5. fp32, flag on.

Measured on 1x H100 (torch 2.13 cu130, transformers 5.16.1, flash-attn 2.8.3, Qwen2.5-0.5B;
``bench/results/gpu-identity-p1-3`` and ``-p1-4``), the same in all six cases:

* forward: response log-probs, entropy and the loss bitwise identical with the flag on and off, in
  bf16 and in fp32; ``sum_pi_squared`` within 1.1e-7 (the same logits under a different fp32
  reduction grouping over the vocabulary). Asserted as measured: exact, 1e-6 on ``sum_pi_squared``
  (1e-4 for both in fp32, where a re-tiled GEMM would add accumulation-order noise of ~1e-5).
* fp32 gradients: flag on and off agree on every tensor, 1.1e-5 .. 1.4e-5 relative Frobenius
  pooled per layer (worst tensor 5.7e-5, a layer-norm weight). The K-row and T-row GEMMs are the
  same math in fp32, so this is the correctness proof; asserted at 1e-4 per tensor.
* bf16 gradients: flag on and off differ by 1.7e-2 relative Frobenius at layer 0, falling to
  2.2e-3 at layer 23 (3.6e-2 on the worst layer-0 tensor, up to 1.4e-1 on q/k biases). The head's
  backward GEMMs split the vocabulary reduction differently for K and T rows, and every bf16
  backward layer re-rounds that one-ulp difference: rounding amplification, as the fp32 agreement
  shows. No arm-vs-arm bf16 tolerance can separate that from a defect (a relative measure is
  dominated by ulp noise on small-norm tensors), so the criterion is scale-free and against the
  truth: with ``g32`` the fp32 flag-off gradient, ``D_off = |g_off_bf16 - g32|`` and
  ``D_on = |g_on_bf16 - g32|`` per tensor, the flag must not make the bf16 approximation of the
  fp32 gradient worse than the baseline's own error, ``D_on <= 2 * D_off + 1e-6 * |g32|``
  (independent rounding noise gives sqrt(2); 2 is margin; the floor covers tensors whose errors
  vanish). Both distances and their ratio are printed per layer. The baseline's own bf16 error is
  2.9e-2 nat on the log-probs (``fp32 vs bf16 flag-off`` line, max 4.3e-1), three orders above the
  fp32 agreement.
* optimizer step: plain SGD (lr 1e-3, no momentum, no weight decay, clipping disabled), so within
  every arm ``param_after == param_before - lr * grad`` up to fp32 rounding (a residual of at most
  4 ulp of ``|param| + |lr * grad|``, asserted), and the weights before the step must be bitwise
  identical across arms. The update is not compared across arms: ``param_after - param_before`` is
  a catastrophic cancellation that carries ``eps_fp32 * |param| / |update|`` of noise (1.0e-4 ..
  1.3e-4 relative on FSDP1 flat params in fp32), and in bf16 it is the gradient again; its
  relative difference and sign agreement are printed. The parameters after the step may differ by
  two first-step updates plus 1e-2 relative (the form that also holds for AdamW's sign-like step).
* fp32 reference sanity: its flag-off log-probs must agree with the bf16 flag-off ones within bf16
  noise (mean |delta| below 0.5 nat; measured 2.9e-2), which rules out a broken packed-attention
  mask behind an fp32 identity.

verl's stock GPU unit-test selection collects this file and runs it with ``-x``, so the measured
deltas are printed and the bounds carry margin. Run it with
``pytest -s -rs tests/workers/test_response_only_lm_head_on_gpu.py`` on one CUDA device with
Qwen2.5-0.5B at ``~/models/Qwen/Qwen2.5-0.5B`` or ``/root/models/Qwen/Qwen2.5-0.5B`` (or set
``VERL_RESPONSE_ONLY_LM_HEAD_MODEL``); without a device or a model the six cases skip and the skip
reason names the paths tried.
"""

import os

os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")  # flash-attn backward without atomics
os.environ.setdefault("NCCL_DEBUG", "WARN")

import re
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict
from torch.distributed.tensor import DTensor

from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.device import get_nccl_backend, get_torch_device
from verl.workers.config import ActorConfig, FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

MODEL_CANDIDATES = [
    os.path.expanduser(path)
    for path in (
        os.environ.get("VERL_RESPONSE_ONLY_LM_HEAD_MODEL"),
        "~/models/Qwen/Qwen2.5-0.5B",
        "/root/models/Qwen/Qwen2.5-0.5B",
    )
    if path
]
MODEL_PATH = next((path for path in MODEL_CANDIDATES if os.path.isdir(path)), "")
PROMPT_LENS = [1, 300, 77, 520, 33, 160]  # a one-token prompt is the shortest the guard allows
RESPONSE_LENS = [40, 128, 9, 64, 1, 256]
VARIANTS = {
    "plain": {},
    "gradient-checkpointing": {"enable_gradient_checkpointing": True},
    "lora": {"lora_rank": 8, "lora_alpha": 16},
}
STRATEGIES = ["fsdp", "fsdp2"]
LR = 1e-3  # plain SGD: the update is exactly -LR * grad
FP32_EPS = 2.0**-23
TOLERANCES = {  # rows / sum_pi_squared: (rtol, atol) per element; grad: fp32 arm-vs-arm relative Frobenius per tensor
    "bf16": {"rows": (0.0, 0.0), "sum_pi_squared": (0.0, 1e-6)},
    "fp32": {"rows": (0.0, 1e-4), "sum_pi_squared": (0.0, 1e-4), "grad": 1e-4},
}
TRUTH_RATIO, TRUTH_FLOOR = 2.0, 1e-6  # bf16: D_on <= TRUTH_RATIO * D_off + TRUTH_FLOOR * |g_fp32| per tensor
SGD_RESIDUAL_ULPS = 4.0  # |param_after - (param_before - lr * grad)| within this many fp32 ulp of |param| + |lr * grad|
CROSS_PRECISION_MEAN_NAT = 0.5  # bf16 vs fp32 flag-off log-probs: bf16 noise, not a different attention


def _nested(rows, dtype=torch.long):
    return torch.nested.as_nested_tensor([torch.tensor(r, dtype=dtype) for r in rows], layout=torch.jagged)


def _make_batch(vocab_size: int) -> TensorDict:
    """Six packed sequences with the fields the actor worker hands the engine, as one micro-batch."""
    generator = torch.Generator().manual_seed(0)
    prompts, responses, input_ids, position_ids, loss_mask, advantages = [], [], [], [], [], []
    for prompt_len, response_len in zip(PROMPT_LENS, RESPONSE_LENS, strict=True):
        prompt = torch.randint(0, vocab_size, (prompt_len,), generator=generator).tolist()
        response = torch.randint(0, vocab_size, (response_len,), generator=generator).tolist()
        prompts.append(prompt)
        responses.append(response)
        input_ids.append(prompt + response)
        position_ids.append(list(range(prompt_len + response_len)))
        loss_mask.append([1] * response_len)
        advantages.append(torch.randn(response_len, generator=generator).tolist())
    batch = TensorDict(
        {
            "input_ids": _nested(input_ids),
            "position_ids": _nested(position_ids),
            "loss_mask": _nested(loss_mask),
            "prompts": _nested(prompts),
            "responses": _nested(responses),
            "response_mask": _nested(loss_mask),
            "advantages": _nested(advantages, torch.float32),
            "old_log_probs": _nested([[0.0] * r for r in RESPONSE_LENS], torch.float32),  # frozen in _run_arm
        },
        batch_size=[len(PROMPT_LENS)],
    )
    tu.assign_non_tensor(
        batch,
        temperature=1.0,
        use_remove_padding=True,
        use_fused_kernels=False,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=len(PROMPT_LENS),
        global_batch_size=len(PROMPT_LENS),
        pad_mode=DatasetPadMode.NO_PADDING,
        pad_token_id=0,
        max_response_len=max(RESPONSE_LENS),
        max_response_length=max(RESPONSE_LENS),
        calculate_entropy=True,
        calculate_sum_pi_squared=True,
        return_model_output=True,
    )
    return batch


def _force_fp32_math() -> None:
    """fp32 means fp32: no TF32 in cuBLAS / cuDNN, and torch's math sdpa kernel (deterministic, any dtype)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)


def _build_engine(flag: bool, variant: dict, strategy: str, precision: str):
    from verl.workers.engine import EngineRegistry

    fp32 = precision == "fp32"
    torch.manual_seed(0)  # the LoRA init; the base weights come from the checkpoint
    model_config = HFModelConfig(
        path=MODEL_PATH,
        load_tokenizer=False,
        use_remove_padding=True,
        override_config={"attn_implementation": "sdpa"} if fp32 else {},
        **{"enable_gradient_checkpointing": False, **variant},
    )
    engine_config = FSDPEngineConfig(
        forward_only=False,
        strategy=strategy,
        fsdp_size=1,
        use_remove_padding=True,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=len(PROMPT_LENS),
        response_only_lm_head=flag,
        mixed_precision={"param_dtype": "fp32", "reduce_dtype": "fp32", "buffer_dtype": "fp32"} if fp32 else None,
    )
    engine = EngineRegistry.new(
        model_type="language_model",
        backend=strategy,
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=FSDPOptimizerConfig(optimizer="SGD", lr=LR, weight_decay=0.0, clip_grad=1e9),
        checkpoint_config=CheckpointConfig(),
    )
    engine.initialize()
    _force_fp32_math()
    assert engine.response_only_lm_head is flag, "the flag must survive build time on this layout"
    assert engine._autocast_dtype == (torch.float32 if fp32 else torch.bfloat16), "unexpected compute dtype"
    return engine


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _snapshot(module, kind: str) -> dict[str, torch.Tensor]:
    """fp32 copies of every gradient (``kind="grad"``) or trainable parameter (any other ``kind``)."""
    out = {}
    for name, param in module.named_parameters():
        tensor = param.grad if kind == "grad" else (param if param.requires_grad else None)
        if tensor is not None:
            out[f"{kind}/{name}"] = _local(tensor).detach().float().clone()
    return out


def _consumer_view(model_output, batch: TensorDict) -> dict[str, torch.Tensor]:
    """What the PPO loss and the metrics read: the response rows, padded to (bsz, max_response_len)."""
    keys = ("log_probs", "entropy", "sum_pi_squared")
    return {key: no_padding_2_padding(model_output[key], batch.to(model_output[key].device)).float() for key in keys}


def _run_arm(flag: bool, variant: dict, strategy: str, precision: str, batch: TensorDict, *, freeze: bool, label: str):
    """A forward-only pass, one PPO backward and one SGD step on a fresh engine; returns what to compare."""
    engine = _build_engine(flag, variant, strategy, precision)
    out: dict[str, torch.Tensor] = {}
    with engine.eval_mode():
        forward = engine.forward_backward_batch(batch, loss_function=None, forward_only=True)
    out.update({f"forward/{key}": value for key, value in _consumer_view(forward["model_output"], batch).items()})
    if freeze:  # the first flag-off log-probs are every arm's old_log_probs: ratio 1 in the PPO loss
        log_probs = out["forward/log_probs"].cpu()
        rows = [log_probs[i, :length] for i, length in enumerate(RESPONSE_LENS)]
        batch["old_log_probs"] = torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    actor_config = ActorConfig(
        strategy=strategy,
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=-1,
        entropy_coeff=0.01,
        calculate_entropy=True,
        calculate_sum_pi_squared=True,
    )
    with engine.train_mode(zero_grad_on_exit=False):
        engine.optimizer_zero_grad()
        train = engine.forward_backward_batch(batch, loss_function=partial(ppo_loss, config=actor_config))
    out.update({f"train/{key}": value for key, value in _consumer_view(train["model_output"], batch).items()})
    out["loss/ppo"] = torch.as_tensor(train["loss"], dtype=torch.float64)
    grads = _snapshot(engine.module, "grad")
    trainable = {name for name, param in engine.module.named_parameters() if param.requires_grad}
    assert {key.split("/", 1)[1] for key in grads} == trainable, "every trainable tensor, and nothing else, has a grad"
    if variant.get("lora_rank"):
        assert trainable and all("lora_" in name for name in trainable), "LoRA: only adapter tensors are trainable"
    print(
        f"[response_only_lm_head][{label}] flag={'on' if flag else 'off'}: gradients on {len(grads)} trainable tensors"
    )
    out.update(grads)
    out.update(_snapshot(engine.module, "before"))
    with engine.train_mode(zero_grad_on_exit=True):
        engine.optimizer_step()
    out.update(_snapshot(engine.module, "param"))
    out.update({f"update/{name}": out[f"param/{name}"] - out[f"before/{name}"] for name in trainable})
    residual = torch.zeros((), device=next(iter(grads.values())).device)
    for (
        name
    ) in trainable:  # SGD consistency within the arm: param_after == param_before - lr * grad up to fp32 rounding
        before, grad = out[f"before/{name}"], out[f"grad/{name}"]
        num = (out[f"update/{name}"] + LR * grad).norm()
        den = FP32_EPS * (before.norm() + (LR * grad).norm())
        residual = torch.maximum(residual, num / den.clamp_min(1e-30))
    out["sgd/residual_ulps"] = residual
    del engine
    get_torch_device().empty_cache()
    return out


def _layer_of(key: str) -> str:
    match = re.search(r"layers\.(\d+)\.", key)
    if match:
        return f"layer {int(match.group(1)):02d}"
    for name in ("embed", "lm_head", "norm"):
        if name in key:
            return name
    return "root"  # FSDP1's root flat param: the embedding, the final norm and the (tied) head


def _compare(label: str, enabled: dict, reference: dict, precision: str = "bf16", truth: dict | None = None):
    """Print every measured delta (the run log keeps them), then return the tolerance violations.

    ``truth`` (the fp32 flag-off arm) turns the bf16 gradient comparison into the distance-to-truth
    criterion of the module docstring; without it the bf16 gradients are reported only.
    """
    tol = TOLERANCES[precision]
    assert enabled.keys() == reference.keys()
    failures: list[str] = []
    layers: dict[
        str, list
    ] = {}  # layer -> [tensors, max on-vs-off rel_fro, sum D_off^2, sum D_on^2, sum |g32|^2, max ratio]
    summary: dict[
        str, list
    ] = {}  # group -> [max_abs, worst rel_fro, its key, sum |delta|^2, sum |ref|^2, min sign agr.]
    for key in sorted(reference):
        on, off = enabled[key], reference[key]
        delta = (on - off).abs()
        max_abs, rel_fro = delta.max().item(), (delta.norm() / off.norm().clamp_min(1e-30)).item()
        group = key.split("/", 1)[0]
        if group == "before":
            if not torch.equal(on, off):
                failures.append(f"{key}: weights differ before the step (max |delta| {max_abs:.3e})")
            continue
        if group == "sgd":
            print(f"[response_only_lm_head][{label}] SGD residual: off={off.item():.3f} on={on.item():.3f} ulp")
            for arm, value in (("off", off.item()), ("on", on.item())):
                if value > SGD_RESIDUAL_ULPS:
                    failures.append(f"flag {arm}: param_after != param_before - lr * grad ({value:.2f} ulp)")
            continue
        if group in ("forward", "train", "loss"):
            rtol, atol = tol["sum_pi_squared"] if key.endswith("sum_pi_squared") else tol["rows"]
            valid = (on != 0) | (off != 0)  # padded positions are zero in both arms
            rows = max(int(valid.sum().item()), 1)
            mean_delta = (on - off)[valid].mean().abs().item() if valid.any() else 0.0
            mean_tol = (atol + rtol * off[valid].abs().mean().item()) / rows**0.5 if valid.any() else atol
            bad = int((delta > atol + rtol * off.abs()).sum().item())
            print(
                f"[response_only_lm_head][{label}] {key}: max_abs={max_abs:.3e} mean_delta={mean_delta:.3e} rows={rows}"
            )
            if bad:
                failures.append(
                    f"{key}: {bad} elements beyond rtol {rtol:g} / atol {atol:g} (max |delta| {max_abs:.3e})"
                )
            if mean_delta > mean_tol:
                failures.append(f"{key}: batch mean differs by {mean_delta:.3e} > {mean_tol:.3e}")
            continue
        stats = summary.setdefault(group, [0.0, -1.0, "", 0.0, 0.0, 1.0])
        stats[0] = max(stats[0], max_abs)
        if rel_fro > stats[1]:
            stats[1], stats[2] = rel_fro, key
        stats[3] += delta.pow(2).sum().item()
        stats[4] += off.pow(2).sum().item()
        if group == "grad":
            row = layers.setdefault(_layer_of(key), [0, 0.0, 0.0, 0.0, 0.0, 0.0])
            row[0], row[1] = row[0] + 1, max(row[1], rel_fro)
            if precision == "fp32" and rel_fro > tol["grad"]:
                failures.append(f"{key}: fp32 arms differ by {rel_fro:.3e} relative Frobenius > {tol['grad']:g}")
            if truth is not None:
                g32 = truth[key]
                d_off, d_on, scale = (off - g32).norm().item(), (on - g32).norm().item(), g32.norm().item()
                ratio = d_on / (d_off + TRUTH_FLOOR * scale + 1e-30)
                row[2], row[3], row[4], row[5] = (
                    row[2] + d_off**2,
                    row[3] + d_on**2,
                    row[4] + scale**2,
                    max(row[5], ratio),
                )
                if d_on > TRUTH_RATIO * d_off + TRUTH_FLOOR * scale:
                    failures.append(
                        f"{key}: distance to the fp32 gradient {d_on:.3e} with the flag on vs {d_off:.3e} off "
                        f"(ratio {ratio:.2f} > {TRUTH_RATIO:g})"
                    )
        elif group == "update":  # -LR * grad: reported only (across arms it is the gradient again)
            nonzero = off != 0
            agree = (on.sign() == off.sign())[nonzero].float().mean().item() if nonzero.any() else 1.0
            stats[5] = min(stats[5], agree)
        elif group == "param":  # two opposite first-step updates (|update| <= LR each) plus 1e-2 relative
            if not bool((delta <= 2 * LR * 1.01 + 1e-2 * off.abs()).all()):
                failures.append(f"{key}: |delta| {max_abs:.3e} exceeds 2 * lr + 1e-2 * |ref|")
    for layer, (count, worst, d_off2, d_on2, scale2, ratio) in sorted(layers.items()):
        line = f"[response_only_lm_head][{label}] grad {layer}: n={count} max on-vs-off rel_fro={worst:.3e}"
        if truth is not None:
            d_off, d_on = (d_off2 / max(scale2, 1e-30)) ** 0.5, (d_on2 / max(scale2, 1e-30)) ** 0.5
            line += f" D_off={d_off:.3e} D_on={d_on:.3e} max ratio={ratio:.2f}"
        print(line)
    for group, (max_abs, worst, key, num, den, agree) in summary.items():
        pooled = (num / max(den, 1e-30)) ** 0.5
        print(
            f"[response_only_lm_head][{label}] {group}: pooled rel_fro={pooled:.3e} max_abs={max_abs:.3e} "
            f"worst {key} rel_fro={worst:.3e}" + (f" min sign agreement={agree:.4f}" if group == "update" else "")
        )
    return failures


def _check(label: str, enabled: dict, reference: dict, precision: str = "bf16", truth: dict | None = None) -> None:
    failures = _compare(label, enabled, reference, precision, truth)
    assert not failures, f"{label}: {len(failures)} check(s) failed:\n  " + "\n  ".join(failures[:30])


def _worker(rank: int, world_size: int, rendezvous_file: str, variant_name: str, strategy: str) -> None:
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_nccl_backend(), init_method=f"file://{rendezvous_file}", rank=rank, world_size=world_size
    )
    from transformers import AutoConfig

    variant, label = VARIANTS[variant_name], f"{strategy}/{variant_name}"
    batch = _make_batch(AutoConfig.from_pretrained(MODEL_PATH).vocab_size)
    run = partial(_run_arm, variant=variant, strategy=strategy, batch=batch)
    reference = run(False, precision="bf16", freeze=True, label=f"{label}/bf16")
    baseline = run(False, precision="bf16", freeze=False, label=f"{label}/bf16")
    for key in sorted(reference):  # determinism: a second flag-off engine reproduces the first bitwise
        max_abs = (baseline[key] - reference[key]).abs().max().item()
        assert torch.equal(baseline[key], reference[key]), f"{label}/bf16 flag-off twice: {key} max_abs={max_abs:.3e}"
    del baseline
    print(f"[response_only_lm_head][{label}/bf16] flag-off twice: bitwise identical ({len(reference)} tensors)")
    enabled = run(True, precision="bf16", freeze=False, label=f"{label}/bf16")
    truth = run(False, precision="fp32", freeze=False, label=f"{label}/fp32")
    cross = (truth["forward/log_probs"] - reference["forward/log_probs"])[reference["forward/log_probs"] != 0].abs()
    mean_nat, max_nat = cross.mean().item(), cross.max().item()
    print(f"[response_only_lm_head][{label}] fp32 vs bf16 flag-off log-probs: mean={mean_nat:.3e} max={max_nat:.3e}")
    failures = []
    if mean_nat > CROSS_PRECISION_MEAN_NAT:
        failures.append(f"fp32 reference is not the bf16 forward within bf16 noise: mean |delta| {mean_nat:.3e} nat")
    failures += _compare(f"{label}/bf16", enabled, reference, "bf16", truth=truth)
    del enabled, reference
    get_torch_device().empty_cache()
    enabled = run(True, precision="fp32", freeze=False, label=f"{label}/fp32")
    failures += _compare(f"{label}/fp32", enabled, truth, "fp32")
    assert not failures, f"{label}: {len(failures)} check(s) failed:\n  " + "\n  ".join(failures[:30])
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.skipif(
    not MODEL_PATH,
    reason=f"no model directory among {MODEL_CANDIDATES}; link it or set VERL_RESPONSE_ONLY_LM_HEAD_MODEL",
)
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_flag_on_matches_flag_off(variant, strategy, tmp_path):
    print(f"[response_only_lm_head] model {MODEL_PATH}, strategy {strategy}, variant {variant}")
    rendezvous_file = str(tmp_path / f"rdzv_{strategy}_{variant}")
    mp.spawn(fn=_worker, args=(1, rendezvous_file, variant, strategy), nprocs=1, join=True)
