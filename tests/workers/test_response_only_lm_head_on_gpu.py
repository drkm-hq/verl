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
checkpoint and seed with ``fsdp_config.response_only_lm_head`` off and on, in the default bf16
mixed precision (fp32 master weights, bf16 compute). Compared after an fp32 upcast:

(a) response log-probs, (b) entropy and (c) ``sum_pi_squared`` from a forward-only pass and from
the training forward, (d) every parameter gradient after one backward of the real PPO policy
loss with ``entropy_coeff > 0`` (the entropy is in the loss), (e) every parameter after one
optimizer step. Variants: plain, gradient checkpointing, LoRA (PEFT, rank 8), each on ``fsdp``
(FSDP1, the default strategy) and ``fsdp2``. A third engine with the flag off is the determinism
baseline: it must reproduce the first one bitwise (flash-attn runs with
``FLASH_ATTENTION_DETERMINISTIC=1``), so every flag-on delta is the feature's.

Tolerances. The transformer trunk is untouched; what changes is the LM-head GEMM and the
log-softmax running over the K kept rows instead of all T packed rows (the same reduction over
the hidden dimension under a possibly different kernel tiling) and the head's backward GEMMs (a
different split of the reduction over the vocabulary), so bf16 logits and the head's input
gradient may round differently by one ulp. One ulp of a top logit (0.125 at logit 20) moves a
row's entropy by ~0.03 and its Σπ² by ~0.02, hence per response row ``rtol=2e-2, atol=5e-2`` on
log-probs, entropy and Σπ² (the labels are random tokens, so the rtol term also covers a one-ulp
flip of the label's own logit), while the batch means must agree within that row tolerance
divided by sqrt(rows): full-row deltas on at most sqrt(rows) rows. The PPO loss, a mean, gets
``rtol=1e-2, atol=5e-3``. Gradients and the parameter update get a relative Frobenius-norm
difference below 1e-2 and an element-wise ``assert_close`` with ``rtol=1e-2`` and ``atol`` of
1e-2 of the tensor's largest magnitude; the update also needs 99 % sign agreement over its
non-zero elements. The step uses plain SGD (lr 1e-3, no momentum, no weight decay, clipping
disabled) so the update is exactly ``-lr * grad`` and the gradient tolerance carries over;
AdamW's first step is ~``lr * sign(grad)``, which turns a sign flip on a gradient element at
the noise floor into a ``2 * lr`` delta that no identity tolerance can judge. The parameters
themselves are bounded by two such first-step updates plus 1e-2 relative. verl's GPU job runs
pytest with ``-x``, so the tolerances stay conservative; the measured deltas are printed for the
follow-up that tightens them.

Run: ``pytest -s -rs tests/workers/test_response_only_lm_head_on_gpu.py`` on one CUDA device with
Qwen2.5-0.5B at ``~/models/Qwen/Qwen2.5-0.5B`` or ``/root/models/Qwen/Qwen2.5-0.5B`` (or set
``VERL_RESPONSE_ONLY_LM_HEAD_MODEL``). Without a device or a model the six cases skip and the
skip reason names the paths tried.
"""

import os

os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")  # flash-attn backward without atomics
os.environ.setdefault("NCCL_DEBUG", "WARN")

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
REL_FRO_TOL = 1e-2  # gradients and updates


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


def _build_engine(flag: bool, variant: dict, strategy: str):
    from verl.workers.engine import EngineRegistry

    torch.manual_seed(0)  # the LoRA init; the base weights come from the checkpoint
    model_config = HFModelConfig(
        path=MODEL_PATH,
        load_tokenizer=False,
        use_remove_padding=True,
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
    assert engine.response_only_lm_head is flag, "the flag must survive build time on this layout"
    return engine


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _snapshot(module, kind: str) -> dict[str, torch.Tensor]:
    """fp32 copies of every gradient (``kind="grad"``) or trainable parameter (``kind="param"``)."""
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


def _run_arm(
    flag: bool, variant: dict, strategy: str, batch: TensorDict, freeze_old_log_probs: bool
) -> dict[str, torch.Tensor]:
    """A forward-only pass, one PPO backward and one AdamW step on a fresh engine; returns what to compare."""
    engine = _build_engine(flag, variant, strategy)
    out: dict[str, torch.Tensor] = {}
    with engine.eval_mode():
        forward = engine.forward_backward_batch(batch, loss_function=None, forward_only=True)
    out.update({f"forward/{key}": value for key, value in _consumer_view(forward["model_output"], batch).items()})
    if freeze_old_log_probs:  # the flag-off log-probs are every arm's old_log_probs: ratio 1 in the PPO loss
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
    out.update(_snapshot(engine.module, "grad"))
    before = _snapshot(engine.module, "param")
    with engine.train_mode(zero_grad_on_exit=True):
        engine.optimizer_step()
    out.update(_snapshot(engine.module, "param"))
    out.update({"update/" + key.split("/", 1)[1]: out[key] - before[key] for key in before})
    del engine, before
    get_torch_device().empty_cache()
    return out


def _check(variant: str, enabled: dict, reference: dict) -> None:
    """Print the measured deltas (the run log keeps them) and assert the tolerances of the module docstring."""
    assert enabled.keys() == reference.keys()
    groups: dict[
        str, list
    ] = {}  # group -> [max_abs, worst rel_fro, its key, sum |delta|^2, sum |ref|^2, min sign agreement]
    for key in sorted(reference):
        on, off = enabled[key], reference[key]
        delta = (on - off).abs()
        max_abs, rel_fro = delta.max().item(), (delta.norm() / off.norm().clamp_min(1e-30)).item()
        group, tag = key.split("/", 1)[0], f"{variant} {key}"
        if group in ("forward", "train", "loss"):
            valid = (on != 0) | (off != 0)  # padded positions are zero in both arms
            rows = max(int(valid.sum().item()), 1)
            mean_delta = (on - off)[valid].mean().abs().item() if valid.any() else 0.0
            rtol, atol = (1e-2, 5e-3) if group == "loss" else (2e-2, 5e-2)
            mean_tol = (atol + rtol * off[valid].abs().mean().item()) / rows**0.5 if valid.any() else atol
            print(f"[response_only_lm_head][{tag}]: max_abs={max_abs:.3e} mean_delta={mean_delta:.3e} rows={rows}")
            torch.testing.assert_close(on, off, rtol=rtol, atol=atol, msg=lambda m, t=tag: f"{t}: {m}")
            assert mean_delta <= mean_tol, f"{tag}: batch mean differs by {mean_delta:.3e} > {mean_tol:.3e}"
            continue
        stats = groups.setdefault(group, [0.0, -1.0, "", 0.0, 0.0, 1.0])
        stats[0] = max(stats[0], max_abs)
        if rel_fro > stats[1]:
            stats[1], stats[2] = rel_fro, key
        stats[3] += delta.pow(2).sum().item()
        stats[4] += off.pow(2).sum().item()
        if group in ("grad", "update"):  # the update is -LR * grad: the gradient tolerance carries over
            assert rel_fro < REL_FRO_TOL, f"{tag}: relative Frobenius difference {rel_fro:.3e}"
            atol = 1e-2 * max(off.abs().max().item(), 1e-30)
            torch.testing.assert_close(on, off, rtol=1e-2, atol=atol, msg=lambda m, t=tag: f"{t}: {m}")
            if group == "update":
                nonzero = off != 0
                agree = (on.sign() == off.sign())[nonzero].float().mean().item() if nonzero.any() else 1.0
                stats[5] = min(stats[5], agree)
                assert agree >= 0.99, f"{tag}: update sign agreement {agree:.4f} < 0.99"
        elif group == "param":  # two opposite first-step updates (|update| <= LR each) plus 1e-2 relative
            bound = 2 * LR * 1.01 + 1e-2 * off.abs()
            assert bool((delta <= bound).all()), f"{tag}: |delta| {max_abs:.3e} exceeds 2 * lr + 1e-2 * |ref|"
    for group, (max_abs, worst, key, num, den, agree) in groups.items():
        pooled = (num / max(den, 1e-30)) ** 0.5
        print(
            f"[response_only_lm_head][{variant}] {group}: pooled rel_fro={pooled:.3e} max_abs={max_abs:.3e} "
            f"worst {key} rel_fro={worst:.3e}" + (f" min sign agreement={agree:.4f}" if group == "update" else "")
        )


def _worker(rank: int, world_size: int, rendezvous_file: str, variant_name: str, strategy: str) -> None:
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_nccl_backend(), init_method=f"file://{rendezvous_file}", rank=rank, world_size=world_size
    )
    from transformers import AutoConfig

    variant, label = VARIANTS[variant_name], f"{strategy}/{variant_name}"
    batch = _make_batch(AutoConfig.from_pretrained(MODEL_PATH).vocab_size)
    reference = _run_arm(False, variant, strategy, batch, freeze_old_log_probs=True)
    baseline = _run_arm(False, variant, strategy, batch, freeze_old_log_probs=False)
    for key in sorted(reference):  # determinism: a second flag-off engine reproduces the first bitwise
        max_abs = (baseline[key] - reference[key]).abs().max().item()
        assert torch.equal(baseline[key], reference[key]), f"{label} flag-off twice: {key} max_abs={max_abs:.3e}"
    del baseline
    print(f"[response_only_lm_head][{label}] flag-off twice: bitwise identical ({len(reference)} tensors)")
    enabled = _run_arm(True, variant, strategy, batch, freeze_old_log_probs=False)
    _check(label, enabled, reference)
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
