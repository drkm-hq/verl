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
loss with ``entropy_coeff > 0`` (the entropy is in the loss), (e) every parameter after one AdamW
step. Variants: plain, gradient checkpointing, LoRA (PEFT, rank 8). A third engine with the
flag off is the determinism baseline: it must reproduce the first one bitwise (flash-attn runs
with ``FLASH_ATTENTION_DETERMINISTIC=1``), so every flag-on delta is the feature's.

Tolerances. The transformer trunk is untouched; what changes is the LM-head GEMM and the
log-softmax running over the K kept rows instead of all T packed rows: the same reduction over
the hidden dimension, possibly under a different kernel tiling, so bf16 logits may round
differently at the ulp level. The O(1) quantities (log-probs, entropy, ``sum_pi_squared``, the
loss) get ``rtol=1e-2, atol=1e-2``; gradients and updated parameters get a relative
Frobenius-norm difference below 1e-2 and an element-wise ``assert_close`` with ``rtol=1e-2``
and ``atol`` of 1e-2 of the tensor's largest magnitude (their absolute scale is far below 1).
The measured deltas are printed so the run log keeps them; the tolerances are to be tightened
to a few times the measured values once a run exists.

Run: ``pytest -s tests/workers/test_response_only_lm_head_on_gpu.py`` on one CUDA device with
``~/models/Qwen/Qwen2.5-0.5B`` (override the path with ``VERL_RESPONSE_ONLY_LM_HEAD_MODEL``).
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

MODEL_PATH = os.path.expanduser(os.environ.get("VERL_RESPONSE_ONLY_LM_HEAD_MODEL", "~/models/Qwen/Qwen2.5-0.5B"))
PROMPT_LENS = [1, 300, 77, 520, 33, 160]  # a one-token prompt is the shortest the guard allows
RESPONSE_LENS = [40, 128, 9, 64, 1, 256]
VARIANTS = {
    "plain": {},
    "gradient-checkpointing": {"enable_gradient_checkpointing": True},
    "lora": {"lora_rank": 8, "lora_alpha": 16},
}
O1_TOL = {"rtol": 1e-2, "atol": 1e-2}  # log-probs, entropy, sum_pi_squared, loss
REL_FRO_TOL = 1e-2  # gradients, updated parameters


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


def _build_engine(flag: bool, variant: dict):
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
        strategy="fsdp2",
        fsdp_size=1,
        use_remove_padding=True,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=len(PROMPT_LENS),
        response_only_lm_head=flag,
    )
    engine = EngineRegistry.new(
        model_type="language_model",
        backend="fsdp2",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=FSDPOptimizerConfig(lr=1e-3, lr_scheduler_type="constant"),
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


def _run_arm(flag: bool, variant: dict, batch: TensorDict, freeze_old_log_probs: bool) -> dict[str, torch.Tensor]:
    """A forward-only pass, one PPO backward and one AdamW step on a fresh engine; returns what to compare."""
    engine = _build_engine(flag, variant)
    out: dict[str, torch.Tensor] = {}
    with engine.eval_mode():
        forward = engine.forward_backward_batch(batch, loss_function=None, forward_only=True)
    out.update({f"forward/{key}": value for key, value in _consumer_view(forward["model_output"], batch).items()})
    if freeze_old_log_probs:  # the flag-off log-probs are every arm's old_log_probs: ratio 1 in the PPO loss
        log_probs = out["forward/log_probs"].cpu()
        rows = [log_probs[i, :length] for i, length in enumerate(RESPONSE_LENS)]
        batch["old_log_probs"] = torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    actor_config = ActorConfig(
        strategy="fsdp2",
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
    groups: dict[str, list] = {}  # group -> [max_abs, worst rel_fro, its key, sum |delta|^2, sum |reference|^2]
    for key in sorted(reference):
        on, off = enabled[key], reference[key]
        delta = (on - off).abs()
        max_abs, rel_fro = delta.max().item(), (delta.norm() / off.norm().clamp_min(1e-30)).item()
        group = key.split("/", 1)[0]
        if group in ("forward", "train", "loss"):
            max_rel = (delta / off.abs().clamp_min(1e-2)).max().item()
            print(f"[response_only_lm_head][{variant}] {key}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
            torch.testing.assert_close(on, off, **O1_TOL, msg=lambda m, k=key: f"{variant} {k}: {m}")
            continue
        stats = groups.setdefault(group, [0.0, -1.0, "", 0.0, 0.0])
        stats[0] = max(stats[0], max_abs)
        if rel_fro > stats[1]:
            stats[1], stats[2] = rel_fro, key
        stats[3] += delta.pow(2).sum().item()
        stats[4] += off.pow(2).sum().item()
        if group in ("grad", "param"):  # "update" is reported only: AdamW's first step is ~lr * sign(grad),
            # so a sign flip on a gradient element at the noise floor is a 2 * lr delta, not an identity failure
            assert rel_fro < REL_FRO_TOL, f"{variant} {key}: relative Frobenius difference {rel_fro:.3e}"
            atol = 1e-2 * max(off.abs().max().item(), 1e-30)
            torch.testing.assert_close(on, off, rtol=1e-2, atol=atol, msg=lambda m, k=key: f"{variant} {k}: {m}")
    for group, (max_abs, worst, key, num, den) in groups.items():
        pooled = (num / max(den, 1e-30)) ** 0.5
        print(
            f"[response_only_lm_head][{variant}] {group}: pooled rel_fro={pooled:.3e} max_abs={max_abs:.3e} "
            f"worst {key} rel_fro={worst:.3e}"
        )


def _worker(rank: int, world_size: int, rendezvous_file: str, variant_name: str) -> None:
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_nccl_backend(), init_method=f"file://{rendezvous_file}", rank=rank, world_size=world_size
    )
    from transformers import AutoConfig

    variant = VARIANTS[variant_name]
    batch = _make_batch(AutoConfig.from_pretrained(MODEL_PATH).vocab_size)
    reference = _run_arm(False, variant, batch, freeze_old_log_probs=True)
    baseline = _run_arm(False, variant, batch, freeze_old_log_probs=False)
    for key in sorted(reference):  # determinism: a second flag-off engine reproduces the first bitwise
        max_abs = (baseline[key] - reference[key]).abs().max().item()
        assert torch.equal(baseline[key], reference[key]), f"{variant_name} flag-off twice: {key} max_abs={max_abs:.3e}"
    del baseline
    print(f"[response_only_lm_head][{variant_name}] flag-off twice: bitwise identical ({len(reference)} tensors)")
    enabled = _run_arm(True, variant, batch, freeze_old_log_probs=False)
    _check(variant_name, enabled, reference)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.skipif(not os.path.isdir(MODEL_PATH), reason=f"no model at {MODEL_PATH}")
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_flag_on_matches_flag_off(variant, tmp_path):
    mp.spawn(fn=_worker, args=(1, str(tmp_path / f"rdzv_{variant}"), variant), nprocs=1, join=True)
