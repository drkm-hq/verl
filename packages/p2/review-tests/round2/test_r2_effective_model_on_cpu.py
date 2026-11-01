# Round-2 re-review tests for contrib/p2-oldlogprob-skip @ ba60cf8e (effective-model validation, module walk).
# Run from wt/p2 with PYTHONPATH=<worktree> and the CPU harness; see REVIEW-ROUND2-VERDICT.md.
"""Each test states the behaviour the reviewer expects of the fixed branch; a failing test is a finding.
The one test named ``documents_`` pins a known limit of the design instead (it passes)."""

import copy
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from peft import LoHaConfig, TaskType, get_peft_model
from torch import nn
from transformers import Qwen2VLConfig, Qwen3Config, Qwen3ForCausalLM
from transformers.models.phimoe import PhimoeConfig, PhimoeForCausalLM

from verl.utils.config import _hf_config_dropouts, _validate_skip_old_log_prob_config
from verl.utils.model import find_train_mode_dependent_modules
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, os.pardir))  # the repository root

_CONFIG_DIR = Path(os.path.join(_REPO, "wt/p2/verl/trainer/config"))
MODEL = "actor_rollout_ref.model"


@pytest.fixture(scope="module")
def on_policy_config():
    overrides = [
        "actor_rollout_ref.actor.skip_old_log_prob_when_on_policy=true", "data.train_batch_size=8",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8", "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.n=2", "algorithm.adv_estimator=grpo",
    ]  # fmt: skip
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        return compose(config_name="ppo_trainer", overrides=overrides)


def _with(config, **updates):
    config = copy.deepcopy(config)
    for key, value in updates.items():
        OmegaConf.update(config, key, value, force_add=True)
    return config


def qwen3_config(**kwargs):
    base = dict(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, head_dim=16, attn_implementation="eager", architectures=["Qwen3ForCausalLM"],
    )  # fmt: skip
    return Qwen3Config(**{**base, **kwargs})


def _gap(model, seed=0):
    tokens = torch.arange(1, 9).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        old = model(tokens).logits.log_softmax(-1)
        torch.manual_seed(seed)
        model.train()
        new = model(tokens).logits.log_softmax(-1)
    return (old - new).abs().max().item()


def _rejected(config, *fragments):
    with pytest.raises(ValueError, match="requires a strictly on-policy update") as excinfo:
        _validate_skip_old_log_prob_config(config)
    for fragment in fragments:
        assert fragment in str(excinfo.value)


def _save_loha_adapter(path, rank_dropout):
    """A peft LoHa adapter (lycoris family): its train-mode rate is ``rank_dropout``, stored by peft as a dict
    ``{adapter_name: rate}`` on the layer, not as an nn.Dropout and not under ``lora_dropout``."""
    torch.manual_seed(0)
    model = get_peft_model(
        Qwen3ForCausalLM(qwen3_config()),
        LoHaConfig(r=4, alpha=4, rank_dropout=rank_dropout, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM),
    )  # fmt: skip
    for name, param in model.named_parameters():
        if "hada_w" in name:
            torch.nn.init.normal_(param, std=0.05)
    model.save_pretrained(path)
    return str(path)


def test_loha_adapter_rank_dropout_is_rejected_by_the_validator(on_policy_config, tmp_path):
    """verl's ``_build_lora_module`` loads ANY peft adapter type through ``PeftModel.from_pretrained``; the
    validator reads only ``lora_dropout`` from adapter_config.json."""
    qwen3_config().save_pretrained(tmp_path / "base")
    adapter = _save_loha_adapter(tmp_path / "loha", 0.5)
    config = _with(on_policy_config, **{f"{MODEL}.path": str(tmp_path / "base"), f"{MODEL}.lora_rank": 4,
                                        f"{MODEL}.lora_adapter_path": adapter})  # fmt: skip
    assert _hf_config_dropouts(config.actor_rollout_ref.model) != []
    _rejected(config, "dropout")


def test_loha_adapter_rank_dropout_is_found_by_the_module_walk(tmp_path):
    engine = FSDPEngineWithLMHead.__new__(FSDPEngineWithLMHead)
    engine.model_config = SimpleNamespace(use_shm=False, lora_adapter_path=_save_loha_adapter(tmp_path / "loha", 0.5))
    torch.manual_seed(0)
    loaded = engine._build_lora_module(Qwen3ForCausalLM(qwen3_config()))
    assert loaded.base_model.model.model.layers[0].self_attn.q_proj.rank_dropout == {"default": 0.5}
    assert _gap(loaded) > 1e-5  # the real loader built a train-mode-dependent actor
    assert find_train_mode_dependent_modules(loaded) != []


class _HardCodedDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        return F.dropout(self.proj(x), 0.5, self.training)


def test_documents_that_code_level_train_mode_dependence_is_outside_both_layers():
    """A hard-coded ``F.dropout(p)`` (no module, no rate attribute) cannot be seen by a structural walk: the
    docs/PR body must present the walk as covering the named mechanisms, not as a guarantee."""
    torch.manual_seed(0)
    module = _HardCodedDropout()
    x = torch.randn(2, 4)
    module.eval()
    y_eval = module(x)
    module.train()
    assert not torch.equal(module(x), y_eval)
    assert find_train_mode_dependent_modules(module) == []


def test_batchnorm_with_running_stats_is_found():
    """BatchNorm uses batch statistics in train mode and running statistics in eval mode."""
    module = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    x = torch.randn(8, 4)
    module.train()
    y_train = module(x)
    module.eval()
    assert not torch.allclose(module(x), y_train)
    assert find_train_mode_dependent_modules(module) != []


def test_hf_config_path_and_nested_override_are_the_effective_config(tmp_path):
    qwen3_config().save_pretrained(tmp_path / "clean")
    qwen3_config(attention_dropout=0.1).save_pretrained(tmp_path / "dropout")
    both = {"path": str(tmp_path / "clean"), "hf_config_path": str(tmp_path / "dropout")}
    assert _hf_config_dropouts(OmegaConf.create(both)) == [("attention_dropout", 0.1)]
    Qwen2VLConfig(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2),
        vision_config=dict(depth=1, embed_dim=32, hidden_size=32, num_heads=2, in_channels=3, patch_size=14, spatial_merge_size=2, temporal_patch_size=2),
    ).save_pretrained(tmp_path / "vl")  # fmt: skip
    nested = {"path": str(tmp_path / "vl"), "override_config": {"text_config": {"attention_dropout": 0.5}}}
    assert _hf_config_dropouts(OmegaConf.create({"path": str(tmp_path / "vl")})) == []
    assert _hf_config_dropouts(OmegaConf.create(nested)) == [("attention_dropout", 0.5)]


def test_walk_sees_through_fsdp2_and_checkpoint_wrapper():
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    try:
        model = Qwen3ForCausalLM(qwen3_config(attention_dropout=0.5, num_hidden_layers=2))
        model.model.layers[0] = checkpoint_wrapper(model.model.layers[0])
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)
        assert [reason for _, reason in find_train_mode_dependent_modules(model)] == ["attention_dropout=0.5"] * 2
    finally:
        dist.destroy_process_group()


def test_phimoe_sparsemixer_is_exact_at_zero_jitter_and_rejected_above(on_policy_config, tmp_path):
    """PhiMoE's router samples experts (Gumbel) in train mode; with both jitter rates at 0 the mask admits
    only the arg-max, so train == eval; any positive jitter is a config key the scan names."""
    kw = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
              num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2, attention_dropout=0.0,
              attn_implementation="eager", architectures=["PhimoeForCausalLM"])  # fmt: skip
    PhimoeConfig(router_jitter_noise=0.0, input_jitter_noise=0.0, **kw).save_pretrained(tmp_path / "quiet")
    _validate_skip_old_log_prob_config(_with(on_policy_config, **{f"{MODEL}.path": str(tmp_path / "quiet")}))
    torch.manual_seed(0)
    model = PhimoeForCausalLM(PhimoeConfig(router_jitter_noise=0.0, input_jitter_noise=0.0, **kw)).float()
    assert find_train_mode_dependent_modules(model) == [] and max(_gap(model, s) for s in range(3)) == 0.0
    PhimoeConfig(router_jitter_noise=0.01, input_jitter_noise=0.0, **kw).save_pretrained(tmp_path / "jitter")
    _rejected(_with(on_policy_config, **{f"{MODEL}.path": str(tmp_path / "jitter")}), "router_jitter_noise=0.01")
