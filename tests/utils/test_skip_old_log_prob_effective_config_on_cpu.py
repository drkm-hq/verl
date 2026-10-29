# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""``actor.skip_old_log_prob_when_on_policy`` validates the EFFECTIVE actor model, not the saved base HF
config: ``override_config`` is applied first, every train-mode-only setting counts (dropout, MoE router
jitter, layer drop, drop path) and a loaded LoRA adapter's ``lora_dropout`` is read from its
``adapter_config.json``. The three configurations of the independent review (override dropout, saved
Mixtral router jitter, nonzero adapter dropout) are rejected by name; an override that disables a saved
dropout is accepted (the effective config wins)."""

import copy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import BartConfig, Dinov2Config, MixtralConfig, Qwen3Config, Qwen3ForCausalLM

from verl.utils.config import _hf_config_dropouts, _validate_skip_old_log_prob_config

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "verl" / "trainer" / "config"
MODEL = "actor_rollout_ref.model"


@pytest.fixture(scope="module")
def on_policy_config():
    overrides = [
        "actor_rollout_ref.actor.skip_old_log_prob_when_on_policy=true",
        "data.train_batch_size=8",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.n=2",
        "algorithm.adv_estimator=grpo",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        return compose(config_name="ppo_trainer", overrides=overrides)


def _with(config, **updates):
    config = copy.deepcopy(config)
    for key, value in updates.items():
        OmegaConf.update(config, key, value, force_add=True)
    return config


def qwen3_config(**kwargs):
    return Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, head_dim=16, attn_implementation="eager", architectures=["Qwen3ForCausalLM"], **kwargs,
    )  # fmt: skip


def mixtral_config(**kwargs):
    return MixtralConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, num_local_experts=2, num_experts_per_tok=2, attention_dropout=0.0,
        attn_implementation="eager", architectures=["MixtralForCausalLM"], **kwargs,
    )  # fmt: skip


def save_adapter(path, lora_dropout):
    """A LoRA adapter on the dropout-free tiny Qwen3, written with peft as the review's probe does."""
    lora = LoraConfig(
        r=4, lora_alpha=4, lora_dropout=lora_dropout, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(Qwen3ForCausalLM(qwen3_config()), lora)
    for name, param in model.named_parameters():  # peft zero-inits lora_B: give the adapter an effect
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)
    model.save_pretrained(path)
    return str(path)


def _rejected(config, *fragments):
    with pytest.raises(ValueError, match="requires a strictly on-policy update") as excinfo:
        _validate_skip_old_log_prob_config(config)
    for fragment in fragments:
        assert fragment in str(excinfo.value)


@pytest.mark.parametrize("key", ["override_config.attention_dropout", "override_config.model_config.attention_dropout"])
def test_dropout_enabled_through_override_config_is_rejected(on_policy_config, tmp_path, key):
    """Review case 1: a dropout-free saved Qwen3 with attention dropout enabled by override_config (the plain
    key and the Megatron-style ``model_config`` sub-key both reach ``HFModelConfig.__post_init__``)."""
    qwen3_config().save_pretrained(tmp_path / "clean")
    config = _with(on_policy_config, **{f"{MODEL}.path": str(tmp_path / "clean"), f"{MODEL}.{key}": 0.5})
    assert _hf_config_dropouts(config.actor_rollout_ref.model) == [("attention_dropout", 0.5)]
    _rejected(config, "attention_dropout=0.5", "after override_config")


def test_saved_router_jitter_is_rejected(on_policy_config, tmp_path):
    """Review case 3: Mixtral's ``router_jitter_noise`` multiplies the hidden states by uniform noise when
    ``self.training``; it is not a dropout key."""
    mixtral_config(router_jitter_noise=0.1).save_pretrained(tmp_path / "mixtral")
    config = _with(on_policy_config, **{f"{MODEL}.path": str(tmp_path / "mixtral")})
    assert _hf_config_dropouts(config.actor_rollout_ref.model) == [("router_jitter_noise", 0.1)]
    _rejected(config, "router_jitter_noise=0.1")
    mixtral_config(router_jitter_noise=0.0).save_pretrained(tmp_path / "quiet")
    _validate_skip_old_log_prob_config(_with(on_policy_config, **{f"{MODEL}.path": str(tmp_path / "quiet")}))


def test_loaded_lora_adapter_dropout_is_rejected(on_policy_config, tmp_path):
    """Review case 2: a dropout-free base with ``lora_rank > 0`` and a ``lora_adapter_path`` whose
    adapter_config.json has ``lora_dropout=0.3`` (peft builds an nn.Dropout for it)."""
    qwen3_config().save_pretrained(tmp_path / "base")
    base = {f"{MODEL}.path": str(tmp_path / "base"), f"{MODEL}.lora_rank": 4}
    adapter = save_adapter(tmp_path / "adapter", 0.3)
    config = _with(on_policy_config, **base, **{f"{MODEL}.lora_adapter_path": adapter})
    assert _hf_config_dropouts(config.actor_rollout_ref.model) == [("lora_dropout", 0.3)]
    _rejected(config, "lora_dropout=0.3", "LoRA adapter")
    # accepted: an adapter without dropout, and verl's own LoraConfig (no adapter path: peft's default 0.0)
    zero = _with(on_policy_config, **base, **{f"{MODEL}.lora_adapter_path": save_adapter(tmp_path / "zero", 0.0)})
    assert _hf_config_dropouts(zero.actor_rollout_ref.model) == []
    _validate_skip_old_log_prob_config(zero)
    _validate_skip_old_log_prob_config(_with(on_policy_config, **base))
    # the adapter is only read when LoRA is on
    off = _with(on_policy_config, **{f"{MODEL}.path": str(tmp_path / "base"), f"{MODEL}.lora_adapter_path": adapter})
    assert _hf_config_dropouts(off.actor_rollout_ref.model) == []


def test_override_that_disables_a_saved_dropout_is_accepted(on_policy_config, tmp_path):
    """The effective config wins in both directions: a saved ``attention_dropout=0.1`` overridden to 0 is clean."""
    qwen3_config(attention_dropout=0.1).save_pretrained(tmp_path / "dropout")
    saved = _with(on_policy_config, **{f"{MODEL}.path": str(tmp_path / "dropout")})
    _rejected(saved, "attention_dropout=0.1")
    disabled = _with(saved, **{f"{MODEL}.override_config.attention_dropout": 0.0})
    assert _hf_config_dropouts(disabled.actor_rollout_ref.model) == []
    _validate_skip_old_log_prob_config(disabled)


def test_unreadable_adapter_or_config_warns_that_the_worker_enforces_the_rule(on_policy_config, tmp_path, capsys):
    qwen3_config().save_pretrained(tmp_path / "base")
    missing = {f"{MODEL}.lora_rank": 4, f"{MODEL}.lora_adapter_path": str(tmp_path / "missing-adapter")}
    config = _with(on_policy_config, **{f"{MODEL}.path": str(tmp_path / "base")}, **missing)
    assert _hf_config_dropouts(config.actor_rollout_ref.model) is None
    _validate_skip_old_log_prob_config(config)
    out = capsys.readouterr().out
    assert "could not load the model's HF config" in out and "enforced on the built actor module" in out


def test_layerdrop_and_drop_path_keys_are_detected(tmp_path):
    """The other train-mode-only keys transformers 5.16 configs declare: layer drop (BART-style encoders and
    decoders) and stochastic depth (``drop_path_rate`` on the vision towers)."""
    BartConfig(
        encoder_layerdrop=0.1, decoder_layerdrop=0.2, d_model=8, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=1, decoder_attention_heads=1, encoder_ffn_dim=8, decoder_ffn_dim=8, vocab_size=16,
    ).save_pretrained(tmp_path / "bart")  # fmt: skip
    found = _hf_config_dropouts(OmegaConf.create({"path": str(tmp_path / "bart")}))
    assert ("encoder_layerdrop", 0.1) in found and ("decoder_layerdrop", 0.2) in found
    Dinov2Config(drop_path_rate=0.2, hidden_size=8, num_hidden_layers=1, num_attention_heads=1).save_pretrained(
        tmp_path / "dinov2"
    )
    assert _hf_config_dropouts(OmegaConf.create({"path": str(tmp_path / "dinov2")})) == [("drop_path_rate", 0.2)]


def save_loha_adapter(path, rank_dropout):
    """A peft LoHa adapter (lycoris family): verl's loader accepts any peft_type, and peft keeps its train-mode
    rate ``rank_dropout`` as a dict on the layer, not as an nn.Dropout (round-2 re-review D1)."""
    from peft import LoHaConfig

    torch.manual_seed(0)
    loha = LoHaConfig(
        r=4, alpha=4, rank_dropout=rank_dropout, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(Qwen3ForCausalLM(qwen3_config()), loha)
    for name, param in model.named_parameters():
        if "hada_w" in name:
            torch.nn.init.normal_(param, std=0.05)
    model.save_pretrained(path)
    return str(path)


def test_non_lora_peft_adapter_is_rejected_by_type_and_rate(on_policy_config, tmp_path):
    qwen3_config().save_pretrained(tmp_path / "base")
    base = {f"{MODEL}.path": str(tmp_path / "base"), f"{MODEL}.lora_rank": 4}
    loha = save_loha_adapter(tmp_path / "loha", 0.5)
    config = _with(on_policy_config, **base, **{f"{MODEL}.lora_adapter_path": loha})
    assert _hf_config_dropouts(config.actor_rollout_ref.model) == [("peft_type", "LOHA"), ("rank_dropout", 0.5)]
    _rejected(config, "peft_type=LOHA", "rank_dropout=0.5", "only a peft_type=LORA adapter")
    # without rank dropout the type alone rejects it: LoHa / LoKr keep other train-mode rates, are not LoRA
    quiet = _with(
        on_policy_config, **base, **{f"{MODEL}.lora_adapter_path": save_loha_adapter(tmp_path / "loha0", 0.0)}
    )
    assert _hf_config_dropouts(quiet.actor_rollout_ref.model) == [("peft_type", "LOHA")]
    _rejected(quiet, "peft_type=LOHA")
