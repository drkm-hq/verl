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
"""The worker-side guarantee of ``actor.skip_old_log_prob_when_on_policy``: once the actor is built,
``ActorRolloutRefWorker._check_actor_for_on_policy_skip`` walks the module
(``verl.utils.model.find_train_mode_dependent_modules``) and refuses the flag when a sub-module's forward
depends on train/eval mode, on every rank and before any update. The three configurations of the
independent review are caught on real tiny models, the LoRA one through the engine's real
``_build_lora_module`` loader; each flagged model has a measurable train/eval log-prob gap at identical
weights, the clean one none."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import MixtralConfig, MixtralForCausalLM, Qwen3Config, Qwen3ForCausalLM

from verl.utils.model import find_train_mode_dependent_modules
from verl.workers.config.model import HFModelConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine_workers import ActorRolloutRefWorker


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
    lora = LoraConfig(
        r=4, lora_alpha=4, lora_dropout=lora_dropout, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(Qwen3ForCausalLM(qwen3_config()), lora)
    for name, param in model.named_parameters():  # peft zero-inits lora_B: give the adapter an effect
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)
    model.save_pretrained(path)
    return str(path)


def _train_eval_gap(model):
    tokens = torch.arange(1, 7).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        old = model(tokens).logits.log_softmax(-1)
        model.train()
        new = model(tokens).logits.log_softmax(-1)
    return (old - new).abs().max().item()


def test_functional_attention_dropout_is_found():
    torch.manual_seed(0)
    dropout = Qwen3ForCausalLM(qwen3_config(attention_dropout=0.5))
    assert find_train_mode_dependent_modules(dropout) == [("model.layers.0.self_attn", "attention_dropout=0.5")]
    assert _train_eval_gap(dropout) > 1e-3
    clean = Qwen3ForCausalLM(qwen3_config())
    assert find_train_mode_dependent_modules(clean) == []
    assert _train_eval_gap(clean) == 0.0


def test_mixtral_router_jitter_is_found():
    torch.manual_seed(0)
    model = MixtralForCausalLM(mixtral_config(router_jitter_noise=0.1))
    assert find_train_mode_dependent_modules(model) == [("model.layers.0.mlp", "jitter_noise=0.1")]
    assert _train_eval_gap(model) > 1e-6
    assert find_train_mode_dependent_modules(MixtralForCausalLM(mixtral_config())) == []


def _lora_engine(**model_config):
    engine = FSDPEngineWithLMHead.__new__(FSDPEngineWithLMHead)
    engine.model_config = SimpleNamespace(use_shm=False, **model_config)
    return engine


def test_loaded_lora_dropout_is_found_through_the_real_loader(tmp_path):
    torch.manual_seed(0)
    engine = _lora_engine(lora_adapter_path=save_adapter(tmp_path / "adapter", 0.3))
    loaded = engine._build_lora_module(Qwen3ForCausalLM(qwen3_config()))
    assert loaded.peft_config["default"].lora_dropout == 0.3
    found = find_train_mode_dependent_modules(loaded)
    assert [path.split(".")[-3:] for path, _ in found] == [
        ["q_proj", "lora_dropout", "default"],
        ["v_proj", "lora_dropout", "default"],
    ]
    assert {reason for _, reason in found} == {"Dropout(p=0.3)"}
    assert _train_eval_gap(loaded) > 1e-3
    # an adapter without dropout loads an nn.Identity; verl's own LoraConfig never sets lora_dropout (peft's 0.0)
    engine = _lora_engine(lora_adapter_path=save_adapter(tmp_path / "zero", 0.0))
    assert find_train_mode_dependent_modules(engine._build_lora_module(Qwen3ForCausalLM(qwen3_config()))) == []
    engine = _lora_engine(
        lora_adapter_path=None, lora_rank=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
        target_parameters=None, exclude_modules=None,
    )  # fmt: skip
    own = engine._build_lora_module(Qwen3ForCausalLM(qwen3_config()))
    assert own.peft_config["default"].lora_dropout == 0.0
    assert find_train_mode_dependent_modules(own) == []


def test_the_module_built_from_the_effective_config_is_what_gets_walked(tmp_path):
    """``HFModelConfig`` applies override_config: dropout switched on by an override is found on the built
    module, dropout switched off by an override is gone (the effective config wins, in both directions)."""
    qwen3_config().save_pretrained(tmp_path / "clean")
    qwen3_config(attention_dropout=0.1).save_pretrained(tmp_path / "dropout")
    on = {"attention_dropout": 0.5, "attn_implementation": "eager"}
    switched_on = HFModelConfig(path=str(tmp_path / "clean"), load_tokenizer=False, override_config=on)
    assert find_train_mode_dependent_modules(Qwen3ForCausalLM(switched_on.hf_config)) == [
        ("model.layers.0.self_attn", "attention_dropout=0.5")
    ]
    off = {"attention_dropout": 0.0, "attn_implementation": "eager"}
    switched_off = HFModelConfig(path=str(tmp_path / "dropout"), load_tokenizer=False, override_config=off)
    assert find_train_mode_dependent_modules(Qwen3ForCausalLM(switched_off.hf_config)) == []


class _DropPath(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return x


def test_dropout_modules_drop_path_and_pipeline_chunks():
    model = nn.Sequential(
        nn.Dropout(0.0), nn.Dropout(0.1), nn.AlphaDropout(0.2), nn.Dropout2d(0.0), _DropPath(0.0), _DropPath(0.3),
        nn.Linear(2, 2),
    )  # fmt: skip
    assert find_train_mode_dependent_modules(model) == [
        ("1", "Dropout(p=0.1)"),
        ("2", "AlphaDropout(p=0.2)"),
        ("5", "_DropPath(p=0.3)"),
    ]
    chunks = [nn.Sequential(nn.Linear(2, 2)), nn.Sequential(nn.Dropout(0.5))]
    assert find_train_mode_dependent_modules(chunks) == [("[1].0", "Dropout(p=0.5)")]
    assert find_train_mode_dependent_modules(nn.Dropout(0.5)) == [("", "Dropout(p=0.5)")]


def _worker(module, flag=True):
    config = OmegaConf.create({"actor": {"skip_old_log_prob_when_on_policy": flag}})
    return SimpleNamespace(config=config, actor=SimpleNamespace(engine=SimpleNamespace(module=module)))


def test_worker_refuses_a_train_mode_dependent_actor_and_names_up_to_five_modules():
    check = ActorRolloutRefWorker._check_actor_for_on_policy_skip
    with pytest.raises(ValueError, match="model.layers.0.self_attn: attention_dropout=0.5"):
        check(_worker(Qwen3ForCausalLM(qwen3_config(attention_dropout=0.5))))
    check(_worker(Qwen3ForCausalLM(qwen3_config())))  # clean: accepted
    check(_worker(Qwen3ForCausalLM(qwen3_config(attention_dropout=0.5)), flag=False))  # flag off: no walk
    with pytest.raises(ValueError) as excinfo:
        check(_worker(nn.Sequential(*[nn.Dropout(0.1) for _ in range(7)])))
    message = str(excinfo.value)
    assert "7 train-mode-dependent sub-module(s)" in message and "4: Dropout(p=0.1)" in message
    assert "5: Dropout" not in message and message.endswith("; +2 more")
    with pytest.raises(ValueError, match="exposes no `module`"):
        check(_worker(None))
