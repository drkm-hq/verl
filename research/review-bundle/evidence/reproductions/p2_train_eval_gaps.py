"""Read-only P2 review: the exactness validator accepts train/eval-dependent actors.

Run with the existing CPU environment:
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=<repo>/wt/p2 \
  TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 \
  <repo>/.venv-cpu/bin/python -B \
  work/p2-independent/reproduce_train_eval_gaps.py

All generated checkpoint/config files live beside this script. Nothing in wt/p2 is changed.
"""

from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from peft import LoraConfig, TaskType, get_peft_model
from transformers import MixtralConfig, MixtralForCausalLM, Qwen3Config, Qwen3ForCausalLM

from verl.utils.config import _hf_config_dropouts, _validate_skip_old_log_prob_config
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = "<repo>/wt/p2/verl/trainer/config"


def on_policy_config(model_path):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.skip_old_log_prob_when_on_policy=true",
                "data.train_batch_size=8",
                "actor_rollout_ref.actor.ppo_mini_batch_size=8",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
                "actor_rollout_ref.rollout.n=2",
                "algorithm.adv_estimator=grpo",
            ],
        )
    cfg.actor_rollout_ref.model.path = str(model_path)
    return cfg


def accepted(cfg):
    dropouts = _hf_config_dropouts(cfg.actor_rollout_ref.model)
    assert dropouts == [], dropouts
    _validate_skip_old_log_prob_config(cfg)
    print("detected dropout=[]; full skip validator ACCEPTED")


def train_eval_difference(model):
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        model.eval()
        old = model(tokens).logits.log_softmax(-1)
        model.train()
        new = model(tokens).logits.log_softmax(-1)
    difference = (old - new).abs().max().item()
    assert difference > 1e-5, difference
    print(f"same weights, max |eval logprob - train logprob| = {difference}")


def reproduce_router_jitter():
    path = ROOT / "mixtral-jitter"
    config = MixtralConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=2,
        router_jitter_noise=0.1,
        attention_dropout=0.0,
        attn_implementation="eager",
    )
    config.save_pretrained(path)
    print("\nCase: stored Mixtral router_jitter_noise=0.1")
    accepted(on_policy_config(path))
    torch.manual_seed(0)
    train_eval_difference(MixtralForCausalLM(config).float())


def reproduce_loaded_lora_dropout():
    path = ROOT / "lora"
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attention_dropout=0.0,
        attn_implementation="eager",
    )
    config.save_pretrained(path / "base")
    torch.manual_seed(0)
    model = get_peft_model(
        Qwen3ForCausalLM(config),
        LoraConfig(
            r=4,
            lora_alpha=4,
            lora_dropout=0.3,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)
    model.save_pretrained(path / "adapter")

    cfg = on_policy_config(path / "base")
    cfg.actor_rollout_ref.model.lora_rank = 4
    cfg.actor_rollout_ref.model.lora_adapter_path = str(path / "adapter")
    print("\nCase: dropout-free Qwen3 with loaded LoRA adapter lora_dropout=0.3")
    accepted(cfg)
    engine = FSDPEngineWithLMHead.__new__(FSDPEngineWithLMHead)
    engine.model_config = SimpleNamespace(lora_adapter_path=str(path / "adapter"), use_shm=False)
    torch.manual_seed(0)
    # The actual engine loader, not a mock of adapter application.
    loaded = engine._build_lora_module(Qwen3ForCausalLM(config))
    assert loaded.peft_config["default"].lora_dropout == 0.3
    train_eval_difference(loaded)


if __name__ == "__main__":
    reproduce_router_jitter()
    reproduce_loaded_lora_dropout()
