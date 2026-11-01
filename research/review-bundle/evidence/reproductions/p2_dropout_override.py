"""Independent review reproducer; does not modify the contribution checkout."""
import tempfile

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl.utils.config import _hf_config_dropouts, _validate_skip_old_log_prob_config
from verl.workers.config.model import HFModelConfig

torch.manual_seed(2026)
torch.set_num_threads(1)
with tempfile.TemporaryDirectory() as path:
    Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        head_dim=16, attention_dropout=0.0, architectures=["Qwen3ForCausalLM"],
    ).save_pretrained(path)
    with initialize_config_dir(version_base=None, config_dir="<repo>/wt/p2/verl/trainer/config"):
        cfg = compose(config_name="ppo_trainer", overrides=[
            "actor_rollout_ref.actor.skip_old_log_prob_when_on_policy=true",
            "data.train_batch_size=8", "actor_rollout_ref.actor.ppo_mini_batch_size=8",
            "algorithm.adv_estimator=grpo",
        ])
    cfg.actor_rollout_ref.model.path = path
    OmegaConf.update(cfg, "actor_rollout_ref.model.override_config.attention_dropout", 0.5, force_add=True)
    OmegaConf.update(cfg, "actor_rollout_ref.model.override_config.attn_implementation", "eager", force_add=True)
    print("Validator sees dropout:", _hf_config_dropouts(cfg.actor_rollout_ref.model))
    _validate_skip_old_log_prob_config(cfg)
    print("Skip validator: ACCEPTED")
    actual = HFModelConfig(path=path, load_tokenizer=False, override_config={
        "attention_dropout": 0.5, "attn_implementation": "eager",
    })
    print("Actual HFModelConfig attention_dropout:", actual.hf_config.attention_dropout)
    model = Qwen3ForCausalLM(actual.hf_config).float()
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        model.eval()
        old = model(x).logits.log_softmax(-1)
        model.train()
        new = model(x).logits.log_softmax(-1)
    print("Train vs eval max abs log-prob difference:", (new-old).abs().max().item())
    assert actual.hf_config.attention_dropout == 0.5
    assert not torch.equal(old, new)
