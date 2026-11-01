# Adversarial review tests, round 2: target the fixes in a9fafde8 / 506d2df1 / 4602a7af.
"""Each test states the behaviour the reviewer expects of the FIXED branch; a failure is a residual finding."""

import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils import tensordict_utils as tu
from verl.utils.config import _hf_config_dropouts, _validate_skip_old_log_prob_config
from verl.workers.engine_workers import TrainingWorker
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))  # the repository root

_CONFIG_DIR = Path(os.path.join(_REPO, "wt/p2")) / "verl" / "trainer" / "config"
BASE = [
    "actor_rollout_ref.actor.skip_old_log_prob_when_on_policy=true",
    "data.train_batch_size=16",
    "actor_rollout_ref.actor.ppo_mini_batch_size=16",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.rollout.n=8",
    "algorithm.adv_estimator=grpo",
]


def _compose(*extra):
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        return compose(config_name="ppo_trainer", overrides=[*BASE, *extra])


def _with(config, **updates):
    config = copy.deepcopy(config)
    for key, value in updates.items():
        OmegaConf.update(config, key, value, force_add=True)
    return config


# ---------------------------------------------------------------- D1: trainer check vs worker assert
class _Stub(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def _trainer(ppo_mini, n):
    t = _Stub.__new__(_Stub)
    t.global_steps = 1
    t.config = OmegaConf.create(
        {
            "trainer": {"critic_warmup": 0},
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": ppo_mini, "skip_old_log_prob_when_on_policy": True},
                "rollout": {"n": n},
            },
        }
    )
    t.actor_rollout_wg = SimpleNamespace(compute_log_prob=lambda b: pytest.fail("forward ran"))
    return t


def _batch(sampled, padding):
    keys = [f"s{i}" for i in range(sampled)] + [f"p{i}" for i in range(padding)]
    tags = [{} for _ in range(sampled)] + [{"is_padding": True} for _ in range(padding)]
    return KVBatchMeta(keys=keys, tags=tags, partition_id="train")


class _ReachedLoop(Exception):
    pass


def _worker_accepts(rows_per_rank, mini_per_rank):
    """train_mini_batch's new assert, isolated: True if the worker would proceed to the iterator."""
    engine = SimpleNamespace(
        get_data_parallel_size=lambda: 1,
        get_data_parallel_rank=lambda: 0,
        train_mode=lambda **k: (_ for _ in ()).throw(_ReachedLoop()),
    )
    worker = SimpleNamespace(engine=engine, engine_config=SimpleNamespace(), profiler=None)
    data = TensorDict({"row": torch.arange(rows_per_rank)}, batch_size=[rows_per_rank])
    tu.assign_non_tensor(data, mini_batch_size=mini_per_rank, epochs=1, on_policy_skip_old_log_prob=True)
    try:
        TrainingWorker.train_mini_batch(worker, data)
    except AssertionError as e:
        assert "exactly one mini-batch" in str(e)
        return False
    except _ReachedLoop:  # the assert passed and the mini-batch loop was reached
        return True
    pytest.fail("unreachable")


@pytest.mark.parametrize(
    ("sampled", "padding", "expect"),
    [
        (128, 0, True),  # exact
        (120, 8, True),  # padding rows inside ONE mini-batch: one optimizer step, exact
        (128, 128, False),  # sampled rows == mini rows but padding doubles the batch (critic lcm case)
        (129, 127, False),  # one extra trajectory, padded to 2x
        (64, 0, False),  # fewer rows than one mini-batch (no padding, e.g. a custom sampler)
    ],
)
def test_trainer_check_and_worker_assert_count_the_same_thing(sampled, padding, expect):
    train_batch_size, n, dp = 16, 8, 2
    trainer = _trainer(train_batch_size, n)
    batch = _batch(sampled, padding)
    total = sampled + padding
    if expect:
        assert trainer._compute_old_log_prob(batch, {}) is batch
    else:
        with pytest.raises(ValueError, match="exactly one mini-batch") as e:
            trainer._compute_old_log_prob(batch, {})
        assert f"{total} rows ({sampled} sampled + {padding} padding)" in str(e.value)
    # the worker sees total/dp rows against a per-rank mini-batch of ppo_mini*n/dp
    assert _worker_accepts(total // dp, train_batch_size * n // dp) is expect


# ---------------------------------------------------------------- D2: engine restriction
@pytest.mark.parametrize(("strategy", "ok"), [("fsdp", True), ("fsdp2", True), ("megatron", False)])
def test_strategy_gate(strategy, ok):
    cfg = _with(_compose(), **{"actor_rollout_ref.actor.strategy": strategy})
    if ok:
        _validate_skip_old_log_prob_config(cfg)
    else:
        with pytest.raises(ValueError, match=f"actor.strategy={strategy}"):
            _validate_skip_old_log_prob_config(cfg)


# ---------------------------------------------------------------- D7: dropout detection
def _save(cfg, tmp_path, name):
    d = tmp_path / name
    cfg.save_pretrained(d)
    return str(d)


def test_dropout_in_top_level_and_text_config_is_detected(tmp_path):
    from transformers import LlamaConfig, LlavaConfig, Qwen3Config

    top = _save(
        Qwen3Config(
            attention_dropout=0.1, hidden_size=8, num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1
        ),
        tmp_path,
        "top",
    )
    assert _hf_config_dropouts(OmegaConf.create({"path": top})) == [("attention_dropout", 0.1)]
    nested = _save(
        LlavaConfig(
            text_config=LlamaConfig(attention_dropout=0.2, hidden_size=8, num_attention_heads=1, num_key_value_heads=1)
        ),
        tmp_path,
        "nested",
    )
    assert ("attention_dropout", 0.2) in _hf_config_dropouts(OmegaConf.create({"path": nested}))
    clean = _save(
        Qwen3Config(
            attention_dropout=0.0, hidden_size=8, num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1
        ),
        tmp_path,
        "clean",
    )
    assert _hf_config_dropouts(OmegaConf.create({"path": clean})) == []
    cfg = _with(_compose(), **{"actor_rollout_ref.model.path": top})
    with pytest.raises(ValueError, match="attention_dropout=0.1"):
        _validate_skip_old_log_prob_config(cfg)


def test_vision_tower_dropout_is_detected_too(tmp_path):
    """A VLM actor runs its vision tower in train mode during the update as well; dropout there perturbs
    the image features and hence the log-probs exactly like text dropout does."""
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig

    path = _save(
        LlavaConfig(
            text_config=LlamaConfig(attention_dropout=0.0, hidden_size=8, num_attention_heads=1, num_key_value_heads=1),
            vision_config=CLIPVisionConfig(attention_dropout=0.1, hidden_size=8, num_attention_heads=1),
        ),
        tmp_path,
        "vision",
    )
    assert _hf_config_dropouts(OmegaConf.create({"path": path})), "vision_config dropout not detected"


def test_unloadable_config_warns_and_accepts(tmp_path):
    cfg = _with(_compose(), **{"actor_rollout_ref.model.path": str(tmp_path / "does-not-exist")})
    out = io.StringIO()
    with redirect_stdout(out):
        _validate_skip_old_log_prob_config(cfg)  # accepted
    assert "could not load the model's HF config" in out.getvalue()
