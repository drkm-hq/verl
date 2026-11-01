# Round-3 re-review tests for contrib/p2-oldlogprob-skip @ 5532f3a4 (R2-01: W4A4 QAT train-mode observers).
# Run from wt/p2 with PYTHONPATH=<worktree> and the CPU harness; see REVIEW-ROUND3-VERDICT.md.
"""Each test states the behaviour the reviewer expects of the fixed branch; a failing test is a finding."""

import ast
import copy
from enum import Enum
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from transformers import Qwen3Config

from verl.utils.config import _validate_skip_old_log_prob_config, omega_conf_to_dataclass
from verl.utils.model import find_train_mode_dependent_modules
from verl.workers.config.actor import FSDPActorConfig
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, os.pardir))  # the repository root

_WT = Path(os.path.join(_REPO, "wt/p2"))
_LINEAR = _WT / "verl" / "utils" / "qat" / "linear.py"
A = "actor_rollout_ref.actor"
QAT = f"{A}.fsdp_config.qat"
REASON = "the activation observer updates input_global_scale in train mode only"


def _qat_classes():
    """verl's unmodified QATMode / QATLinear from source, CUDA quantizer replaced (triton is not installed)."""
    nodes = [n for n in ast.parse(_LINEAR.read_text()).body if isinstance(n, ast.ClassDef) and n.name in {"QATMode", "QATLinear"}]

    class Spy:
        @staticmethod
        def apply(x, amax, group_size):
            return x

    scope = dict(torch=torch, nn=nn, F=F, Enum=Enum, Optional=Optional, FP4_E2M1_MAX=6.0, FP8_E4M3_MAX=448.0, STEFP4QuantTriton=Spy)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_LINEAR), "exec"), scope)
    return scope["QATMode"], scope["QATLinear"]


@pytest.fixture(scope="module")
def on_policy_config(tmp_path_factory):
    path = tmp_path_factory.mktemp("model")
    Qwen3Config(attention_dropout=0.0).save_pretrained(path)
    overrides = [
        f"{A}.skip_old_log_prob_when_on_policy=true", "data.train_batch_size=8", f"{A}.ppo_mini_batch_size=8",
        f"{A}.ppo_micro_batch_size_per_gpu=1", "algorithm.adv_estimator=grpo",
    ]  # fmt: skip
    with initialize_config_dir(version_base=None, config_dir=str(_WT / "verl" / "trainer" / "config")):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    config.actor_rollout_ref.model.path = str(path)
    return config


def _with(config, **updates):
    config = copy.deepcopy(config)
    for key, value in updates.items():
        OmegaConf.update(config, key, value, force_add=True)
    return config


def _rejected(config, *fragments):
    with pytest.raises(ValueError, match="requires a strictly on-policy update") as excinfo:
        _validate_skip_old_log_prob_config(config)
    for fragment in fragments:
        assert fragment in str(excinfo.value)


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2", "fsdp_turbo"])
def test_validator_reads_the_key_the_fsdp_engine_reads(on_policy_config, strategy):
    """The engine reads ``engine_config.qat``; for every accepted strategy ``FSDPActorConfig.__post_init__`` makes
    that ``fsdp_config`` (dp_actor.yaml composes it from engine/fsdp.yaml), the key the validator inspects."""
    config = _with(on_policy_config, **{f"{A}.strategy": strategy, f"{QAT}.enable": True, f"{QAT}.mode": "w4a4"})
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    assert isinstance(actor, FSDPActorConfig) and actor.engine is actor.fsdp_config
    assert (actor.engine.qat.enable, actor.engine.qat.mode) == (True, "w4a4")
    _rejected(config, "actor.fsdp_config.qat.mode=w4a4 with qat.enable=True")


def test_upper_case_mode_is_rejected_as_apply_qat_lowercases_it(on_policy_config):
    QATMode, _ = _qat_classes()
    with pytest.raises(ValueError):
        QATMode("W4A4")
    assert QATMode("W4A4".lower()) is QATMode.W4A4  # core.apply_qat: QATMode(config.mode.lower())
    _rejected(_with(on_policy_config, **{f"{QAT}.enable": True, f"{QAT}.mode": "W4A4"}), "w4a4")


def test_enum_valued_mode_is_rejected(on_policy_config):
    """A programmatic DictConfig can carry ``QATMode.W4A4`` itself: ``apply_qat`` lowercases the str-enum to
    ``"w4a4"`` and the engine's restore gate compares equal, so W4A4 is applied; the validator must see it too."""
    QATMode, _ = _qat_classes()
    config = _with(on_policy_config, **{f"{QAT}.enable": True, f"{QAT}.mode": QATMode.W4A4})
    assert config.actor_rollout_ref.actor.fsdp_config.qat.mode.lower() == "w4a4"
    _rejected(config, "w4a4")


@pytest.mark.parametrize("observer", ["static_minmax", "memoryless_minmax", "minmax"])
def test_every_w4a4_observer_is_rejected_and_flagged(on_policy_config, observer):
    """All three observers run under ``if self.training`` (linear.py); the rejection must not depend on the observer."""
    QATMode, QATLinear = _qat_classes()
    _rejected(_with(on_policy_config, **{f"{QAT}.enable": True, f"{QAT}.mode": "w4a4", f"{QAT}.activation_observer": observer}), "w4a4")
    module = QATLinear(8, 8, mode=QATMode.W4A4, activation_observer=observer)
    assert find_train_mode_dependent_modules(module) == [("", f"QATLinear(mode=w4a4): {REASON}")]


def test_subclass_under_checkpoint_wrapper_is_flagged_and_w4a16_has_no_training_branch():
    QATMode, QATLinear = _qat_classes()

    class Sub(QATLinear):
        pass

    wrapped = nn.Sequential(checkpoint_wrapper(nn.Sequential(Sub(8, 8, mode=QATMode.W4A4))))
    assert find_train_mode_dependent_modules(wrapped) == [("0._checkpoint_wrapped_module.0", f"Sub(mode=w4a4): {REASON}")]
    assert find_train_mode_dependent_modules(QATLinear(8, 8, mode=QATMode.W4A16)) == []
    # the W4A16 acceptance rests on this: every ``self.training`` read in linear.py sits in _fake_quantize_activation
    tree = ast.parse(_LINEAR.read_text())
    owners = {
        fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        for node in ast.walk(fn)
        if isinstance(node, ast.Attribute) and node.attr == "training" and isinstance(node.value, ast.Name) and node.value.id == "self"
    }
    assert owners == {"_fake_quantize_activation"}


def test_qat_on_ref_or_critic_only_and_the_inert_top_level_actor_qat_key_are_accepted(on_policy_config):
    """The skip concerns the actor's forward only. ``actor.qat`` (actor.yaml's top-level block) has no reader in the
    FSDP engine, which takes ``engine_config.qat`` = ``fsdp_config.qat``; accepting it is consistent with the engine."""
    _validate_skip_old_log_prob_config(_with(on_policy_config, **{"actor_rollout_ref.ref.fsdp_config.qat.enable": True, "actor_rollout_ref.ref.fsdp_config.qat.mode": "w4a4"}))
    _validate_skip_old_log_prob_config(_with(on_policy_config, **{"critic.model.fsdp_config.qat.enable": True, "critic.model.fsdp_config.qat.mode": "w4a4"}))
    top = _with(on_policy_config, **{f"{A}.qat.enable": True, f"{A}.qat.mode": "w4a4"})
    _validate_skip_old_log_prob_config(top)
    actor = omega_conf_to_dataclass(top.actor_rollout_ref.actor)
    assert dict(actor.qat)["enable"] is True and actor.engine.qat.enable is False


def test_probe_configuration_is_rejected_with_the_documented_message(on_policy_config):
    _rejected(
        _with(on_policy_config, **{f"{QAT}.enable": True, f"{QAT}.mode": "w4a4"}),
        "actor.fsdp_config.qat.mode=w4a4 with qat.enable=True (W4A4 QAT's activation observers update the activation "
        "quantization scale in train mode only, so the update pass would quantize with different scales than the "
        "eval-mode old-log-prob pass; W4A16 is weight-only and accepted, but untested with the skip)",
    )
    _validate_skip_old_log_prob_config(_with(on_policy_config, **{f"{QAT}.enable": True, f"{QAT}.mode": "w4a16"}))
