# Adversarial review tests for contrib/p2-oldlogprob-skip (config validation).
# ADAPTED after the fixes (REVIEW.md "Fixes applied"): plain Megatron is now rejected too (D2: unverified
# engines raise), and the padding cases assert the trainer's runtime refusal (D1) instead of the impossible
# arithmetic.
# Run from the worktree with PYTHONPATH=<worktree>; see REVIEW.md section 5.
"""Validation gaps: every test states the behaviour the reviewer expects; a failing test is a finding."""

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils.config import _validate_skip_old_log_prob_config, validate_config
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))  # the repository root

_WT = Path(os.path.join(_REPO, "wt/p2"))
_CONFIG_DIR = _WT / "verl" / "trainer" / "config"

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


def test_parameter_sync_step_is_read_for_any_trainer_mode():
    """The trainer reads trainer.v1.<mode>.parameter_sync_step for every mode (trainer_base.py:131), so a
    colocate_async run with parameter_sync_step=2 samples train_batch_size/2 prompts per update."""
    cfg = _with(
        _compose(),
        **{
            "trainer.v1.trainer_mode": "colocate_async",
            "trainer.v1.colocate_async.parameter_sync_step": 2,
        },
    )
    with pytest.raises(
        ValueError, match="!= data.train_batch_size / parameter_sync_step = 8"
    ):
        _validate_skip_old_log_prob_config(cfg)
    cfg = _with(_compose(), **{"trainer.v1.sync.parameter_sync_step": 2})
    with pytest.raises(ValueError, match="parameter_sync_step = 8"):
        _validate_skip_old_log_prob_config(cfg)


def test_megatron_dynamic_context_parallel_must_be_rejected():
    """megatron/transformer_impl.py:1436-1437 drops model_output from the training pass when
    dynamic_context_parallel=True, so the update pass cannot hand back log_probs; the flag must raise at
    validation instead of an IndexError inside train_mini_batch."""
    cfg = _compose("model_engine=megatron")
    assert cfg.actor_rollout_ref.actor.strategy == "megatron"
    with pytest.raises(ValueError, match="actor.strategy=megatron"):  # D2 fix: unverified engines raise
        _validate_skip_old_log_prob_config(cfg)
    cfg = _with(
        cfg, **{"actor_rollout_ref.actor.megatron.dynamic_context_parallel": True}
    )
    with pytest.raises(ValueError):
        _validate_skip_old_log_prob_config(cfg)


def test_fused_kernels_with_entropy_bonus_is_accepted_and_flagged_as_grad():
    """use_fused_kernels=True with entropy_coeff>0: the FSDP fused branch ignores entropy_requires_grad and
    keeps the fused kernel's grad-attached entropy, so the bonus is intact (test_rev_fused_entropy_*)."""
    cfg = _with(
        _compose(),
        **{
            "actor_rollout_ref.model.use_fused_kernels": True,
            "actor_rollout_ref.actor.entropy_coeff": 0.001,
        },
    )
    _validate_skip_old_log_prob_config(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)


def _trainer_stub(
    train_batch_size, ppo_mini, n, use_critic=False, critic_warmup=0, global_steps=1
):
    return SimpleNamespace(
        use_critic=use_critic,
        global_steps=global_steps,
        config=OmegaConf.create(
            {
                "trainer": {"critic_warmup": critic_warmup},
                "critic": {"ppo_mini_batch_size": ppo_mini},
                "actor_rollout_ref": {
                    "actor": {"ppo_mini_batch_size": ppo_mini},
                    "rollout": {"n": n},
                },
                "data": {"train_batch_size": train_batch_size},
            }
        ),
    )


@pytest.mark.parametrize("extra_rows", [0, 1, 7, 128])
def test_balance_batch_padding_keeps_exactly_one_mini_batch(extra_rows):
    """The validator checks ppo_mini_batch_size == train_batch_size in prompts, but the batch that reaches
    the worker has len(batch) rows: the replay buffer materialises every `{uid}_{session}_{index}` key of
    the sampled prompts (replay_buffer.py:393-406; a session can emit several trajectories,
    agent_loop_tq.py:198), and _balance_batch pads to lcm(dp, ppo_mini*n) (trainer_base.py:1609-1631).
    With any extra row the padded batch is k >= 2 mini-batches: the worker runs k optimizer steps and
    ppo_loss is told on_policy_skip_old_log_prob=True on every one of them."""
    train_batch_size, n, dp = 16, 8, 8
    stub = _trainer_stub(train_batch_size, train_batch_size, n)
    multiple = PPOTrainer._get_required_batch_multiple(stub, dp)
    rows = train_batch_size * n + extra_rows
    padded = math.ceil(rows / multiple) * multiple
    mini_batch_rows = train_batch_size * n
    assert padded % mini_batch_rows == 0  # make_iterator's divisibility assert passes
    num_mini_batches = padded // mini_batch_rows

    # D1 fix: the trainer refuses any batch that is not exactly one mini-batch of rows, before any forward
    from transfer_queue import KVBatchMeta

    class _Stub(PPOTrainer):
        def on_step_end(self):
            pass

        def on_sample_end(self):
            pass

    trainer = _Stub.__new__(_Stub)
    trainer.global_steps = 1
    trainer.config = OmegaConf.create(
        {
            "trainer": {"critic_warmup": 0},
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": train_batch_size, "skip_old_log_prob_when_on_policy": True},
                "rollout": {"n": n},
            },
        }
    )
    trainer.actor_rollout_wg = SimpleNamespace(
        compute_log_prob=lambda batch: pytest.fail("no forward may run"),
        update_actor=lambda batch: pytest.fail("no update may run"),
    )
    batch = KVBatchMeta(
        keys=[str(i) for i in range(padded)],
        tags=[{"is_padding": i >= rows} for i in range(padded)],
        partition_id="train",
    )
    if num_mini_batches == 1:
        assert trainer._compute_old_log_prob(batch, {}) is batch
    else:
        with pytest.raises(ValueError, match="exactly one mini-batch") as excinfo:
            trainer._compute_old_log_prob(batch, {})
        assert f"{padded} rows ({rows} sampled + {padded - rows} padding)" in str(excinfo.value)
        with pytest.raises(ValueError, match="exactly one mini-batch"):
            trainer._update_actor(batch, {})
