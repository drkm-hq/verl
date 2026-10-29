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
"""V1 trainer wiring of ``actor.skip_old_log_prob_when_on_policy``.

``_compute_old_log_prob`` runs no forward; ``_update_actor`` asks the worker for the update pass's
per-token outputs, reads the actor metrics off the returned KVBatchMeta and stores the response
part of those outputs as ``old_log_probs``/``entropy`` exactly as the old-log-prob pass would have
(same ``actor/entropy`` and ``training/rollout_*`` definitions). TransferQueue is replaced by an
in-memory stand-in and the worker group by a mock, so no ray or GPU is needed.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils import tensordict_utils as tu

PROMPT_LENS, RESP_LENS = [3, 2], [2, 3]


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _FakeTransferQueue:
    """``kv_batch_get`` / ``kv_batch_put`` over one in-memory TensorDict."""

    def __init__(self, store):
        self.store = store
        self.puts = []

    def kv_batch_get(self, keys, partition_id, select_fields):
        present = [field for field in select_fields if field in self.store.keys()]  # TQ drops unknown fields
        return self.store.select(*present)

    def kv_batch_put(self, keys, partition_id, fields):
        self.store.update(fields)
        self.puts.append(sorted(fields.keys()))
        return KVBatchMeta(keys=list(keys), tags=[{}] * len(keys), partition_id=partition_id)


def _nested(rows):
    return torch.nested.as_nested_tensor(list(rows), layout=torch.jagged)


def _store():
    torch.manual_seed(0)
    return tu.get_tensordict(
        tensor_dict={
            "log_probs": _nested(-torch.rand(p + r) for p, r in zip(PROMPT_LENS, RESP_LENS, strict=True)),
            "entropy": _nested(torch.rand(p + r) for p, r in zip(PROMPT_LENS, RESP_LENS, strict=True)),
            "response_mask": _nested(torch.ones(r, dtype=torch.long) for r in RESP_LENS),
            "responses": _nested(torch.randint(0, 10, (r,)) for r in RESP_LENS),
            "rollout_log_probs": _nested(-torch.rand(r) for r in RESP_LENS),
        }
    )


def _trainer(skip: bool, entropy_coeff: float = 0.0, rollout_correction=None, critic_warmup: int = 0):
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.global_steps = 1
    trainer.config = OmegaConf.create(
        {
            "trainer": {"critic_warmup": critic_warmup},
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 2,
                    "calculate_entropy": False,
                    "entropy_coeff": entropy_coeff,
                    "ppo_epochs": 1,
                    "data_loader_seed": 1,
                    "shuffle": False,
                    "loss_agg_mode": "token-mean",
                    "loss_scale_factor": None,
                    "skip_old_log_prob_when_on_policy": skip,
                    "policy_loss": {},
                },
                "rollout": {"n": 1, "temperature": 1.0, "calculate_log_probs": True},
            },
            "algorithm": {"rollout_correction": rollout_correction},
        }
    )
    trainer.actor_rollout_wg = MagicMock()
    return trainer


def _batch(num_rows: int = 2, num_padding: int = 0):
    keys = [f"s{i}" for i in range(num_rows)]
    tags = [{"is_padding": i >= num_rows - num_padding} for i in range(num_rows)]
    return KVBatchMeta(keys=keys, tags=tags, partition_id="train")


def _update_output(batch):
    # what the TQ bridge hands back for a batch-sized update_actor output: a KVBatchMeta whose extra_info
    # carries one metrics dict per collecting rank (rank 0 first)
    return KVBatchMeta(
        keys=list(batch.keys),
        tags=[{}] * len(batch.keys),
        partition_id=batch.partition_id,
        extra_info={"metrics": [{"mfu": 0.5, "pg_loss": [1.0]}, {"mfu": 0.7, "pg_loss": [9.0]}]},
    )


def _check_stored(fake: _FakeTransferQueue, metrics: dict):
    """old_log_probs/entropy are the response slices of the full-sequence outputs (response token r is
    predicted by sequence position prompt_len - 1 + r) and the metrics are computed from them."""
    assert fake.puts == [["entropy", "old_log_probs"]]
    expected = {}
    pristine = _store()  # the full-sequence outputs as the worker wrote them (seeded, so identical)
    for key, out in (("old_log_probs", "log_probs"), ("entropy", "entropy")):
        rows = pristine[out].unbind()
        expected[key] = [row[p - 1 : p - 1 + r] for row, p, r in zip(rows, PROMPT_LENS, RESP_LENS, strict=True)]
        stored = fake.store[key]
        assert [t.shape[0] for t in stored.unbind()] == RESP_LENS
        assert torch.equal(stored.values(), torch.cat(expected[key]))
    assert metrics["actor/entropy"] == pytest.approx(torch.cat(expected["entropy"]).mean().item())
    assert metrics["training/rollout_probs_diff_valid"] == 1
    assert "training/rollout_logprobs_diff_max" in metrics


def test_compute_old_log_prob_runs_no_forward_under_the_skip(monkeypatch):
    fake = _FakeTransferQueue(_store())
    monkeypatch.setattr(trainer_base, "tq", fake)
    trainer = _trainer(skip=True)
    batch, metrics = _batch(), {}

    assert trainer._compute_old_log_prob(batch, metrics) is batch

    trainer.actor_rollout_wg.compute_log_prob.assert_not_called()
    assert metrics == {} and fake.puts == [] and "old_log_probs" not in fake.store.keys()


def test_compute_old_log_prob_without_the_flag_still_runs_the_pass(monkeypatch):
    fake = _FakeTransferQueue(_store())
    monkeypatch.setattr(trainer_base, "tq", fake)
    trainer = _trainer(skip=False)
    batch, metrics = _batch(), {}
    trainer.actor_rollout_wg.compute_log_prob.return_value = _batch()

    trainer._compute_old_log_prob(batch, metrics)

    trainer.actor_rollout_wg.compute_log_prob.assert_called_once()
    assert batch.extra_info["calculate_entropy"] is True and batch.extra_info["compute_loss"] is False
    _check_stored(fake, metrics)


@pytest.mark.parametrize("entropy_coeff", [0.0, 0.01])
def test_update_actor_requests_and_stores_the_update_pass_outputs(monkeypatch, entropy_coeff):
    fake = _FakeTransferQueue(_store())
    monkeypatch.setattr(trainer_base, "tq", fake)
    trainer = _trainer(skip=True, entropy_coeff=entropy_coeff)
    batch, metrics = _batch(), {}
    extra = {}

    def update_actor(meta):
        extra.update(meta.extra_info)  # what the worker saw (the skip keys are removed after the call)
        return _update_output(meta)

    trainer.actor_rollout_wg.update_actor.side_effect = update_actor

    assert trainer._update_actor(batch, metrics) is batch

    assert extra["calculate_entropy"] is True
    assert extra["entropy_requires_grad"] is (entropy_coeff != 0.0)
    assert extra["return_model_output"] is True
    assert extra["on_policy_skip_old_log_prob"] is True
    # rank 0's metrics, as on the TensorDict path
    assert metrics["perf/mfu/actor"] == 0.5 and metrics["actor/pg_loss"] == 1.0
    _check_stored(fake, metrics)


def test_update_actor_without_the_flag_is_unchanged(monkeypatch):
    fake = _FakeTransferQueue(_store())
    monkeypatch.setattr(trainer_base, "tq", fake)
    trainer = _trainer(skip=False)
    batch, metrics = _batch(), {}
    trainer.actor_rollout_wg.update_actor.return_value = {"metrics": {"mfu": 0.5, "pg_loss": [2.0]}}

    trainer._update_actor(batch, metrics)

    for key in ("return_model_output", "on_policy_skip_old_log_prob", "entropy_requires_grad"):
        assert key not in batch.extra_info
    assert batch.extra_info["calculate_entropy"] is False
    assert metrics["actor/pg_loss"] == 2.0 and metrics["perf/mfu/actor"] == 0.5
    assert fake.puts == []


def test_skip_keys_do_not_linger_in_extra_info_after_the_update(monkeypatch):
    fake = _FakeTransferQueue(_store())
    monkeypatch.setattr(trainer_base, "tq", fake)
    trainer = _trainer(skip=True)
    batch = _batch()
    trainer.actor_rollout_wg.update_actor.return_value = _update_output(batch)

    trainer._update_actor(batch, {})

    sent = trainer.actor_rollout_wg.update_actor.call_args.args[0]
    assert sent is batch  # the keys travelled with the call ...
    for key in ("return_model_output", "on_policy_skip_old_log_prob", "entropy_requires_grad"):
        assert key not in batch.extra_info  # ... and are gone afterwards


@pytest.mark.parametrize(("num_rows", "num_padding"), [(4, 2), (4, 0), (3, 1)])
def test_a_batch_that_is_not_exactly_one_mini_batch_is_refused(monkeypatch, num_rows, num_padding):
    """ppo_mini_batch_size x n = 2 rows: a padded batch of 4 rows would be two optimizer steps, the second
    one after the weights moved; the skip must refuse before any forward, naming the row counts."""
    fake = _FakeTransferQueue(_store())
    monkeypatch.setattr(trainer_base, "tq", fake)
    trainer = _trainer(skip=True)
    batch = _batch(num_rows, num_padding)

    with pytest.raises(ValueError, match="exactly one mini-batch") as excinfo:
        trainer._compute_old_log_prob(batch, {})
    assert f"{num_rows} rows ({num_rows - num_padding} sampled + {num_padding} padding)" in str(excinfo.value)
    trainer.actor_rollout_wg.compute_log_prob.assert_not_called()

    with pytest.raises(ValueError, match="exactly one mini-batch"):
        trainer._update_actor(batch, {})
    trainer.actor_rollout_wg.update_actor.assert_not_called()
    assert fake.puts == []


def test_critic_warmup_steps_run_the_old_log_prob_pass(monkeypatch):
    """No actor update during critic warmup, so there is no update pass to take the log-probs from: the
    old pass runs as before and actor/entropy, training/rollout_* keep being logged."""
    fake = _FakeTransferQueue(_store())
    monkeypatch.setattr(trainer_base, "tq", fake)
    trainer = _trainer(skip=True, critic_warmup=5)
    batch, metrics = _batch(), {}
    trainer.actor_rollout_wg.compute_log_prob.return_value = _batch()

    trainer._compute_old_log_prob(batch, metrics)

    trainer.actor_rollout_wg.compute_log_prob.assert_called_once()
    _check_stored(fake, metrics)


def test_trainer_init_validates_the_flag_itself(monkeypatch):
    """A trainer built outside main_ppo (recipes, custom mains) must still refuse an inexact configuration."""
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[4] / "verl" / "trainer" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.skip_old_log_prob_when_on_policy=true",
                "actor_rollout_ref.actor.ppo_epochs=2",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
            ],
        )
    monkeypatch.setattr(trainer_base, "build_checkpoint_callback", lambda config: None)
    monkeypatch.setattr(trainer_base, "RolloutMoELoadBalanceMetricsAccumulator", lambda **kwargs: None)
    monkeypatch.setattr(_StubTrainer, "_build_replay_buffer", lambda self: None)

    with pytest.raises(ValueError, match="actor.ppo_epochs=2"):
        _StubTrainer(config)


def test_metrics_only_rollout_correction_is_logged_after_the_update(monkeypatch):
    rollout_correction = {
        "rollout_is": None,
        "rollout_is_threshold": 2.0,
        "rollout_is_batch_normalize": False,
        "rollout_rs": None,
        "rollout_rs_threshold": None,
        "bypass_mode": False,
    }
    fake = _FakeTransferQueue(_store())
    monkeypatch.setattr(trainer_base, "tq", fake)
    trainer = _trainer(skip=True, rollout_correction=rollout_correction)
    batch, metrics = _batch(), {}
    trainer.actor_rollout_wg.update_actor.return_value = _update_output(batch)

    trainer._update_actor(batch, metrics)

    assert any(key.startswith("rollout_corr/") for key in metrics)
    _check_stored(fake, metrics)  # and nothing but old_log_probs/entropy was written back
