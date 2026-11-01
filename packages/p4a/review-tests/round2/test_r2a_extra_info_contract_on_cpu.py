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
"""R2a (round-2 finding 1, the reward-manager contract), attacked under all five reward-loop managers:
the dataset's ``extra_info`` absent / None / empty / an OmegaConf mapping / a string, ``__num_turns__`` and
``reward_scores`` absent, dataset keys colliding with the loop's, nested-value sharing, multi-row sessions."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

import verl.experimental.reward_loop.reward_manager.base as base_module
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager
from verl.experimental.reward_loop.reward_manager.gdpo import GDPORewardManager
from verl.experimental.reward_loop.reward_manager.limited import RateLimitedRewardManager
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
from verl.experimental.reward_loop.reward_manager.remote import RemoteRewardManager

MANAGERS = [NaiveRewardManager, DAPORewardManager, GDPORewardManager, RateLimitedRewardManager, RemoteRewardManager]
# Round 3 (the submitter's R2-04, fix 0d4cd8ca): with the flag off each manager keeps main's contract. naive / remote merge the
# loop's tool fields plus num_turns / rollout_reward_scores always; rate_limited merges the tool fields always and adds
# the two rollout fields only with the flag on; dapo / gdpo merge only with the flag on.
TOOL_FIELDS_ALWAYS = (NaiveRewardManager, RemoteRewardManager, RateLimitedRewardManager)
ROLLOUT_FIELDS_ALWAYS = (NaiveRewardManager, RemoteRewardManager)
TOOL_KEYS = ("generated_turn_spans", "turn_rewards", "episode_done", "tool_rewards")


def meta(row=0):
    return {"turn_scores": [], "tool_rewards": [1.0], "generated_turn_spans": [(0, 2)], "turn_rewards": [1.0],
            "episode_done": True, "row": row}


def batch(n=1, *, extra_info="absent", num_turns=3, reward_scores="absent", per_row_dataset=False):
    """Rows the way ``AgentLoopWorker._compute_score`` builds them (one object array per dataset kwarg)."""
    ntb = {"data_source": np.array(["toy"] * n), "reward_model": np.array([{"ground_truth": "x"}] * n, dtype=object)}
    fields = np.empty(n, dtype=object)
    for i in range(n):
        fields[i] = meta(i)
    ntb["tool_extra_fields"] = fields
    if per_row_dataset:
        column = np.empty(n, dtype=object)
        for i in range(n):
            column[i] = {"dataset_row": i}
        ntb["extra_info"] = column
    elif not (isinstance(extra_info, str) and extra_info == "absent"):
        column = np.empty(n, dtype=object)
        for i in range(n):
            column[i] = extra_info
        ntb["extra_info"] = column
    if num_turns is not None:
        ntb["__num_turns__"] = np.array([num_turns] * n)
    if not (isinstance(reward_scores, str) and reward_scores == "absent"):
        ntb["reward_scores"] = np.array([reward_scores] * n, dtype=object)
    tensors = TensorDict({"responses": torch.tensor([[1, 2]] * n), "attention_mask": torch.tensor([[1, 1, 1]] * n)},
                         batch_size=[n])
    return DataProto(batch=tensors, non_tensor_batch=ntb)


def manager(cls, compute_score, flag=None):
    """``flag=None`` leaves the key out of the config (= off); True / False set it the way the trainer config does."""
    raw = {"reward": {}, "trainer": {"experiment_name": "r2a"}}
    if flag is not None:
        raw["actor_rollout_ref"] = {"rollout": {"multi_turn": {"turn_level_rewards": flag}}}
    config = OmegaConf.create(raw)
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "final response")
    if cls is RemoteRewardManager:  # its __init__ starts Ray actors: in-process stand-in for the worker pool
        obj = object.__new__(cls)
        RewardManagerBase.__init__(obj, config, tokenizer, compute_score)
        obj.is_async_reward_score, obj.reward_router_address, obj.reward_model_tokenizer = False, None, None

        class _Handle:
            @staticmethod
            async def remote(**kwargs):
                return compute_score(**kwargs)

        obj.choose_reward_worker = lambda: SimpleNamespace(compute_score=_Handle)
        return obj
    if cls is RateLimitedRewardManager:
        for attr in ("_semaphore", "_rpm_limiter", "_tpm_limiter"):
            setattr(cls, attr, None)
        cls._class_initialized = False
    return cls(config, tokenizer, compute_score)


def capture(store):
    def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
        if extra_info is not None:
            store.update(extra_info)
        store["__obj__"] = extra_info
        return 1.0

    return compute_score


@pytest.mark.parametrize("flag", [True, False], ids=["on", "off"])
@pytest.mark.parametrize("cls", MANAGERS)
@pytest.mark.parametrize("extra_info", ["absent", None, {}, OmegaConf.create({"split": "train"})],
                         ids=["absent", "none", "empty", "dictconfig"])
async def test_metadata_reaches_the_reward_fn_per_the_manager_contract(cls, extra_info, flag):
    """Flag on: every manager hands over a fresh dict with the loop's keys, num_turns and rollout_reward_scores.
    Flag off: naive / remote the same; rate_limited the loop's keys without the two rollout fields; dapo / gdpo the
    dataset's extra_info exactly as stored (the object itself, None stays None, absent is a fresh {})."""
    seen, data = {}, batch(extra_info=extra_info)
    if cls is GDPORewardManager and not flag and extra_info is None:  # gdpo assigns experiment_name: as on main
        with pytest.raises(TypeError):
            await manager(cls, capture(seen), flag=flag).run_single(data)
        return
    result = await manager(cls, capture(seen), flag=flag).run_single(data)
    assert result["reward_score"] == 1.0
    stored = data.non_tensor_batch["extra_info"][0] if "extra_info" in data.non_tensor_batch else None
    if flag or cls in TOOL_FIELDS_ALWAYS:
        for key in TOOL_KEYS:
            assert seen[key] == meta()[key], key
        assert type(seen["__obj__"]) is dict and seen["__obj__"] is not stored, "a fresh plain dict"
        if isinstance(extra_info, DictConfig):
            assert seen["split"] == "train"
        if flag or cls in ROLLOUT_FIELDS_ALWAYS:
            assert seen["num_turns"] == 3 and seen["rollout_reward_scores"] == {}
        else:
            assert "num_turns" not in seen and "rollout_reward_scores" not in seen
    else:  # dapo / gdpo with the flag off
        assert not (set(TOOL_KEYS) | {"num_turns", "rollout_reward_scores"}) & set(seen)
        if isinstance(extra_info, str):  # absent
            assert seen["__obj__"] == {} or seen["__obj__"] == {"experiment_name": "r2a"}
        elif extra_info is None:
            assert seen["__obj__"] is None
        else:
            assert seen["__obj__"] is stored, "the dataset's own object, as on main"


def test_num_turns_absent_is_none_and_reward_scores_default_to_an_empty_dict():
    info = RewardManagerBase.extra_info_for_reward_fn(batch(num_turns=None)[0])
    assert info["num_turns"] is None and info["rollout_reward_scores"] == {}
    assert RewardManagerBase.extra_info_for_reward_fn(batch(reward_scores={"acc": 1.0})[0])["rollout_reward_scores"] == {
        "acc": 1.0
    }


def test_dataset_keys_that_collide_with_the_loop_keys_lose_to_the_loop():
    info = RewardManagerBase.extra_info_for_reward_fn(
        batch(extra_info={"turn_rewards": "dataset", "num_turns": 99, "episode_done": "dataset", "split": "x"})[0]
    )
    assert info["turn_rewards"] == [1.0] and info["num_turns"] == 3 and info["episode_done"] is True
    assert info["split"] == "x"


@pytest.mark.parametrize(
    "cls,flag",
    [(NaiveRewardManager, True), (NaiveRewardManager, False), (RemoteRewardManager, True), (RemoteRewardManager, False),
     (RateLimitedRewardManager, True), (RateLimitedRewardManager, False), (DAPORewardManager, True)],
)
async def test_a_non_mapping_extra_info_is_passed_through_unchanged_with_one_warning_per_type(cls, flag, caplog):
    """LOW 1, closed by d8c4b925: on the merge path a ``str`` extra_info column reaches the reward function untouched,
    without the merge and without ``num_turns``, with one warning per type (main's naive / remote / rate_limited
    raised TypeError on it)."""
    seen = {}

    def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
        seen["extra_info"] = extra_info
        return 1.0

    base_module._NON_MAPPING_EXTRA_INFO_WARNED.clear()
    caplog.set_level(logging.WARNING)
    for _ in range(2):
        result = await manager(cls, compute_score, flag=flag).run_single(batch(extra_info='{"split": "train"}'))
        assert result["reward_score"] == 1.0 and seen["extra_info"] == '{"split": "train"}'
    warned = [record for record in caplog.records if "not a mapping" in record.getMessage()]
    assert len(warned) == 1 and "str" in warned[0].getMessage(), "one warning per type"


async def test_dapo_with_the_flag_off_passes_a_non_mapping_extra_info_through_without_any_warning(caplog):
    """Round 3 (0d4cd8ca): dapo off is main's code path, the stored object itself and no helper logic at all."""
    seen, data = {}, batch(extra_info='{"split": "train"}')

    def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
        seen["__obj__"] = extra_info
        return 1.0

    base_module._NON_MAPPING_EXTRA_INFO_WARNED.clear()
    caplog.set_level(logging.WARNING)
    await manager(DAPORewardManager, compute_score, flag=False).run_single(data)
    assert seen["__obj__"] is data.non_tensor_batch["extra_info"][0]
    assert not [record for record in caplog.records if "not a mapping" in record.getMessage()]


@pytest.mark.parametrize("flag", [True, False], ids=["on", "off"])
async def test_gdpo_still_cannot_take_a_non_mapping_extra_info_exactly_as_on_main(flag):
    """Outside the fix: ``gdpo`` assigns ``extra_info["experiment_name"]`` after the helper, so a ``str`` raises
    TypeError there in both flag states, as on main @ 2ab35292. A pre-existing inconsistency (LOW), pinned."""
    with pytest.raises(TypeError):
        await manager(GDPORewardManager, lambda **kwargs: 1.0, flag=flag).run_single(batch(extra_info='{"split": "train"}'))


async def test_list_valued_tool_fields_are_copied_so_a_mutating_reward_function_cannot_reach_the_row():
    """LOW 2, closed by d8c4b925: the span / reward lists handed over are copies of the loop's lists."""

    def mutating(data_source, solution_str, ground_truth, extra_info, **kwargs):
        extra_info["generated_turn_spans"].append((9, 9))  # a reward function editing what it was given
        extra_info["turn_rewards"][0] = 99.0
        extra_info["new_key"] = 1
        return 1.0

    data = batch()
    await manager(NaiveRewardManager, mutating).run_single(data)
    row = data.non_tensor_batch["tool_extra_fields"][0]
    assert "new_key" not in row, "the dict itself is a copy"
    assert row["generated_turn_spans"] == [(0, 2)] and row["turn_rewards"] == [1.0], "the lists are copies too"


@pytest.mark.parametrize("flag", [True, False], ids=["on", "off"])
@pytest.mark.parametrize("cls", MANAGERS)
async def test_a_multi_output_session_is_scored_on_the_last_row_with_that_rows_metadata(cls, flag):
    seen = {}
    await manager(cls, capture(seen), flag=flag).run_single(batch(n=3, per_row_dataset=True))
    assert seen["dataset_row"] == 2, "the last row's dataset extra_info"
    if flag or cls in TOOL_FIELDS_ALWAYS:
        assert seen["row"] == 2, "the last row's loop metadata"
    else:
        assert "row" not in seen
