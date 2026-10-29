# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os
import time
import uuid
from collections import Counter, defaultdict

import numpy as np
import transfer_queue as tq
from omegaconf import DictConfig
from transfer_queue import KVBatchMeta

from verl.utils.skip import SkipManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS = int(os.getenv("VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS", "60"))

DAPO_FILTERED_REWARD_COUNTS_KEY = "_dapo_filtered_reward_counts"
FILTER_GROUPS_REWARD_METRIC = "reward"


def _accumulate_eviction_metrics(acc: dict, new: dict, stale_count: int) -> None:
    """Merge one poll iteration's eviction metrics into ``acc`` in place.

    ``stale_count`` weights the staleness mean so it stays a true per-sample average across iterations.
    """
    stale_count_key = next((k for k in new if k.endswith("/off_policy/evicted_samples")), None)
    prev_stale_total = acc.get(stale_count_key, 0) if stale_count_key else 0

    for key, value in new.items():
        if key.endswith("/evicted_samples_staleness/mean"):
            denom = prev_stale_total + stale_count
            acc[key] = (acc.get(key, 0.0) * prev_stale_total + value * stale_count) / denom if denom else value
        elif key.endswith("/evicted_samples_staleness/max"):
            acc[key] = max(acc.get(key, value), value)
        elif key.endswith("/evicted_samples_staleness/min"):
            acc[key] = min(acc.get(key, value), value)
        elif key == DAPO_FILTERED_REWARD_COUNTS_KEY:
            # Dict-valued diagnostic: merge {metric_value: count} across poll iterations.
            merged = Counter(acc.get(key, {}))
            merged.update(value)
            acc[key] = dict(merged)
        else:
            acc[key] = acc.get(key, 0) + value


# TODO: Pass custom sampler to TransferQueue:
# https://github.com/Ascend/TransferQueue/blob/main/tutorial/05_custom_sampler.py


class ReplayBuffer:
    """ReplayBuffer is used by trainer to sample trajectories produced during rollout.

    We use [TransferQueue](https://github.com/Ascend/TransferQueue) as kv store to store trajectories.

    ### [Trajectories storage format]
    The key format is `{uid}_{session_id}_{index}`, where:
    - uid: Auto generated unique id when prompt is sampled from dataset.
    - session_id: Session id for GRPO group sampling: [0, n).
    - index: Index of output trajectory in a session.

    There're two types of data associated with each key: tag and value. The tag are arbitrary metadata:
    `{"status": "running", ...}` used to track the status of the trajectory.

    The value is a dictionary containing the following fields:
    - messages/datasource/reward_model/...: fields from dataset.
    - prompt_ids/response_ids/response_mask/...: fields from AgentLoopOutput.

    TransferQueue store tag and value separately, the tag are stored in meta server, while the value is stored
    in storage units.

    ### [GRPO group sampling control]
    Except trajectories, we also store raw prompts in TransferQueue with key `{uid}`, with `status` tag to track
    status of GRPO group sampling.
    - pending: the prompt is sampled from dataset but its sessions are not yet started.
    - running: all sessions of the prompt are running.
    - finished: all sessions of the prompt are finished without error.
    - failure: all sessions of the prompt are finished, but at least one session failed.
    Only prompts with status `finished` or `failure` enter terminal-group handling.

    ### [Terminal-group eviction/refill matrix]
    ``drop`` means off-policy staleness dropping. Both off-policy strategies (``drop`` and the dropless
    ``wait``, which blocks until stale in-flight prompts finish) are only for async trainers; sync sampling
    is on-policy, so ``max_off_policy_strategy`` is a NO-OP there.
    ``DAPO`` means filtering groups whose configured reward metric is identical across all trajectories,
             for async reward-computation path.
    ``failure`` is the group status described above.
    The matrix applies to the training partition, in which ``k`` is the number of prompts to evict.
    Validation treats all terminal groups as sampleable

    |   trainer mode   |   drop   |               DAPO               |            failure            |
    | ---------------- | -------- | -------------------------------- | ----------------------------- |
    |       sync       |   NO-OP  |    Evict ``k``; refill ``2k``    |  NO-OP or opt-in refill ``k`` |
    |      async       |              All the same: Evict ``k``; refill ``k``.                       |

    In sync mode, DAPO is opt-in and trades generation time for training stability. Each ``k`` evictions add
    ``2k`` logical refill credits, but prompts are fetched only as bounded pending/running slots become available.
    Terminal groups are filtered while other requests remain in flight. Once enough groups are sampleable, inflight
    requests are drained and discarded. By default, failed groups remain sampleable and missing trajectories are
    padded downstream. Setting ``sync_refill_failed_groups=True`` allows refilling failed samples.
    In async mode, ``num_warmup_batches`` absorbs retry cost, so all three paths refill exactly ``k`` prompts.

    Args:
        trainer_mode (str): Trainer mode.
        trainer_config (DictConfig): Trainer configuration.
        max_off_policy_threshold (int | None): Maximum number of model versions that trajectory can span.
            None disables off-policy version control entirely: no samples are dropped or awaited for
            staleness (off-policy degree is then only bounded by rollout concurrency and feed/consume
            balance).
        max_off_policy_strategy (str): How to handle trajectory that exceeds the maximum number of model versions.
            Ignored when ``max_off_policy_threshold`` is None.
        sampler_kwargs (dict): Additional kwargs for the custom sampler.
        poll_interval (float, optional): Poll interval in seconds. Defaults to 2.0.
        refill_fn (callable, optional): Trainer-injected function that submits an exact number of fresh prompts.
        filter_groups_metric (str, optional): DAPO group-filtering metric. ``"reward"`` reads the canonical
            pre-KL reward from ``rm_scores``; other names read each trajectory's
            ``extra_fields.reward_extra_info``. ``None`` disables DAPO filtering.
        train_batch_size (int, optional): Prompt count represented by one Sync DAPO in-flight batch.
        gen_batch_size (int, optional): Dataloader fetch granularity for refill dispatches.
        max_inflight_gen_batches (int): Maximum Sync DAPO prompt batches concurrently pending or running.
        sync_refill_failed_groups (bool): Whether sync sampling replaces failed groups with no trajectories.
        inflight_ttl_s (float, optional): Seconds a training prompt may stay pending/running (measured from
            its ``dispatched_at`` tag, engine queueing included) before it is handed to ``reissue_fn``.
            ``None`` disables the TTL.
        max_inflight_reissues (int): Re-issues one prompt may receive before it is marked ``failure`` instead
            (the ``attempt`` tag counts dispatches; the original is attempt 1). Defaults to 3.
        max_inflight_reissues_per_step (int, optional): Re-issues the TTL may trigger within one training step
            (the sampling calls that share a ``global_steps``) before the remaining expired prompts are waited
            for. ``None`` (default) resolves to ``max_inflight_reissues`` times the step's prompt count
            (``train_batch_size``, else the call's batch size). Prompts past ``max_inflight_reissues`` are
            failed regardless of it (``max_inflight_reissues=0``: fail on the first expiry, never re-issue).
        reissue_fn (callable, optional): Trainer-injected ``(partition_id, uids) -> list[str]`` that discards
            the partial rows of those in-flight prompts, re-dispatches them under fresh uids and returns the
            uids it actually retired (see [In-flight recovery]). Required when ``inflight_ttl_s`` is set.

    ### [In-flight recovery]
    Only the agent loop worker that runs a prompt writes its terminal status. If that worker dies or its
    generation silently hangs, the prompt stays ``running`` forever and sampling never completes. With
    ``inflight_ttl_s`` the loops re-issue such prompts through ``reissue_fn`` and keep polling.
    A re-issued prompt gets a fresh uid, so late writes from the retired attempt (a slow-not-dead worker) can
    never merge into the new group; the buffer sweeps any key the retired attempt still writes. The trainer
    never sees partial rows: the old trajectories are cleared before the new attempt is dispatched. After
    ``max_inflight_reissues`` re-issues a prompt is marked ``failure`` under a fresh uid and the existing
    failure paths (padding or refill) finish the step. That per-prompt cap alone does not bound a step (the
    refill that replaces an exhausted prompt starts at attempt 1), so re-issues are also bounded per step:
    once ``max_inflight_reissues_per_step`` is spent (counted across the sampling calls that share a
    ``global_steps``) the TTL re-issues nothing more in that step and the loop waits for the held-back
    generations as it does without a TTL (``{prefix}/inflight_ttl_suspended`` = 1, one warning per step);
    exhausted prompts are still failed and the DAPO drain still discards. A TTL below the honest generation
    time therefore costs a bounded number of regenerations and then degrades to waiting; it never livelocks.
    Validation prompts are never regenerated: the TTL does not apply to ``val``.
    """

    def __init__(
        self,
        trainer_mode: str,
        trainer_config: DictConfig,
        max_off_policy_threshold: int | None,
        max_off_policy_strategy: str,
        sampler_kwargs: DictConfig,
        poll_interval: float = 2.0,
        refill_fn=None,
        filter_groups_metric: str | None = None,
        train_batch_size: int | None = None,
        gen_batch_size: int | None = None,
        max_inflight_gen_batches: int = 1,
        sync_refill_failed_groups: bool = False,
        inflight_ttl_s: float | None = None,
        max_inflight_reissues: int = 3,
        max_inflight_reissues_per_step: int | None = None,
        reissue_fn=None,
    ):
        self.trainer_mode = trainer_mode
        self.trainer_config = trainer_config
        self.max_off_policy_threshold = max_off_policy_threshold
        self.max_off_policy_strategy = max_off_policy_strategy
        self.sampler_kwargs = sampler_kwargs
        self.poll_interval = poll_interval
        self.refill_fn = refill_fn
        self.filter_groups_metric = filter_groups_metric
        self.train_batch_size = train_batch_size
        self.gen_batch_size = gen_batch_size
        self.max_inflight_gen_batches = max_inflight_gen_batches
        self.sync_refill_failed_groups = sync_refill_failed_groups
        self.inflight_ttl_s = inflight_ttl_s
        self.max_inflight_reissues = max_inflight_reissues
        self.max_inflight_reissues_per_step = max_inflight_reissues_per_step
        self.reissue_fn = reissue_fn

        if self.inflight_ttl_s is not None:
            assert (
                isinstance(self.inflight_ttl_s, int | float)
                and not isinstance(self.inflight_ttl_s, bool)
                and self.inflight_ttl_s > 0
            ), f"Invalid inflight_ttl_s: {self.inflight_ttl_s}, must be a positive number of seconds or null"
        assert (
            isinstance(self.max_inflight_reissues, int)
            and not isinstance(self.max_inflight_reissues, bool)
            and self.max_inflight_reissues >= 0
        ), f"Invalid max_inflight_reissues: {self.max_inflight_reissues}, must be a non-negative integer"
        if self.max_inflight_reissues_per_step is not None:
            assert (
                isinstance(self.max_inflight_reissues_per_step, int)
                and not isinstance(self.max_inflight_reissues_per_step, bool)
                and self.max_inflight_reissues_per_step >= 0
            ), (
                f"Invalid max_inflight_reissues_per_step: {self.max_inflight_reissues_per_step}, must be a "
                "non-negative integer or null"
            )
        if self.inflight_ttl_s is not None and self.reissue_fn is None:
            raise ValueError("inflight_ttl_s requires reissue_fn to re-issue expired in-flight prompts")
        if self.max_off_policy_threshold is not None:
            assert isinstance(self.max_off_policy_threshold, int) and self.max_off_policy_threshold > 0, (
                f"Invalid max off policy threshold: {self.max_off_policy_threshold}, "
                "must be an integer greater than 0, or null to disable off-policy version control"
            )
        else:
            logger.info(
                "[V1ReplayBuffer] off-policy version control disabled (max_off_policy_threshold is null): "
                "no samples will be dropped or awaited for staleness"
            )
        assert self.max_off_policy_strategy in ["drop", "wait"], (
            f"Invalid max off policy strategy: {self.max_off_policy_strategy}, must be one of ['drop', 'wait']"
        )
        if self.filter_groups_metric is not None and self.refill_fn is None:
            raise ValueError("Group filtering (filter_groups_metric) requires refill_fn to replace evicted groups")
        if self.sync_refill_failed_groups and self.refill_fn is None:
            raise ValueError("sync_refill_failed_groups requires refill_fn to replace failed groups")
        self._validate_mode_config()
        # partition_id => {key: tag}
        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)
        self.pending_keys: dict[str, set] = defaultdict(set)
        self.running_keys: dict[str, set] = defaultdict(set)
        self.finished_keys: dict[str, set] = defaultdict(set)
        self.failure_keys: dict[str, set] = defaultdict(set)
        # partition_id => {prompt_key: global_steps}, used to prioritize older samples.
        self.prompt_global_steps: dict[str, dict[str, int]] = defaultdict(dict)
        # partition_id => {prompt_key: wall-clock dispatch time}, used by the in-flight TTL.
        self.prompt_dispatched_at: dict[str, dict[str, float]] = defaultdict(dict)
        # Finished groups are immutable, so their DAPO classification can be reused across polling iterations.
        self._dapo_classification_cache: dict[str, dict[str, float | None]] = defaultdict(dict)
        # partition_id => uids of re-issued attempts whose late writes must be swept (see [In-flight recovery]).
        self._retired_uids: dict[str, set[str]] = defaultdict(set)
        # partition_id => {prompt_key: dispatch attempt (1 = original)}, bounds re-issues per prompt.
        self.prompt_attempt: dict[str, dict[str, int]] = defaultdict(dict)
        # partition_id => the global_steps whose re-issue budget is open / re-issues spent in it / warned once.
        self._ttl_budget_step: dict[str, int] = {}
        self._ttl_reissues_in_step: dict[str, int] = defaultdict(int)
        self._ttl_suspension_warned: set[str] = set()

    def _validate_mode_config(self) -> None:
        if self.filter_groups_metric is not None:
            if not isinstance(self.max_inflight_gen_batches, int) or self.max_inflight_gen_batches <= 0:
                raise ValueError("max_inflight_gen_batches must be a positive integer")
        if self.sync_refill_failed_groups and self.gen_batch_size != 1:
            raise ValueError("sync_refill_failed_groups requires gen_batch_size=1")

    def _sync_metadata_from_transfer_queue(self):
        """Sync the metadata from TransferQueue."""
        self.partitions.clear()
        self.pending_keys.clear()
        self.running_keys.clear()
        self.finished_keys.clear()
        self.failure_keys.clear()
        self.prompt_global_steps.clear()
        self.prompt_dispatched_at.clear()
        self.prompt_attempt.clear()

        data = tq.kv_list()
        if data is None:
            return

        for partition_id, items in data.items():
            partition = self.partitions[partition_id]
            for key, tag in items.items():
                if tag.get("is_prompt", False):
                    # see: [GRPO group sampling control]
                    self.prompt_global_steps[partition_id][key] = tag["global_steps"]
                    if tag.get("dispatched_at") is not None:
                        self.prompt_dispatched_at[partition_id][key] = tag["dispatched_at"]
                    self.prompt_attempt[partition_id][key] = tag.get("attempt", 1)
                    match tag["status"]:
                        case "pending":
                            self.pending_keys[partition_id].add(key)
                        case "running":
                            self.running_keys[partition_id].add(key)
                        case "finished":
                            self.finished_keys[partition_id].add(key)
                        case "failure":
                            self.failure_keys[partition_id].add(key)
                        case _:
                            raise ValueError(f"Unknown status: {tag['status']}")
                else:
                    # see: [Trajectories storage format]
                    if key not in partition:
                        partition[key] = {}
                    partition[key].update(tag)

    @staticmethod
    def _metrics_prefix(partition_id: str) -> str:
        return "training" if partition_id == "train" else "validation"

    def _clear_groups(self, partition_id: str, uids: set[str]) -> None:
        """Remove prompt groups from TransferQueue and the active metadata snapshot."""
        if not uids:
            return

        trajectory_keys = {key for key in self.partitions[partition_id] if key.split("_")[0] in uids}
        tq.kv_clear(
            partition_id=partition_id,
            keys=[*uids, *trajectory_keys],
        )

        # Keep same-poll decisions consistent with tq.
        for key in trajectory_keys:
            del self.partitions[partition_id][key]
        for status_keys in (self.pending_keys, self.running_keys, self.finished_keys, self.failure_keys):
            status_keys[partition_id].difference_update(uids)
        for uid in uids:
            self.prompt_global_steps[partition_id].pop(uid, None)
            self.prompt_dispatched_at[partition_id].pop(uid, None)
            self.prompt_attempt[partition_id].pop(uid, None)
            self._dapo_classification_cache[partition_id].pop(uid, None)

    def _expired_inflight_keys(self, partition_id: str, now: float | None = None) -> set[str]:
        """Training prompts still pending/running whose age since ``dispatched_at`` exceeds ``inflight_ttl_s``.

        The age is dispatch-to-terminal, engine queueing included (``running`` is stamped before the engine
        queue). Prompts without a ``dispatched_at`` tag (older checkpoints, custom submitters) never expire.
        """
        if self.inflight_ttl_s is None or partition_id == "val":
            return set()
        now = time.time() if now is None else now
        dispatched_at = self.prompt_dispatched_at[partition_id]
        return {
            uid
            for uid in self._inflight_uids(partition_id)
            if uid in dispatched_at and now - dispatched_at[uid] > self.inflight_ttl_s
        }

    def _inflight_uids(self, partition_id: str) -> set[str]:
        return self.pending_keys[partition_id] | self.running_keys[partition_id]

    def _registered_prompt_uids(self, partition_id: str) -> set[str]:
        status_sets = (self.pending_keys, self.running_keys, self.finished_keys, self.failure_keys)
        return set().union(*(status_keys[partition_id] for status_keys in status_sets))

    def _sweep_retired_attempts(self, partition_id: str) -> int:
        """Clear any key a retired (re-issued or failed) attempt wrote after it was replaced."""
        retired = self._retired_uids[partition_id]
        if not retired:
            return 0
        late_keys = {key for key in self.partitions[partition_id] if key.split("_")[0] in retired}
        late_keys |= self._registered_prompt_uids(partition_id) & retired
        if late_keys:
            tq.kv_clear(partition_id=partition_id, keys=list(late_keys))
            for key in late_keys:
                self.partitions[partition_id].pop(key, None)
            for status_keys in (self.pending_keys, self.running_keys, self.finished_keys, self.failure_keys):
                status_keys[partition_id].difference_update(late_keys)
        return len(late_keys)

    def _fail_exhausted(self, partition_id: str, uids: set[str]) -> None:
        """Turn groups whose re-issues are exhausted into ``failure`` under fresh uids, so the existing failure
        paths (padding, refill) finish the step; the old attempt is retired and its rows cleared, as for a
        re-issue."""
        tags = [
            {
                "is_prompt": True,
                "status": "failure",
                "global_steps": self.prompt_global_steps[partition_id].get(uid, 0),
                "attempt": self.prompt_attempt[partition_id].get(uid, 1),
            }
            for uid in sorted(uids)
        ]
        tq.kv_batch_put(keys=[uuid.uuid4().hex for _ in tags], partition_id=partition_id, tags=tags)
        self._clear_groups(partition_id, uids)
        self._retired_uids[partition_id] |= uids

    def _reissue_budget(self, batch_size: int) -> int:
        """Re-issues the TTL may trigger within one training step: ``max_inflight_reissues_per_step``, by default
        ``max_inflight_reissues`` x the step's prompt count (``train_batch_size`` when the trainer wired it, else
        the sampling call's ``batch_size``)."""
        if self.max_inflight_reissues_per_step is not None:
            return self.max_inflight_reissues_per_step
        prompts_per_step = self.train_batch_size if self.train_batch_size is not None else batch_size
        return self.max_inflight_reissues * prompts_per_step

    def _begin_ttl_budget(self, partition_id: str, global_steps: int) -> None:
        """Open the step's re-issue accounting: the sampling calls that share a ``global_steps`` share it."""
        if self._ttl_budget_step.get(partition_id) != global_steps:
            self._ttl_budget_step[partition_id] = global_steps
            self._ttl_reissues_in_step[partition_id] = 0
            self._ttl_suspension_warned.discard(partition_id)

    def _within_reissue_budget(
        self, partition_id: str, to_reissue: set[str], eviction_metrics: dict, reissue_budget: int
    ) -> set[str]:
        """The re-issuable expired prompts the step budget still allows (oldest first).

        Beyond it the TTL re-issues nothing more in this step: the held-back prompts are waited for, as without
        a TTL (``{prefix}/inflight_ttl_suspended`` = 1 in every sampling call that holds one back, one warning
        per step). Prompts past ``max_inflight_reissues`` are not subject to it: they are failed regardless.
        """
        allowance = reissue_budget - self._ttl_reissues_in_step[partition_id]
        if len(to_reissue) <= allowance:
            return to_reissue
        eviction_metrics[f"{self._metrics_prefix(partition_id)}/inflight_ttl_suspended"] = 1
        if partition_id not in self._ttl_suspension_warned:
            self._ttl_suspension_warned.add(partition_id)
            logger.warning(
                f"[V1ReplayBuffer] the in-flight TTL re-issued {self._ttl_reissues_in_step[partition_id]} prompts "
                f"in step {self._ttl_budget_step.get(partition_id)} (max_inflight_reissues_per_step="
                f"{reissue_budget}); it re-issues nothing more this step and waits for the "
                f"{len(to_reissue) - max(allowance, 0)} expired generations still in flight (prompts past "
                "max_inflight_reissues are still marked failure)"
            )
        if allowance <= 0:
            return set()
        dispatched_at = self.prompt_dispatched_at[partition_id]
        return set(sorted(to_reissue, key=lambda uid: dispatched_at.get(uid, 0.0))[:allowance])

    def _recover_stuck_inflight(
        self, partition_id: str, eviction_metrics: dict, draining: bool = False, reissue_budget: int | None = None
    ) -> bool:
        """Re-issue or fail in-flight prompts that expired; True when the caller must re-poll.

        Runs on a fresh ``_sync_metadata_from_transfer_queue`` snapshot and re-checks the TransferQueue right
        before acting, so a group that reached a terminal status in the meantime is sampled, never retired.
        Metrics accrue into ``eviction_metrics`` under ``{prefix}/inflight_*``. ``reissue_budget`` caps the
        re-issues the TTL may trigger in the step opened by ``_begin_ttl_budget`` (``None``: no cap); it never
        delays the failure of an exhausted prompt or the drain's discard.
        """
        if self.reissue_fn is None:
            return False
        self._sweep_retired_attempts(partition_id)

        now = time.time()
        stuck = expired = self._expired_inflight_keys(partition_id, now)
        if not stuck:
            return False
        prefix = self._metrics_prefix(partition_id)
        dispatched_at = self.prompt_dispatched_at[partition_id]
        age_max = max((now - dispatched_at[uid] for uid in expired), default=0.0)

        def count(name: str, value: int) -> None:
            eviction_metrics[f"{prefix}/{name}"] = eviction_metrics.get(f"{prefix}/{name}", 0) + value

        # Re-check before acting: a slow-but-alive worker may have finished the group since the snapshot.
        self._sync_metadata_from_transfer_queue()
        stuck &= self._inflight_uids(partition_id)
        if not stuck:
            return True
        if draining:
            # Sync DAPO drain: every in-flight group is discarded as surplus anyway, so do not regenerate.
            self._clear_groups(partition_id, stuck)
            self._retired_uids[partition_id] |= stuck
            count("inflight_discarded_while_draining", len(stuck))
            return True

        attempts = self.prompt_attempt[partition_id]
        exhausted = {uid for uid in stuck if attempts.get(uid, 1) > self.max_inflight_reissues}
        to_reissue = stuck - exhausted
        if to_reissue and reissue_budget is not None:
            to_reissue = self._within_reissue_budget(partition_id, to_reissue, eviction_metrics, reissue_budget)
        if not to_reissue and not exhausted:
            return False  # the step budget holds every expired prompt back: wait, as without a TTL
        reissued: set[str] = set()
        if to_reissue:
            repeats = sorted(uid for uid in to_reissue if attempts.get(uid, 1) >= 2)
            if repeats:
                logger.warning(
                    f"[V1ReplayBuffer] {len(repeats)} prompts are being re-issued again (attempt >= 2), e.g. "
                    f"{repeats[0]}; inflight_ttl_s={self.inflight_ttl_s} must exceed the dispatch-to-terminal time "
                    "including engine queueing, or healthy generations are restarted from scratch"
                )
            logger.warning(
                f"[V1ReplayBuffer] re-issuing {len(to_reissue)} expired in-flight prompts (oldest age: {age_max:.1f}s)"
            )
            result = self.reissue_fn(partition_id, sorted(to_reissue))
            claimed = None if isinstance(result, int) else set(result)
            # Retire only what was actually replaced: a uid still registered turned terminal in the window.
            self._sync_metadata_from_transfer_queue()
            reissued = to_reissue - self._registered_prompt_uids(partition_id)
            if claimed is not None:
                reissued &= claimed
            self._retired_uids[partition_id] |= reissued
            self._ttl_reissues_in_step[partition_id] += len(reissued)
            exhausted &= self._inflight_uids(partition_id)
        if exhausted:
            logger.warning(
                f"[V1ReplayBuffer] {len(exhausted)} prompts exceeded max_inflight_reissues="
                f"{self.max_inflight_reissues}; marking them failure: {sorted(exhausted)[:4]}"
            )
            self._fail_exhausted(partition_id, exhausted)

        count("inflight_reissued", len(reissued))
        count("inflight_reissue_exhausted", len(exhausted))
        if expired & (reissued | exhausted):
            key = f"{prefix}/inflight_expired_age_max"
            eviction_metrics[key] = max(eviction_metrics.get(key, 0.0), age_max)
        return True

    def _dapo_filtered_keys(self, partition_id: str) -> tuple[set[str], Counter]:
        """Finished groups whose configured DAPO metric is identical across all trajectories.

        Returns the filtered uids and a ``{shared_metric_value: group_count}`` breakdown built in the
        same scope, so the diagnostic (which reward level the no-signal groups collapse to) travels
        with the uids through the return value instead of via hidden instance state.
        """
        if partition_id == "val" or self.filter_groups_metric is None:
            return set(), Counter()

        finished_uids = self.finished_keys[partition_id]
        classification_cache = self._dapo_classification_cache[partition_id]
        for uid in classification_cache.keys() - finished_uids:
            del classification_cache[uid]

        new_finished_uids = finished_uids - classification_cache.keys()
        trajectory_keys = [key for key in self.partitions[partition_id] if key.split("_")[0] in new_finished_uids]
        use_canonical_reward = self.filter_groups_metric == FILTER_GROUPS_REWARD_METRIC
        metrics_by_uid: dict[str, list[float]] = defaultdict(list)
        missing_metric_uids = new_finished_uids - {key.split("_")[0] for key in trajectory_keys}

        if trajectory_keys:
            data = tq.kv_batch_get(
                keys=trajectory_keys,
                partition_id=partition_id,
                select_fields=["rm_scores"] if use_canonical_reward else ["extra_fields"],
            )
            metric_data = list(data["rm_scores"] if use_canonical_reward else data["extra_fields"])
        else:
            metric_data = []

        for key, value in zip(trajectory_keys, metric_data, strict=True):
            uid = key.split("_")[0]
            if use_canonical_reward:
                metrics_by_uid[uid].append(float(value.sum().item()))
            else:
                extra_fields = getattr(value, "data", value)
                reward_extra_info = extra_fields.get("reward_extra_info", {}) if isinstance(extra_fields, dict) else {}
                if self.filter_groups_metric not in reward_extra_info:
                    missing_metric_uids.add(uid)
                else:
                    metrics_by_uid[uid].append(float(reward_extra_info[self.filter_groups_metric]))

        if missing_metric_uids:
            raise RuntimeError(
                f"Finished groups are missing DAPO metric {self.filter_groups_metric!r}: "
                f"{sorted(missing_metric_uids)[:5]}"
            )

        for uid in new_finished_uids:
            values = metrics_by_uid[uid]
            classification_cache[uid] = float(values[0]) if len(values) > 1 and float(np.std(values)) == 0.0 else None

        filtered_rewards = {uid: reward for uid, reward in classification_cache.items() if reward is not None}
        return set(filtered_rewards), Counter(filtered_rewards.values())

    def _terminal_eviction_reasons(
        self, global_steps: int, partition_id: str
    ) -> tuple[set[str], set[str], set[str], Counter]:
        """Return stale, DAPO-filtered, and failed groups (plus the DAPO value->count breakdown).

        The three sets may overlap. Callers clear and refill their union, so one prompt is never handled
        twice. ``dapo_counts`` is the {shared_metric_value: group_count} diagnostic for ``dapo_uids``; it
        rides along in the return value so no hidden state is needed between production and consumption.
        """
        if partition_id == "val":
            return set(), set(), set(), Counter()

        dapo_uids, dapo_counts = self._dapo_filtered_keys(partition_id)
        failed_uids = set()
        if self.sync_refill_failed_groups:
            materializable_uids = {key.split("_")[0] for key in self.partitions[partition_id]}
            failed_uids = self.failure_keys[partition_id] - materializable_uids
        return set(), dapo_uids, failed_uids, dapo_counts

    def _sampleable_terminal_keys(
        self,
        partition_id: str,
        eviction_reasons: tuple[set[str], set[str], set[str], Counter],
    ) -> set[str]:
        terminal_uids = self.finished_keys[partition_id] | self.failure_keys[partition_id]
        stale_uids, dapo_uids, failed_uids, _dapo_counts = eviction_reasons
        return terminal_uids - (stale_uids | dapo_uids | failed_uids)

    def _evict_terminal_groups(
        self,
        global_steps: int,
        partition_id: str,
        eviction_reasons: tuple[set[str], set[str], set[str], Counter],
    ) -> tuple[set[str], int, int, dict]:
        """Evict terminal groups selected by any active policy exactly once."""
        stale_uids, dapo_uids, failed_uids, dapo_counts = eviction_reasons
        evicted_uids = stale_uids | dapo_uids | failed_uids
        if not evicted_uids:
            return set(), 0, 0, {}

        prefix = self._metrics_prefix(partition_id)
        metrics: dict = {}
        if stale_uids:
            prompt_global_steps = self.prompt_global_steps[partition_id]
            spans = np.array(
                [global_steps - prompt_global_steps.get(uid, global_steps) + 1 for uid in stale_uids],
                dtype=float,
            )
            metrics.update(
                {
                    f"{prefix}/off_policy/evicted_samples": len(stale_uids),
                    f"{prefix}/off_policy/evicted_samples_staleness/mean": spans.mean(),
                    f"{prefix}/off_policy/evicted_samples_staleness/max": spans.max(),
                    f"{prefix}/off_policy/evicted_samples_staleness/min": spans.min(),
                }
            )
        if dapo_uids:
            metrics[f"{prefix}/filter_groups/evicted_samples"] = len(dapo_uids)
            # Non-scalar diagnostic: how many filtered (no-signal) groups collapsed to each metric value.
            metrics[DAPO_FILTERED_REWARD_COUNTS_KEY] = dict(dapo_counts)
        if failed_uids:
            metrics[f"{prefix}/rollout_failure/evicted_samples"] = len(failed_uids)

        self._clear_groups(partition_id, evicted_uids)
        return evicted_uids, len(stale_uids), len(dapo_uids), metrics

    def _select_prompt_uids(
        self, partition_id: str, sampleable_keys: set[str], batch_size: int
    ) -> tuple[list[str], dict[str, dict], dict[str, int]]:
        prompt_global_steps_snapshot = dict(self.prompt_global_steps[partition_id])
        partition_snapshot = dict(self.partitions[partition_id])
        ordered_keys = sorted(
            sampleable_keys,
            key=lambda key: prompt_global_steps_snapshot.get(key, 0),
        )
        return ordered_keys[:batch_size], partition_snapshot, prompt_global_steps_snapshot

    def _materialize_batch(
        self, partition_id: str, selected_prompt_uids: list[str], partition_snapshot: dict[str, dict]
    ) -> KVBatchMeta:
        tq.kv_clear(partition_id=partition_id, keys=selected_prompt_uids)

        keys, tags = [], []
        selected = set(selected_prompt_uids)
        for key, tag in partition_snapshot.items():
            uid = key.split("_")[0]
            if uid in selected:
                keys.append(key)
                tags.append(tag)
        return KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)

    def _wait_for_next_poll(self, partition_id: str, last_debug_time: float) -> float:
        time.sleep(self.poll_interval)
        now = time.time()
        if now - last_debug_time > VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS:
            logger.info(
                f"pending: {len(self.pending_keys[partition_id])}, "
                f"running: {len(self.running_keys[partition_id])}, "
                f"finished: {len(self.finished_keys[partition_id])}, "
                f"failure: {len(self.failure_keys[partition_id])}"
            )
            return now
        return last_debug_time

    @SkipManager.annotate_tq(role="rollout_tq", phase="sample")
    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> tuple[KVBatchMeta, dict]:
        """Sample a batch using synchronous rollout semantics.

        NOTE: user can customize sampling strategy by setting:
        ```bash
        trainer.v1.sampler.custom_sampler.path = "path/to/your/sampler.py"
        trainer.v1.sampler.custom_sampler.name = "UserCustomReplayBuffer"
        ```

        Args:
            global_steps (int): Global steps of the current training.
            partition_id (str): Partition of TransferQueue, e.g. "train" or "val".
            batch_size (int, optional): Batch size.

        Returns:
            KVBatchMeta: A batch of data.
            dict: Auxiliary metrics.
        """
        last_debug_time = time.time()
        eviction_metrics: dict = {}
        dapo_enabled = partition_id != "val" and self.filter_groups_metric is not None
        refill_credit = 0
        draining = False
        max_inflight_prompts = 0
        if dapo_enabled:
            max_inflight_prompts = self.max_inflight_gen_batches * self.train_batch_size
        reissue_budget = self._reissue_budget(batch_size)
        self._begin_ttl_budget(partition_id, global_steps)

        while True:
            # Eviction, gating, and selection below must all use this snapshot.
            self._sync_metadata_from_transfer_queue()
            if self._recover_stuck_inflight(
                partition_id, eviction_metrics, draining=draining, reissue_budget=reissue_budget
            ):
                continue

            eviction_reasons = self._terminal_eviction_reasons(global_steps, partition_id)
            failed_count = len(eviction_reasons[2])
            evicted_uids, stale_count, dapo_count, metrics = self._evict_terminal_groups(
                global_steps, partition_id, eviction_reasons
            )
            if evicted_uids:
                _accumulate_eviction_metrics(eviction_metrics, metrics, stale_count)

            sampleable_keys = self._sampleable_terminal_keys(partition_id, eviction_reasons)
            has_enough_samples = len(sampleable_keys) >= batch_size
            inflight_count = len(self.pending_keys[partition_id]) + len(self.running_keys[partition_id])

            if not dapo_enabled and failed_count > 0 and not has_enough_samples:
                self.refill_fn(failed_count)
                continue

            if dapo_enabled:
                if has_enough_samples:
                    # Stop speculative dispatch, then drain requests already running under this policy version.
                    draining = True
                    refill_credit = 0
                elif not draining:
                    refill_credit += 2 * dapo_count + failed_count

                if not draining and refill_credit > 0:
                    available_slots = max(0, max_inflight_prompts - inflight_count)
                    dispatch_count = min(refill_credit, available_slots)
                    assert self.gen_batch_size is not None
                    dispatch_count -= dispatch_count % self.gen_batch_size
                    if dispatch_count > 0:
                        assert self.refill_fn is not None
                        self.refill_fn(dispatch_count)
                        refill_credit -= dispatch_count
                        continue

            can_select = has_enough_samples and (not dapo_enabled or inflight_count == 0)
            if can_select:
                selected_prompt_uids, partition_snapshot, _prompt_global_steps_snapshot = self._select_prompt_uids(
                    partition_id, sampleable_keys, batch_size
                )

                # Sync remains bufferless: all speculative requests are drained, then surplus is discarded.
                if dapo_enabled:
                    surplus_uids = sampleable_keys - set(selected_prompt_uids)
                    if surplus_uids:
                        self._clear_groups(partition_id, surplus_uids)
                        key = f"{self._metrics_prefix(partition_id)}/filter_groups/discarded_surplus_samples"
                        eviction_metrics[key] = eviction_metrics.get(key, 0) + len(surplus_uids)
                break

            last_debug_time = self._wait_for_next_poll(partition_id, last_debug_time)

        selected_uids = set(selected_prompt_uids)
        if partition_id != "val" and not any(key.split("_")[0] in selected_uids for key in partition_snapshot):
            message = "Sync replay buffer selected terminal groups with no materializable trajectories."
            if not self.sync_refill_failed_groups:
                message += " Enable trainer.v1.sampler.sync_refill_failed_groups to replace failed groups."
            raise RuntimeError(message)
        return self._materialize_batch(partition_id, selected_prompt_uids, partition_snapshot), eviction_metrics


class ReplayBufferAsync(ReplayBuffer):
    """Async sampling policy over the shared TransferQueue and dynamic-filter implementation."""

    def _validate_mode_config(self) -> None:
        pass

    def _stale_terminal_keys(self, global_steps: int, partition_id: str) -> set[str]:
        if partition_id == "val" or self.max_off_policy_threshold is None or self.max_off_policy_strategy != "drop":
            return set()
        prompt_global_steps = self.prompt_global_steps[partition_id]
        terminal_keys = self.finished_keys[partition_id]
        return {
            uid
            for uid in terminal_keys
            if global_steps - prompt_global_steps.get(uid, global_steps) + 1 > self.max_off_policy_threshold
        }

    def _terminal_eviction_reasons(
        self, global_steps: int, partition_id: str
    ) -> tuple[set[str], set[str], set[str], Counter]:
        if partition_id == "val":
            return set(), set(), set(), Counter()

        stale_uids = self._stale_terminal_keys(global_steps, partition_id)
        dapo_uids, dapo_counts = self._dapo_filtered_keys(partition_id)
        return stale_uids, dapo_uids, set(self.failure_keys[partition_id]), dapo_counts

    def _has_enough_samples(
        self,
        global_steps: int,
        partition_id: str,
        batch_size: int,
        sampleable_keys: set[str],
    ) -> bool:
        # Dropless off-policy control: block sampling while any in-flight prompt has reached the staleness
        # threshold, so it can finish and be trained on instead of dropped.
        if self.max_off_policy_threshold is not None and self.max_off_policy_strategy == "wait":
            for key in self.pending_keys[partition_id] | self.running_keys[partition_id]:
                prompt_global_steps = self.prompt_global_steps[partition_id][key]
                if (global_steps - prompt_global_steps + 1) >= self.max_off_policy_threshold:
                    return False

        return len(sampleable_keys) >= batch_size

    def get_sampleable_count(self, global_steps: int, partition_id: str) -> int:
        """Return the current number of terminal groups eligible for sampling."""
        self._sync_metadata_from_transfer_queue()
        eviction_reasons = self._terminal_eviction_reasons(global_steps, partition_id)
        return len(self._sampleable_terminal_keys(partition_id, eviction_reasons))

    def wait_for_sampleable(self, global_steps: int, partition_id: str, target_count: int) -> tuple[set[str], dict]:
        """Poll until ``target_count`` groups are sampleable, evicting and refilling meanwhile.

        Returns the sampleable uids and the eviction metrics accrued while waiting.
        """
        last_debug_time = time.time()
        eviction_metrics: dict = {}
        reissue_budget = self._reissue_budget(target_count)
        self._begin_ttl_budget(partition_id, global_steps)

        while True:
            # Eviction and selection share one snapshot so newly terminal stale groups wait for the next eviction pass.
            self._sync_metadata_from_transfer_queue()
            if self._recover_stuck_inflight(partition_id, eviction_metrics, reissue_budget=reissue_budget):
                continue

            eviction_reasons = self._terminal_eviction_reasons(global_steps, partition_id)
            evicted_uids, stale_count, _dapo_count, metrics = self._evict_terminal_groups(
                global_steps, partition_id, eviction_reasons
            )
            if evicted_uids:
                _accumulate_eviction_metrics(eviction_metrics, metrics, stale_count)
                if self.refill_fn is not None:
                    self.refill_fn(len(evicted_uids))
                continue

            sampleable_keys = self._sampleable_terminal_keys(partition_id, eviction_reasons)
            if self._has_enough_samples(global_steps, partition_id, target_count, sampleable_keys):
                return sampleable_keys, eviction_metrics

            last_debug_time = self._wait_for_next_poll(partition_id, last_debug_time)

    @SkipManager.annotate_tq(role="rollout_tq", phase="sample")
    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> tuple[KVBatchMeta, dict]:
        """Sample a batch while evicting and replacing stale, DAPO-filtered, or failed groups."""
        sampleable_keys, eviction_metrics = self.wait_for_sampleable(global_steps, partition_id, batch_size)
        selected_prompt_uids, partition_snapshot, prompt_global_steps_snapshot = self._select_prompt_uids(
            partition_id, sampleable_keys, batch_size
        )

        if (
            partition_id != "val"
            and self.max_off_policy_threshold is not None
            and self.max_off_policy_strategy == "drop"
        ):
            selected_spans = [
                global_steps - prompt_global_steps_snapshot.get(uid, global_steps) + 1 for uid in selected_prompt_uids
            ]
            assert all(span <= self.max_off_policy_threshold for span in selected_spans), (
                f"drop strategy selected stale prompts: spans={selected_spans}, "
                f"threshold={self.max_off_policy_threshold}"
            )

        return self._materialize_batch(partition_id, selected_prompt_uids, partition_snapshot), eviction_metrics
