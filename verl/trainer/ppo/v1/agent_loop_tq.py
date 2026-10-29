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

# TODO: move this file to verl.experimental.agent_loop after V1 is stable
"""TransferQueue adapter for AgentLoopManager and AgentLoopWorker"""

import asyncio
import logging
import os
from collections import Counter
from typing import Any
from uuid import uuid4

import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# How long a liveness probe may wait for a busy worker before it is assumed alive (only an actor error is "dead").
WORKER_LIVENESS_PROBE_TIMEOUT_SECONDS = float(os.getenv("VERL_WORKER_LIVENESS_PROBE_TIMEOUT_SECONDS", "2"))
# Consecutive ActorUnavailableError results (transient by Ray's definition) before a worker counts as dead.
WORKER_UNAVAILABLE_PROBES_BEFORE_DEAD = int(os.getenv("VERL_WORKER_UNAVAILABLE_PROBES_BEFORE_DEAD", "3"))
# How long the second-opinion ping after an unavailable dispatch reply may wait before the worker counts as busy.
DISPATCH_RETRY_INTERVAL_SECONDS = float(os.getenv("VERL_DISPATCH_RETRY_INTERVAL_SECONDS", "1"))
# Re-creations of one worker index without a served dispatch in between before the manager gives up.
MAX_WORKER_RELAUNCHES = int(os.getenv("VERL_MAX_WORKER_RELAUNCHES", "5"))


def _chunk_uids(chunk: TensorDict) -> list[str]:
    return [str(uid) for uid in chunk["uid"]] if "uid" in chunk.keys() else []


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    # Greedy REMAX baselines are used for their rewards, not score-centering training heads.
    params.pop("topk_log_probs", None)
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


async def _settle_session_tasks(tasks: list[asyncio.Task[Any]]) -> list[BaseException]:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if isinstance(result, BaseException)]


@ray.remote
class AgentLoopWorkerTQ(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks = set()

    async def ping(self) -> bool:
        """Liveness probe used by :meth:`AgentLoopManagerTQ.reap_dead_workers`."""
        return True

    async def generate_sequences(self, batch: TensorDict) -> None:
        """Spawn agent loop for each sample in the batch without waiting for the results."""
        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )
        # the trainer never consumes the sampler head on validation rollouts
        if config.topk_log_probs and not validate:
            sampling_params["topk_log_probs"] = config.topk_log_probs

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch:
            default_agent_loop = config.agent.default_agent_loop
            batch["agent_name"] = NonTensorData(default_agent_loop)

        trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)

        # create background tasks for each sample in the batch
        for i in range(len(batch)):
            # TODO(wuxibin): add trace support
            trace_this_sample = False
            prompt = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[i]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[i].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    logger.exception(f"Unsupported type {type(v)} for key {k}")

            # “fire-and-forget” background tasks
            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=trace_this_sample)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        """Spawn multiple agent loops in parallel according to rollout.n or rollout.val_kwargs.n."""
        uid, partition_id = prompt["uid"], "train" if not trajectory["validate"] else "val"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        tasks = []
        try:
            # NOTE: user can dynamically adjust n for each sample here, e.g according to task difficulty.
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)
            do_sample = prompt.pop("__do_sample__", True)

            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                apply_greedy_sampling_params(run_sampling_params)

            tasks = []
            for i in range(n):
                task = asyncio.create_task(
                    self._run_agent_loop(
                        run_sampling_params, trajectory=trajectory, trace=trace, session_id=i, **prompt
                    )
                )
                tasks.append(task)

            # Publish a terminal status only after every session settles, so no sibling can write after
            # ReplayBuffer clears a failed group.
            session_errors = await _settle_session_tasks(tasks)
            if session_errors:
                for error in session_errors:
                    logger.error(
                        f"Error in _run_prompt for uid={uid}",
                        exc_info=(type(error), error, error.__traceback__),
                    )
                status = "failure"
            else:
                status = "finished"
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": status})
        except Exception as e:
            logger.exception(f"Error in _run_prompt: {e}")
            if tasks:
                await _settle_session_tasks(tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        """Put agent loop outputs into TransferQueue."""
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        # TODO: Support output:list[AgentLoopOutput]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                output.reward_score = final_output.reward_score
                output.extra_fields["reward_extra_info"] = final_output.extra_fields["reward_extra_info"]

        # NOTE: agent loop may has multiple outputs, put each output into TransferQueue.
        # key format: {uid}_{session_id}_{index}
        # - uid: raw prompt uid from dataset
        # - session_id: session id for rollout.n sampling
        # - index: index of agent loop output
        keys, fields, tags = [], [], []
        for i, output in enumerate(outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            keys.append(f"{uid}_{session_id}_{i}")
            field = output.as_dict()
            field.update(kwargs)
            # do not store raw image/video
            field.pop("multi_modal_data", None)
            # TODO: uniform response_mask and loss_mask
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                    # These tags are used for off-policy staleness control, if a trajectory
                    # spans too many global steps, we need to filter it out.
                    # global_steps: which global steps this sample is from dataloader
                    "global_steps": kwargs["global_steps"],
                    # min_global_steps: start generation model weights version of this trajectory
                    "min_global_steps": field["extra_fields"].get("min_global_steps"),
                    # max_global_steps: end generation model weights version of this trajectory
                    "max_global_steps": field["extra_fields"].get("max_global_steps"),
                }
            )

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )


class AgentLoopManagerTQ(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = AgentLoopWorkerTQ
        super().__init__(*args, **kwargs)
        self.relaunch_dead_workers = bool(self.rollout_config.agent.get("relaunch_dead_workers", False))
        # Rotating start index: small refill / re-issue batches must not always land on worker 0.
        self._dispatch_cursor = 0
        # uid -> index of the worker that received its latest dispatch. Kept only with relaunch_dead_workers on;
        # pruned to the in-flight set on every reap, so it is bounded by the number of prompts in flight.
        self._prompt_owner: dict[str, int] = {}
        # uids a dead worker left in flight, handed to the replay buffer by partition as it polls.
        self._orphaned_uids: set[str] = set()
        # index -> re-creations since the worker last served a dispatch / consecutive unavailable probes.
        self._relaunch_streak: Counter = Counter()
        self._unavailable_streak: Counter = Counter()
        # Recovery counters (``rollout/...``), drained by the trainer once per training step and, re-keyed as
        # ``validation/...``, by ``_validate`` (see PPOTrainer._consume_recovery_metrics).
        self.recovery_metrics: Counter = Counter()

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    async def _init_agent_loop_workers(self):
        num_workers = self.rollout_config.agent.num_workers
        node_ids = self._alive_cpu_node_ids()
        self.agent_loop_workers = [self._launch_worker(i, node_ids) for i in range(num_workers)]

    @staticmethod
    def _alive_cpu_node_ids() -> list[str]:
        return [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]

    def _launch_worker(self, index: int, node_ids: list[str] | None = None) -> ray.actor.ActorHandle:
        """Create agent loop worker ``index`` with the manager's construction arguments.

        Same placement rule as ``AgentLoopManager``: round-robin over the alive CPU nodes with soft affinity.
        Re-creating a dead worker goes through the same path, so one whose node vanished lands elsewhere.
        """
        if node_ids is None:
            node_ids = self._alive_cpu_node_ids()
        return self.agent_loop_workers_class.options(
            name=f"agent_loop_worker_{index}" + f"_{uuid4().hex[:8]}",
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_ids[index % len(node_ids)], soft=True
            ),
        ).remote(
            self.config,
            self.llm_client,
            self.teacher_client,
            self.reward_loop_worker_handles,
        )

    def generate_sequences(self, prompts: TensorDict) -> None:
        """
        Dispatch input batch to agent loop workers without blocking. Workers should put agent loop outputs
        into TransferQueue once an agent loop finished.

        Chunks are assigned from a rotating start index so that a batch smaller than the worker count (refill,
        re-issue) spreads over the workers instead of always hitting the lowest indices. A dispatch whose reply
        is a ``RayActorError`` is never sent again (the call may have executed): with
        ``rollout.agent.relaunch_dead_workers`` the chunk's prompts are parked for the replay buffer to re-issue
        under fresh uids and a dead worker is re-created, otherwise the error is logged and re-raised as is.

        Args:
            prompts (TensorDict): Input batch from train or validation dataset.
        """
        num_workers = len(self.agent_loop_workers)
        chunks = prompts.chunk(num_workers)
        start = self._dispatch_cursor
        self._dispatch_cursor = (start + len(chunks)) % num_workers
        dispatches = []
        for i, chunk in enumerate(chunks):
            index = (start + i) % num_workers
            dispatches.append((index, chunk, self.agent_loop_workers[index].generate_sequences.remote(chunk)))
        for index, chunk, ref in dispatches:
            try:
                ray.get(ref)
            except ray.exceptions.RayActorError as error:
                self._recover_dispatch(index, chunk, error)
                continue
            self._relaunch_streak[index] = 0
            self._unavailable_streak[index] = 0
            if self.relaunch_dead_workers:
                for uid in _chunk_uids(chunk):
                    self._prompt_owner[uid] = index

    def _recover_dispatch(self, index: int, chunk: TensorDict, error: ray.exceptions.RayActorError) -> None:
        """Handle a dispatch whose reply was an actor error. The chunk is never sent a second time.

        Ray cannot say whether the call executed, so a re-send could start a second live attempt of the same
        uids on the same keys. Instead the chunk's prompts stay ``pending``/``running`` in TransferQueue and are
        parked in ``_orphaned_uids``: the replay buffer collects them through :meth:`reap_dead_workers` and
        re-issues them under fresh uids (old rows cleared), or drops them from the parked set once they reach a
        terminal status on their own. A dead worker is re-created; a merely unavailable one is marked suspect.
        """
        uids = _chunk_uids(chunk)
        if not self.relaunch_dead_workers:
            logger.error(
                f"Agent loop worker {index} failed a dispatch of {len(uids)} prompts "
                f"(uids {uids[:4]}{'...' if len(uids) > 4 else ''}, {type(error).__name__}); their TransferQueue "
                "status stays non-terminal. Set actor_rollout_ref.rollout.agent.relaunch_dead_workers=True to "
                "re-create dead workers and trainer.v1.sampler.inflight_ttl_s to re-issue the prompts they left."
            )
            raise error
        if isinstance(error, ray.exceptions.ActorUnavailableError) and not self._confirm_dead_after_unavailable(index):
            logger.warning(
                f"Agent loop worker {index} was unavailable for a dispatch of {len(uids)} prompts; they are parked "
                "for re-issue and the worker is suspect"
            )
        else:
            self._orphan_worker(index, error)
        self._orphaned_uids.update(uids)
        self.recovery_metrics["rollout/prompts_orphaned_at_dispatch"] += len(uids)

    def _confirm_dead_after_unavailable(self, index: int) -> bool:
        """Second opinion after an unavailable dispatch reply: one ping bounded by DISPATCH_RETRY_INTERVAL_SECONDS.

        A ping that answers resets the unavailable streak; one that times out counts as busy; an actor error
        goes through the same consecutive-failure rule as the periodic probe.
        """
        self._unavailable_streak[index] += 1
        ref = self.agent_loop_workers[index].ping.remote()
        ready, _ = ray.wait([ref], timeout=DISPATCH_RETRY_INTERVAL_SECONDS)
        if not ready:
            return False
        try:
            ray.get(ref)
        except ray.exceptions.RayActorError as probe_error:
            return self._probe_marks_dead(index, probe_error)
        self._unavailable_streak[index] = 0
        return False

    def _orphan_worker(self, index: int, error: BaseException) -> None:
        """Park every prompt worker ``index`` still owns and re-create it (when relaunching is enabled)."""
        owned = {uid for uid, owner in self._prompt_owner.items() if owner == index}
        for uid in owned:
            del self._prompt_owner[uid]
        self._orphaned_uids |= owned
        self.recovery_metrics["rollout/prompts_orphaned_by_dead_workers"] += len(owned)
        logger.warning(f"Agent loop worker {index} died with {len(owned)} prompts in flight: {sorted(owned)[:4]}")
        if self.relaunch_dead_workers:
            self._relaunch_worker(index, error)

    def _relaunch_worker(self, index: int, error: BaseException) -> None:
        """Replace worker ``index`` with a fresh actor; give up loudly after ``MAX_WORKER_RELAUNCHES`` in a row."""
        self._relaunch_streak[index] += 1
        if self._relaunch_streak[index] > MAX_WORKER_RELAUNCHES:
            raise RuntimeError(
                f"Agent loop worker {index} was re-created {MAX_WORKER_RELAUNCHES} times without serving a "
                f"dispatch; giving up (last error: {error!r})"
            ) from error
        old_worker = self.agent_loop_workers[index]
        logger.warning(
            f"Agent loop worker {index} is dead ({type(error).__name__}); re-creating it "
            f"(attempt {self._relaunch_streak[index]}/{MAX_WORKER_RELAUNCHES}): {error}"
        )
        self.agent_loop_workers[index] = self._launch_worker(index)
        try:
            # Best effort: an "unavailable" actor must not come back and write statuses for retired attempts.
            ray.kill(old_worker, no_restart=True)
        except Exception:  # noqa: BLE001 - the handle may already be gone
            pass
        self.recovery_metrics["rollout/dead_workers_relaunched"] += 1

    def _probe_marks_dead(self, index: int, error: ray.exceptions.RayActorError) -> bool:
        """``ActorDiedError`` is final; ``ActorUnavailableError`` must repeat on consecutive probes to count."""
        if isinstance(error, ray.exceptions.ActorUnavailableError):
            self._unavailable_streak[index] += 1
            if self._unavailable_streak[index] < WORKER_UNAVAILABLE_PROBES_BEFORE_DEAD:
                logger.warning(
                    f"Agent loop worker {index} unavailable ({self._unavailable_streak[index]}/"
                    f"{WORKER_UNAVAILABLE_PROBES_BEFORE_DEAD} consecutive probes): {error}"
                )
                return False
        self._unavailable_streak[index] = 0
        return True

    def reap_dead_workers(self, inflight_uids: set[str]) -> set[str]:
        """Probe every worker and return the in-flight uids whose owner died (accumulated until consumed).

        A probe that merely times out is a busy, live worker. Dead workers are re-created when
        ``relaunch_dead_workers`` is on. The caller (the replay buffer) re-issues or fails the returned uids
        of the partition it is polling, since the dead worker can never write their terminal status. The set
        also carries chunks parked by a failed dispatch; prompts that reached a terminal status on their own
        leave it at the next call.

        Args:
            inflight_uids: prompt uids pending/running across every partition, used to prune bookkeeping.
        """
        refs = [worker.ping.remote() for worker in self.agent_loop_workers]
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=WORKER_LIVENESS_PROBE_TIMEOUT_SECONDS)
        dead: list[tuple[int, BaseException]] = []
        for index, ref in enumerate(refs):
            if ref not in ready:
                continue
            try:
                ray.get(ref)
                self._unavailable_streak[index] = 0
            except ray.exceptions.RayActorError as error:
                if self._probe_marks_dead(index, error):
                    dead.append((index, error))

        self._prompt_owner = {uid: owner for uid, owner in self._prompt_owner.items() if uid in inflight_uids}
        self._orphaned_uids &= inflight_uids
        for index, error in dead:
            self._orphan_worker(index, error)
        return set(self._orphaned_uids)
