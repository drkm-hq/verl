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
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Callable

import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.ray_utils import get_event_loop

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


RawRewardFn = Callable[..., Any] | None

# types of non-mapping ``extra_info`` values already reported by extra_info_for_reward_fn (one warning each)
_NON_MAPPING_EXTRA_INFO_WARNED: set[str] = set()


class RewardManagerBase(ABC):
    _class_initialized = False

    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer, compute_score: RawRewardFn):
        """Initialize reward manager.

        Args:
            config (DictConfig): YAML config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.compute_score = compute_score
        self.loop = get_event_loop()
        self.init_class(config, tokenizer)

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        """Initialize class state shared across all instances."""
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run_single(self, data: DataProto):
        raise NotImplementedError

    @staticmethod
    def extra_info_for_reward_fn(data_item) -> Any:
        """Build the ``extra_info`` handed to the reward function for one sample.

        A copy of the dataset's ``extra_info`` (the batch's own dict is never mutated) merged with the
        agent loop's ``tool_extra_fields`` (``turn_scores``, ``tool_rewards`` and, under
        ``rollout.multi_turn.turn_level_rewards``, ``generated_turn_spans`` / ``turn_rewards`` /
        ``episode_done``; list values are copied, so the reward function cannot edit the row's metadata),
        plus ``num_turns`` and ``rollout_reward_scores``. Every reward manager builds its ``extra_info``
        here, so a custom reward function sees the same keys whichever manager runs it. A dataset whose
        ``extra_info`` is not a mapping (a string, say) is handed over unchanged, without the merge, with
        one warning per type.
        """
        dataset_extra_info = data_item.non_tensor_batch.get("extra_info", None)
        if dataset_extra_info is not None and not isinstance(dataset_extra_info, Mapping):
            type_name = type(dataset_extra_info).__name__
            if type_name not in _NON_MAPPING_EXTRA_INFO_WARNED:
                _NON_MAPPING_EXTRA_INFO_WARNED.add(type_name)
                logger.warning(
                    "extra_info is a %s, not a mapping: handed to the reward function unchanged, without the "
                    "agent loop's tool metadata (generated_turn_spans / turn_rewards / episode_done / num_turns)",
                    type_name,
                )
            return dataset_extra_info
        extra_info = dict(dataset_extra_info or {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update((k, list(v) if isinstance(v, list) else v) for k, v in tool_extra_fields.items())
        extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)
        extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})
        return extra_info

    @classmethod
    def assemble_rm_scores(cls, data: DataProto, scores: list[float]) -> torch.Tensor:
        """Assemble per-sample reward scores into the ``rm_scores`` tensor for a batch.

        Args:
            data: The concatenated batch passed to :meth:`run_single`.
                ``data.batch["prompts"]``, ``data.batch["responses"]`` and
                ``data.batch["attention_mask"]`` are expected to be present
                for the default implementation.
            scores: List of scalar reward scores, one per sample in ``data``.

        Returns:
            torch.Tensor: The ``rm_scores`` tensor with leading dimension equal to
            ``len(data)``.
        """
        prompt_length = data.batch["prompts"].size(1)
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1)
        rm_scores = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        rm_scores[torch.arange(rm_scores.size(0), device=rm_scores.device), valid_response_length - 1] = (
            rm_scores.new_tensor(scores)
        )
        return rm_scores
