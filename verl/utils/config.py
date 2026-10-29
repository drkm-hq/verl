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

from dataclasses import is_dataclass
from typing import Any, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["omega_conf_to_dataclass", "validate_config"]


def omega_conf_to_dataclass(config: DictConfig | dict, dataclass_type: Optional[type[Any]] = None) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: The dataclass type to convert to. When dataclass_type is None,
            the DictConfig must contain _target_ to be instantiated via hydra.instantiate API.

    Returns:
        The dataclass instance.
    """
    # Got an empty config
    if not config:
        return dataclass_type if dataclass_type is None else dataclass_type()
    # Got an object
    if not isinstance(config, DictConfig | ListConfig | dict | list):
        return config

    if dataclass_type is None:
        assert "_target_" in config, (
            "When dataclass_type is not provided, config must contain _target_. "
            "See trainer/config/ppo_trainer.yaml algorithm section for an example. "
            f"Got config: {config}"
        )
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")
    cfg = OmegaConf.create(config)  # in case it's a dict
    # pop _target_ to avoid hydra instantiate error, as most dataclass do not have _target_
    # Updated (vermouth1992) We add _target_ to BaseConfig so that it is compatible.
    # Otherwise, this code path can't support recursive instantiation.
    # if "_target_" in cfg:
    #     cfg.pop("_target_")
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    # let cfg override the existing vals in `cfg_from_dataclass`
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    # now convert to `dataclass_type`
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object


def update_dict_with_config(dictionary: dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)


def _validate_router_replay_config(actor_config: Any, rollout_correction: Any) -> None:
    engine_config = getattr(actor_config, "engine", None)
    router_replay = getattr(engine_config, "router_replay", None)
    if (
        getattr(router_replay, "mode", "disabled") == "R2"
        and rollout_correction
        and rollout_correction.get("bypass_mode", False)
    ):
        raise ValueError(
            "router_replay.mode='R2' requires algorithm.rollout_correction.bypass_mode=False: "
            "R2 records routes while recomputing actor old_log_probs, but bypass mode skips that forward."
        )


# HF config fields that switch on dropout somewhere in a decoder (the update pass runs in train mode)
_HF_DROPOUT_KEYS = (
    "attention_dropout",
    "hidden_dropout",
    "dropout",
    "resid_pdrop",
    "embd_pdrop",
    "attn_pdrop",
    "hidden_dropout_prob",
    "attention_probs_dropout_prob",
    "activation_dropout",
    "ffn_dropout",
)


def _hf_config_dropouts(model_config: DictConfig) -> Optional[list[tuple[str, float]]]:
    """The dropout rates > 0 set in the actor's HF config (top level and ``text_config``), or None when the
    config cannot be loaded here (missing path, offline hub, custom loader)."""
    path = model_config.get("hf_config_path", None) or model_config.get("path", None)
    if not path:
        return None
    try:
        from transformers import AutoConfig

        from verl.utils.fs import copy_to_local

        hf_config = AutoConfig.from_pretrained(
            copy_to_local(path), trust_remote_code=model_config.get("trust_remote_code", False)
        )
    except Exception:
        return None
    found = []
    for cfg in (hf_config, getattr(hf_config, "text_config", None)):
        for key in _HF_DROPOUT_KEYS:
            value = getattr(cfg, key, None) if cfg is not None else None
            if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
                found.append((key, float(value)))
    return found


def _validate_skip_old_log_prob_config(config: DictConfig) -> None:
    """``actor.skip_old_log_prob_when_on_policy`` is exact only when the whole sampled batch is updated in one
    optimizer step at the sampling weights and nothing before the update reads ``old_log_probs``. Reject every
    configuration where that does not hold, naming all violated conditions; there is no silent fallback."""
    actor = config.actor_rollout_ref.actor
    if not actor.get("skip_old_log_prob_when_on_policy", False):
        return
    trainer = config.trainer
    v1 = trainer.get("v1", None) or {}
    trainer_mode = v1.get("trainer_mode", None)
    parameter_sync_step = (v1.get(trainer_mode, None) or {}).get("parameter_sync_step", 1) if trainer_mode else 1
    sample_batch_size = config.data.train_batch_size // max(int(parameter_sync_step), 1)
    algorithm = config.algorithm
    distillation = config.get("distillation", None)

    violations = []
    if not trainer.get("use_v1", True):
        violations.append("trainer.use_v1=False (the legacy trainer does not implement the skip)")
    strategy = actor.get("strategy", None)
    if strategy not in ("fsdp", "fsdp2"):
        violations.append(
            f"actor.strategy={strategy} (the skip is implemented and verified for the FSDP engines only; Megatron, "
            "VeOmni, TorchTitan and automodel keep the old-log-prob pass: their update-pass output path is "
            "untested, Megatron's update-pass entropy would add a (T, V) logits copy and its dynamic context "
            "parallel drops the training outputs)"
        )
    dropouts = _hf_config_dropouts(config.actor_rollout_ref.model)
    if dropouts is None:
        print(
            "WARNING: actor_rollout_ref.actor.skip_old_log_prob_when_on_policy=True could not load the model's HF "
            "config to check for dropout; the skip stores the update pass's train-mode log-probs as old_log_probs, "
            "which is exact only for a dropout-free model."
        )
    elif dropouts:
        violations.append(
            "the model's HF config enables dropout ("
            + ", ".join(f"{key}={value}" for key, value in dropouts)
            + "): the update pass runs in train mode, so its log-probs would be dropout-sampled instead of the "
            "eval-mode old log-probs"
        )
    if trainer_mode == "separate_async":
        violations.append(
            "trainer.v1.trainer_mode=separate_async (it restores pi_old from a CPU snapshot, so the update "
            "pass does not run at the sampling weights)"
        )
    if actor.ppo_epochs != 1:
        violations.append(
            f"actor.ppo_epochs={actor.ppo_epochs} (needs 1: the second epoch runs after an optimizer step)"
        )
    if actor.ppo_mini_batch_size != sample_batch_size:
        violations.append(
            f"actor.ppo_mini_batch_size={actor.ppo_mini_batch_size} != data.train_batch_size / parameter_sync_step "
            f"= {sample_batch_size} (the sampled batch must be one mini-batch, i.e. one optimizer step; only the "
            "first mini-batch is evaluated at the sampling weights)"
        )
    if actor.get("shuffle", False):
        violations.append("actor.shuffle=True (the returned log-probs must stay row-aligned with the batch)")
    if actor.get("calculate_sum_pi_squared", False):
        violations.append("actor.calculate_sum_pi_squared=True (the variance proxy reads old_log_probs)")
    if algorithm.get("use_kl_in_reward", False):
        violations.append(
            "algorithm.use_kl_in_reward=True (the reward KL penalty reads old_log_probs before the update)"
        )
    rollout_corr = algorithm.get("rollout_correction", None) or {}
    if (
        rollout_corr.get("bypass_mode", False)
        or rollout_corr.get("rollout_is", None) is not None
        or rollout_corr.get("rollout_rs", None) is not None
    ):
        violations.append(
            "algorithm.rollout_correction has bypass_mode, rollout_is or rollout_rs set (bypass mode already avoids "
            "the pass with a different estimator; IS weights and rejection sampling read old_log_probs before the "
            "update); only the metrics-only default (rollout_is=null, rollout_rs=null, bypass_mode=false) is compatible"
        )
    router_replay = (actor.get("engine", None) or {}).get("router_replay", None) or {}
    if router_replay.get("mode", "disabled") == "R2":
        violations.append("actor.engine.router_replay.mode=R2 (it records expert routes during the old-log-prob pass)")
    if algorithm.adv_estimator in ("optimal_token_baseline", "tir_optimal_token_baseline"):
        violations.append(
            f"algorithm.adv_estimator={algorithm.adv_estimator} (it reads old_log_probs before the update)"
        )
    if distillation is not None and distillation.get("enabled", False):
        violations.append("distillation.enabled=True (the distillation losses read old_log_probs)")
    if violations:
        raise ValueError(
            "actor_rollout_ref.actor.skip_old_log_prob_when_on_policy=True requires a strictly on-policy update, "
            "but: " + "; ".join(violations)
        )


def _validate_score_centering_config(config: DictConfig) -> None:
    algorithm_rc = config.algorithm.get("rollout_correction") or {}
    actor_rc = config.actor_rollout_ref.actor.policy_loss.get("rollout_correction") or {}
    if not (algorithm_rc.get("score_centering", False) or actor_rc.get("score_centering", False)):
        return
    if not (algorithm_rc.get("score_centering", False) and actor_rc.get("score_centering", False)):
        raise ValueError(
            "score centering must be enabled on both algorithm.rollout_correction and "
            "actor_rollout_ref.actor.policy_loss.rollout_correction."
        )
    from verl.trainer.config.algorithm import RolloutCorrectionConfig

    # the actor-side mirror is a plain mapping, so run the dataclass checks on it here
    RolloutCorrectionConfig(**actor_rc)
    for key in ("bypass_mode", "loss_type", "rollout_is", "rollout_is_threshold"):
        if actor_rc.get(key) != algorithm_rc.get(key):
            raise ValueError(
                f"score centering requires actor_rollout_ref.actor.policy_loss.rollout_correction.{key} to match "
                f"algorithm.rollout_correction.{key}."
            )
    if not config.trainer.get("use_v1", True):
        raise ValueError("score centering requires the v1 trainer (trainer.use_v1=True).")
    if config.actor_rollout_ref.actor.get("strategy") not in ("fsdp", "fsdp2"):
        raise ValueError("score centering requires actor_rollout_ref.actor.strategy=fsdp or fsdp2.")
    if config.actor_rollout_ref.rollout.get("name") != "vllm":
        raise ValueError("score centering requires actor_rollout_ref.rollout.name=vllm.")
    if not config.actor_rollout_ref.rollout.get("topk_log_probs", 0):
        raise ValueError("score centering requires actor_rollout_ref.rollout.topk_log_probs > 0.")
    if config.actor_rollout_ref.rollout.get("temperature", 1.0) <= 0:
        raise ValueError("score centering requires stochastic training rollouts with temperature > 0.")
    if (config.get("distillation") or {}).get("enabled", False):
        raise ValueError("score centering cannot be combined with distillation.")


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
    """
    # number of GPUs total
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

    # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
    # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        settings = {
            "actor_rollout_ref.ref": "log_prob_micro_batch_size",
            "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
        }

        if name in settings:
            param = settings[name]
            param_per_gpu = f"{param}_per_gpu"

            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                    f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                )

    # Actor validation done in ActorConfig.__post_init__ and validate()
    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)
    _validate_router_replay_config(actor_config, config.algorithm.get("rollout_correction", None))
    _validate_score_centering_config(config)
    _validate_skip_old_log_prob_config(config)

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if use_reference_policy:
            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )

        #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
        check_mutually_exclusive(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.rollout",
        )

    if config.algorithm.get("use_kl_in_reward", False) and config.actor_rollout_ref.actor.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")

    # critic
    if use_critic:
        critic_config = omega_conf_to_dataclass(config.critic)
        critic_config.validate(n_gpus, config.data.train_batch_size)

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated."
            + " Validation datasets are sent to inference engines as a whole batch,"
            + " which will schedule the memory themselves."
        )

    # check eval config
    if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
        assert config.actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    # check LoRA rank in vLLM
    lora_config = config.actor_rollout_ref.model.get("lora", {})
    lora_rank = lora_config.get("rank", 0)
    if lora_rank <= 0:
        lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
    if lora_config.get("merge", False):
        lora_rank = 0
    if lora_rank > 0 and config.actor_rollout_ref.rollout.name == "vllm":
        from verl.workers.rollout.vllm_rollout.utils import get_vllm_max_lora_rank

        get_vllm_max_lora_rank(lora_rank)

    print("[validate_config] All configuration checks passed successfully!")
