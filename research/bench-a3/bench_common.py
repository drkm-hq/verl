"""Shared helpers for the A3 driver-path micro-benchmarks.

Run every script with:
  PYTHONPATH=<repo>/verl \
  <repo>/.venv-a3/bin/python <script> [args]

Nothing here writes into the clone.
"""

import gc
import platform
import statistics
import sys
import time
import uuid

import numpy as np
import torch


def machine_banner():
    import tensordict

    print(
        f"# machine: {platform.machine()} {platform.platform()} | python {sys.version.split()[0]} | "
        f"torch {torch.__version__} | tensordict {tensordict.__version__} | numpy {np.__version__} | "
        f"torch threads {torch.get_num_threads()}"
    )


def timeit(fn, *, repeat=3, warmup=1, label="", min_repeat_s=0.0):
    """Time fn() `repeat` times after `warmup` runs; print median / min in ms. Returns median seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    print(f"{label:70s} median {med * 1e3:10.1f} ms   min {min(times) * 1e3:10.1f} ms   (n={repeat})")
    return med


def make_lengths(N, P, L, *, seed=0, mean_prompt=None, mean_resp=None, clip_frac=0.05):
    """Realistic per-sample valid lengths: prompts ~ N(mean_prompt, 0.25*P) clipped to [16, P];
    responses: a mixture — most ~ lognormal around mean_resp, `clip_frac` of them hit L (truncated)."""
    rng = np.random.default_rng(seed)
    mean_prompt = mean_prompt or P // 2
    mean_resp = mean_resp or L // 3
    plen = np.clip(rng.normal(mean_prompt, 0.25 * P, N).astype(np.int64), 16, P)
    rlen = np.clip(rng.lognormal(np.log(mean_resp), 0.6, N).astype(np.int64), 1, L)
    clipped = rng.random(N) < clip_frac
    rlen[clipped] = L
    return torch.from_numpy(plen), torch.from_numpy(rlen)


def make_padded_batch(N, P, L, *, seed=0, with_logprob_fields=True, n_rollout=8, dtype_mask=torch.int64):
    """Build a dense left/right padded batch shaped like the agent-loop output after union with
    old_log_probs/ref_log_prob (legacy path), or like `to_padded_tensor()` output in V1.

    Keys: prompts (N,P) int64, responses (N,L) int64, response_mask (N,L), input_ids (N,S), attention_mask (N,S),
    position_ids (N,S), rollout_log_probs (N,L) fp32, rm_scores (N,L) fp32 [+ old_log_probs, ref_log_prob fp32].
    """
    S = P + L
    plen, rlen = make_lengths(N, P, L, seed=seed)
    ar_p = torch.arange(P)
    ar_l = torch.arange(L)
    prompt_mask = (ar_p[None, :] >= (P - plen)[:, None]).to(dtype_mask)  # left padded
    response_mask = (ar_l[None, :] < rlen[:, None]).to(dtype_mask)  # right padded
    attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
    g = torch.Generator().manual_seed(seed)
    prompts = torch.randint(0, 150_000, (N, P), generator=g) * prompt_mask
    responses = torch.randint(0, 150_000, (N, L), generator=g) * response_mask
    input_ids = torch.cat([prompts, responses], dim=1)
    position_ids = (torch.cumsum(attention_mask, dim=1) - 1).clamp(min=0) * attention_mask
    rm_scores = torch.zeros(N, L, dtype=torch.float32)
    scores = (torch.rand(N, generator=g) < 0.4).float()  # binary outcome reward with ~40% success
    rm_scores[torch.arange(N), rlen - 1] = scores
    batch = {
        "prompts": prompts,
        "responses": responses,
        "response_mask": response_mask,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "rollout_log_probs": -torch.rand(N, L, generator=g) * response_mask,
        "rm_scores": rm_scores,
    }
    if with_logprob_fields:
        batch["old_log_probs"] = -torch.rand(N, L, generator=g) * response_mask
        batch["ref_log_prob"] = -torch.rand(N, L, generator=g) * response_mask
    uid = np.array([str(uuid.uuid4()) for _ in range(N // n_rollout)], dtype=object)
    uid = np.repeat(uid, n_rollout)
    return batch, uid, plen, rlen


def make_non_tensor_batch(N, uid, *, n_rollout=8, seed=0):
    """Non-tensor keys shaped like what the agent loop echoes back in legacy mode
    (uid, data_source, reward_model, extra_info, raw_prompt, index, tools_kwargs, interaction_kwargs,
    agent_name, priority, __num_turns__, acc, turn_scores, tool_rewards, min/max_global_steps, extras)."""
    rng = np.random.default_rng(seed)
    B = N // n_rollout
    raw_prompt = np.empty(N, dtype=object)
    reward_model = np.empty(N, dtype=object)
    extra_info = np.empty(N, dtype=object)
    tools_kwargs = np.empty(N, dtype=object)
    interaction_kwargs = np.empty(N, dtype=object)
    turn_scores = np.empty(N, dtype=object)
    tool_rewards = np.empty(N, dtype=object)
    extras = np.empty(N, dtype=object)
    for i in range(N):
        b = i // n_rollout
        raw_prompt[i] = [
            {"role": "system", "content": "You are a helpful math assistant. Think step by step."},
            {"role": "user", "content": f"Problem {b}: compute the integral of x^{b % 7} dx from 0 to {b % 13}. " * 8},
        ]
        reward_model[i] = {"style": "rule", "ground_truth": str(b * 17 % 1000)}
        extra_info[i] = {"index": b, "split": "train", "answer": str(b * 17 % 1000), "need_tools_kwargs": False}
        tools_kwargs[i] = {}
        interaction_kwargs[i] = {}
        turn_scores[i] = []
        tool_rewards[i] = []
        extras[i] = {}
    return {
        "uid": uid,
        "data_source": np.array(["openai/gsm8k"] * N, dtype=object),
        "reward_model": reward_model,
        "extra_info": extra_info,
        "raw_prompt": raw_prompt,
        "index": np.array([i // n_rollout for i in range(N)], dtype=object),
        "tools_kwargs": tools_kwargs,
        "interaction_kwargs": interaction_kwargs,
        "agent_name": np.array(["single_turn_agent"] * N, dtype=object),
        "priority": np.arange(N, dtype=np.int64),
        "__num_turns__": np.ones(N, dtype=np.int32),
        "acc": rng.random(N),
        "turn_scores": turn_scores,
        "tool_rewards": tool_rewards,
        "min_global_steps": np.array([1] * N, dtype=object),
        "max_global_steps": np.array([1] * N, dtype=object),
        "extras": extras,
    }


def nbytes(d):
    tot = 0
    for v in d.values():
        if isinstance(v, torch.Tensor):
            tot += v.numel() * v.element_size()
    return tot


def gb(x):
    return f"{x / 1e9:.2f} GB"
