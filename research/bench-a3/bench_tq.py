"""Live local TransferQueue (SimpleStorage, 2 units, local Ray) — the V1 driver's per-step kv_batch_get/put pattern.

Reproduces, with the same field sets, what verl/trainer/ppo/v1/trainer_base.py does on the driver per step:
  _compute_old_log_prob  : get [entropy, log_probs, response_mask] ; put [old_log_probs, entropy]      (1641-1669)
  _compute_ref_log_prob  : get [log_probs, response_mask]          ; put [ref_log_prob]               (1713-1718)
  _compute_values        : get [values, response_mask]             ; put [values]                     (1734-1740)
  _compute_advantage     : get [uid, response_mask, rm_scores, rollout_log_probs, old_log_probs, ref_log_prob, values]
                           put [advantages, returns]                                                    (1746-1801)
  _compute_metrics       : get [prompts, responses, response_mask, values, advantages, returns, rm_scores, num_turns] (1876-1894)
  replay buffer          : kv_list()                                                                    (210)
  dispatch               : kv_retrieve_meta (kv_batch_meta2batch_meta, transferqueue_utils.py:233-248)
plus the agent-loop side put of n-row groups (agent_loop_tq.py:228-233) and a worker-style put of full-seq log_probs.

Usage: bench_tq.py N L   (e.g. 1024 8192; 4096 8192; 4096 32768)
Single node, loopback ZMQ: a lower bound for multi-node deployments.
"""

import os
import sys
import time
import uuid

import numpy as np
import torch

sys.path.insert(0, os.path.join(_REPO, "research/bench-a3"))
from bench_common import machine_banner, make_lengths  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
L = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
P = 2048
n_rollout = 8

import ray  # noqa: E402

ray.init(num_cpus=12, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
import transfer_queue as tq  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from verl.utils.tensordict_utils import list_of_dict_to_tensordict  # noqa: E402
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))  # the repository root

machine_banner()
t0 = time.perf_counter()
tq.init()
print(f"tq.init() {time.perf_counter() - t0:.1f}s")

plen, rlen = make_lengths(N, P, L, seed=0)
plen, rlen = plen.tolist(), rlen.tolist()
print(f"\n## N={N} rows, n={n_rollout}, mean prompt {np.mean(plen):.0f}, mean response {np.mean(rlen):.0f} (max L={L})")


def td_bytes(td):
    tot = 0
    for v in td.values():
        if isinstance(v, torch.Tensor):
            vv = v.values() if v.is_nested else v
            tot += vv.numel() * vv.element_size()
    return tot


def row(i, uid, session):
    p, r = plen[i], rlen[i]
    prompts = torch.randint(0, 150000, (p,), dtype=torch.int64)
    responses = torch.randint(0, 150000, (r,), dtype=torch.int64)
    response_mask = torch.ones(r, dtype=torch.int64)
    input_ids = torch.cat([prompts, responses])
    rm = torch.zeros(r, dtype=torch.float32)
    rm[-1] = float(i % 3 == 0)
    return {
        "prompts": prompts,
        "responses": responses,
        "response_mask": response_mask,
        "loss_mask": response_mask,
        "input_ids": input_ids,
        "attention_mask": torch.ones(p + r, dtype=torch.int64),
        "position_ids": torch.arange(p + r, dtype=torch.int64),
        "rollout_log_probs": -torch.rand(r, dtype=torch.float32),
        "rm_scores": rm,
        "num_turns": 2,
        "uid": uid,
        "session_id": session,
        "index": i // n_rollout,
        "data_source": "openai/gsm8k",
        "reward_model": {"style": "rule", "ground_truth": str(i)},
        "extra_info": {"index": i // n_rollout, "split": "train"},
        "raw_prompt": [{"role": "user", "content": "Problem text " * 40}],
        "extra_fields": {"reward_extra_info": {"acc": float(i % 3 == 0)}, "min_global_steps": 1, "max_global_steps": 1},
        "multi_modal_inputs": {},
        "agent_name": "single_turn_agent",
    }


# ---- agent-loop side: one kv_batch_put per prompt group of n rows ----
keys, tags = [], []
groups = []
for g in range(N // n_rollout):
    uid = str(uuid.uuid4())
    rows = [row(g * n_rollout + s, uid, s) for s in range(n_rollout)]
    gkeys = [f"{uid}_{s}_0" for s in range(n_rollout)]
    gtags = [
        {"status": "success", "prompt_len": plen[g * n_rollout + s], "response_len": rlen[g * n_rollout + s],
         "seq_len": plen[g * n_rollout + s] + rlen[g * n_rollout + s], "global_steps": 1, "min_global_steps": 1, "max_global_steps": 1}
        for s in range(n_rollout)
    ]
    groups.append((gkeys, rows, gtags))
    keys.extend(gkeys)
    tags.extend(gtags)

t0 = time.perf_counter()
tds = [list_of_dict_to_tensordict(rows) for _, rows, _ in groups]
t_pack = time.perf_counter() - t0
total_bytes = sum(td_bytes(td) for td in tds)
print(f"list_of_dict_to_tensordict for {len(groups)} groups: {t_pack * 1e3:.0f} ms  (rollout payload {total_bytes / 1e9:.2f} GB)")
t0 = time.perf_counter()
for (gkeys, _, gtags), td in zip(groups, tds):
    tq.kv_batch_put(keys=gkeys, partition_id="train", fields=td, tags=gtags)
t_put = time.perf_counter() - t0
print(f"agent-loop kv_batch_put x{len(groups)} groups of {n_rollout}: {t_put * 1e3:.0f} ms total, {t_put / len(groups) * 1e3:.1f} ms/group, {total_bytes / t_put / 1e6:.0f} MB/s")
del tds

# prompt tags like _submit_batch_to_rollout
t0 = time.perf_counter()
tq.kv_batch_put(keys=[k.split("_")[0] for k in keys[::n_rollout]], partition_id="train",
                tags=[{"is_prompt": True, "status": "finished", "global_steps": 1} for _ in range(N // n_rollout)])
print(f"kv_batch_put tags-only for {N // n_rollout} prompts: {(time.perf_counter() - t0) * 1e3:.0f} ms")

# ---- replay buffer + dispatch metadata ----
t0 = time.perf_counter()
listing = tq.kv_list()
print(f"kv_list(): {(time.perf_counter() - t0) * 1e3:.0f} ms for {sum(len(v) for v in listing.values())} keys")
client = tq.get_client()
t0 = time.perf_counter()
meta = client.kv_retrieve_meta(keys=keys, partition_id="train", create=False)
print(f"kv_retrieve_meta({N} keys) [every dispatch]: {(time.perf_counter() - t0) * 1e3:.0f} ms")


def timed_get(fields, label):
    t0 = time.perf_counter()
    td = tq.kv_batch_get(keys=keys, partition_id="train", select_fields=fields)
    dt = time.perf_counter() - t0
    b = td_bytes(td)
    print(f"GET {label:28s} {dt * 1e3:8.0f} ms  {b / 1e6:8.0f} MB  {b / dt / 1e6:7.0f} MB/s   fields={fields}")
    return td


def timed_put(td, label):
    t0 = time.perf_counter()
    tq.kv_batch_put(keys=keys, partition_id="train", fields=td)
    dt = time.perf_counter() - t0
    b = td_bytes(td)
    print(f"PUT {label:28s} {dt * 1e3:8.0f} ms  {b / 1e6:8.0f} MB  {b / dt / 1e6:7.0f} MB/s   fields={list(td.keys())}")


# ---- worker-style put of full-sequence log_probs/entropy (what every DP rank does after compute_log_prob) ----
seq_lens = torch.tensor([p + r for p, r in zip(plen, rlen)])
offs = torch.zeros(N + 1, dtype=torch.long)
torch.cumsum(seq_lens, 0, out=offs[1:])
full_lp = torch.nested.nested_tensor_from_jagged(-torch.rand(int(seq_lens.sum())), offsets=offs)
full_ent = torch.nested.nested_tensor_from_jagged(torch.rand(int(seq_lens.sum())), offsets=offs)
timed_put(TensorDict({"log_probs": full_lp, "entropy": full_ent}, batch_size=N), "worker log_probs+entropy (all ranks)")

# ---- driver: _compute_old_log_prob ----
td = timed_get(["entropy", "log_probs", "response_mask"], "old_log_prob fields")
resp_lens = torch.tensor(rlen)
roffs = torch.zeros(N + 1, dtype=torch.long)
torch.cumsum(resp_lens, 0, out=roffs[1:])
resp_lp = torch.nested.nested_tensor_from_jagged(-torch.rand(int(resp_lens.sum())), offsets=roffs)
resp_ent = torch.nested.nested_tensor_from_jagged(torch.rand(int(resp_lens.sum())), offsets=roffs)
timed_put(TensorDict({"old_log_probs": resp_lp, "entropy": resp_ent}, batch_size=N), "old_log_probs+entropy")
t0 = time.perf_counter()
dense = td["log_probs"].to_padded_tensor(0.0)
print(f"to_padded_tensor(log_probs full-seq) on driver: {(time.perf_counter() - t0) * 1e3:.0f} ms -> {tuple(dense.shape)}")

# ---- driver: _compute_ref_log_prob / _compute_values ----
timed_get(["log_probs", "response_mask"], "ref fields")
timed_put(TensorDict({"ref_log_prob": resp_lp.clone()}, batch_size=N), "ref_log_prob")
timed_put(TensorDict({"values": resp_ent.clone()}, batch_size=N), "values")

# ---- driver: _compute_advantage ----
td = timed_get(["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob", "values"], "advantage fields")
t0 = time.perf_counter()
padded = td.to_padded_tensor()
print(f"to_padded_tensor(7 fields) on driver: {(time.perf_counter() - t0) * 1e3:.0f} ms")
timed_put(TensorDict({"advantages": resp_lp.clone(), "returns": resp_lp.clone()}, batch_size=N), "advantages+returns")

# ---- driver: _compute_metrics ----
timed_get(["prompts", "responses", "response_mask", "values", "advantages", "returns", "rm_scores", "num_turns"], "metrics fields")

# ---- cleanup ----
t0 = time.perf_counter()
tq.kv_clear(keys=keys, partition_id="train")
print(f"kv_clear({N} keys): {(time.perf_counter() - t0) * 1e3:.0f} ms")
tq.close()
ray.shutdown()
