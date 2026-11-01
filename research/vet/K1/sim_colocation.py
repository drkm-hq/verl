"""K1 vlm-dedupe: how many duplicate images actually land together under V1's default batching?

Uses verl's real partitioners:
  - dp-rank split : get_seqlen_balanced_partitions(calculate_workload(seqlen), dp, equal_size=True)
                    exactly as verl/trainer/ppo/v1/trainer_base.py:1623-1628 (_balance_batch, unconditional)
  - micro-batches : rearrange_micro_batches(td, max_token_len) as verl/workers/engine/utils.py:112
                    (use_dynamic_bsz=True default, ppo_max_token_len_per_gpu=24576)
Reference config: 128 prompts x n=8, 8 dp ranks, 1 image per prompt.
Reports, per phase, the fraction of vision-encoder work saved by
  (A) within-micro-batch dedupe only
  (B) per-rank cross-phase cache (each unique image on a rank encoded once per step)
  (C) uid-grouped dp split (get_group_balanced_partitions) + within-micro-batch dedupe

Usage: <repo>/.venv-a9/bin/python sim_colocation.py
"""

import sys

import torch

import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))  # the repository root
sys.path.insert(0, os.path.join(_REPO, "verl"))
from tensordict import TensorDict  # noqa: E402

from verl.utils.seqlen_balancing import (  # noqa: E402
    calculate_workload,
    get_group_balanced_partitions,
    get_seqlen_balanced_partitions,
    rearrange_micro_batches,
)

N_PROMPTS, N, DP, MAX_TOK = 128, 8, 8, 24576


def sample_lengths(gen, kind):
    """prompt (text + image tokens) and response lengths per sample."""
    if kind == "geo3k-like":
        prompt = torch.randint(250, 450, (N_PROMPTS,), generator=gen).repeat_interleave(N)
        resp = (torch.randn(N_PROMPTS * N, generator=gen) * 0.6 + 6.4).exp().clamp(16, 2048).long()  # ~600 median
    elif kind == "short-uniform":
        prompt = torch.randint(300, 320, (N_PROMPTS,), generator=gen).repeat_interleave(N)
        resp = torch.randint(100, 400, (N_PROMPTS * N,), generator=gen)
    else:  # long-tail
        prompt = torch.randint(400, 1700, (N_PROMPTS,), generator=gen).repeat_interleave(N)
        resp = (torch.randn(N_PROMPTS * N, generator=gen) * 0.9 + 6.8).exp().clamp(16, 4096).long()
    return prompt + resp


def micro_batches_for(idx, seqlen):
    """Run verl's rearrange_micro_batches on the samples `idx` of one dp rank."""
    lens = seqlen[idx]
    s_max = int(lens.max())
    am = torch.zeros(len(idx), s_max, dtype=torch.long)
    for i, l in enumerate(lens.tolist()):
        am[i, :l] = 1
    td = TensorDict({"input_ids": am.clone(), "attention_mask": am}, batch_size=[len(idx)])
    _, mb_idx = rearrange_micro_batches(td, max_token_len=max(MAX_TOK, s_max))
    return [[idx[i] for i in mb] for mb in mb_idx]


def stats(rank_parts, seqlen, uid):
    total = enc_a = 0
    uniq_rank = 0
    for part in rank_parts:
        part = list(part)
        uniq_rank += len(set(uid[i] for i in part))
        for mb in micro_batches_for(part, seqlen):
            total += len(mb)
            enc_a += len(set(uid[i] for i in mb))
    return total, enc_a, uniq_rank


def main():
    gen = torch.Generator().manual_seed(0)
    uid = [p for p in range(N_PROMPTS) for _ in range(N)]
    print(f"prompts={N_PROMPTS} n={N} dp={DP} max_token_len={MAX_TOK}; 3 encoder passes/step (old_log_prob, ref, update_actor)")
    print(f"{'lengths':14s} {'samples/rank':>12s} {'mb/rank':>8s} | {'A: mb-dedupe':>13s} {'B: rank cache':>14s} {'C: grouped+mb':>14s}  (encoder work saved per phase)")
    for kind in ("geo3k-like", "short-uniform", "long-tail"):
        seqlen = sample_lengths(gen, kind)
        # TQ order is arrival order; shuffle siblings as the async loop would
        perm = torch.randperm(len(seqlen), generator=gen).tolist()
        seqlen_p = seqlen[perm]
        uid_p = [uid[i] for i in perm]
        wl = calculate_workload(seqlen_p)
        parts = get_seqlen_balanced_partitions(wl, k_partitions=DP, equal_size=True)
        total, enc_a, uniq_rank = stats(parts, seqlen_p, uid_p)
        n_mb = sum(len(micro_batches_for(list(p), seqlen_p)) for p in parts)
        # (C) uid-contiguous order, then group-balanced split
        order = sorted(range(len(seqlen)), key=lambda i: uid[i])
        seqlen_g = seqlen[order]
        uid_g = [uid[i] for i in order]
        parts_g = get_group_balanced_partitions(seqlen_g.tolist(), uid_g, k_partitions=DP)
        total_g, enc_c, _ = stats(parts_g, seqlen_g, uid_g)
        print(
            f"{kind:14s} {total // DP:12d} {n_mb / DP:8.1f} | "
            f"{100 * (1 - enc_a / total):12.1f}% {100 * (1 - uniq_rank / total):13.1f}% {100 * (1 - enc_c / total_g):13.1f}%"
        )
    print("B assumes an exact per-rank cache valid across the 3 passes (tower adapter-free); it also removes")
    print("the 2 extra passes: encodes/step = unique-per-rank instead of 3 x samples.")


if __name__ == "__main__":
    main()
