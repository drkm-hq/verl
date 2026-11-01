"""Driver-side advantage estimators: loop vs vectorized, at realistic N and L.

Measures (verl/trainer/ppo/core_algos.py):
  - compute_grpo_outcome_advantage (loop, :268)   vs compute_grpo_vectorized_outcome_advantage (:335)
  - compute_rloo_outcome_advantage (loop, :588)   vs compute_rloo_vectorized_outcome_advantage (:834)
  - compute_opo_outcome_advantage (loop, :640)
  - compute_gae_advantage_return (loop over L, :216) vs a vectorized reference (exact same recurrence, torch ops over N per step is the loop; here we compare against a reverse-scan written with log-space-free cumulative products is not exact for masks, so we only time the loop)
  - compute_reinforce_plus_plus_outcome_advantage (loop over L, :694)
Also checks numerical agreement loop-vs-vectorized on the same inputs.
"""

import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(_REPO, "research/bench-a3"))
from bench_common import machine_banner, make_padded_batch, timeit  # noqa: E402

from verl.trainer.ppo import core_algos  # noqa: E402
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))  # the repository root

torch.manual_seed(0)
machine_banner()

shapes = [(1024, 2048, 8192), (4096, 2048, 8192), (4096, 2048, 32768)]
if len(sys.argv) > 1 and sys.argv[1] == "small":
    shapes = [(1024, 2048, 8192)]

for N, P, L in shapes:
    print(f"\n## N={N} P={P} L={L}")
    batch, uid, plen, rlen = make_padded_batch(N, P, L, with_logprob_fields=False)
    rm = batch["rm_scores"]
    mask = batch["response_mask"]
    maskf = mask.float()

    # ---- GRPO ----
    adv_loop, _ = core_algos.compute_grpo_outcome_advantage(rm, mask, uid)
    adv_vec, _ = core_algos.compute_grpo_vectorized_outcome_advantage(rm, mask, uid)
    print(f"GRPO loop vs vectorized max|diff| = {(adv_loop - adv_vec).abs().max().item():.3e}")
    timeit(lambda: core_algos.compute_grpo_outcome_advantage(rm, mask, uid), label="GRPO loop (core_algos.py:268)")
    timeit(
        lambda: core_algos.compute_grpo_vectorized_outcome_advantage(rm, mask, uid),
        label="GRPO vectorized (core_algos.py:335)",
    )
    # cost split inside the loop version: the scalar-tensor loops vs the final broadcast
    scores = rm.sum(dim=-1)
    timeit(lambda: rm.sum(dim=-1), label="  .. rm_scores.sum(-1) over N x L")
    timeit(lambda: scores.unsqueeze(-1) * mask, label="  .. scalars.unsqueeze(-1) * response_mask (N x L)")

    # ---- RLOO ----
    adv_loop, _ = core_algos.compute_rloo_outcome_advantage(rm, mask, uid)
    adv_vec, _ = core_algos.compute_rloo_vectorized_outcome_advantage(rm, mask, uid)
    print(f"RLOO loop vs vectorized max|diff| = {(adv_loop - adv_vec).abs().max().item():.3e}")
    timeit(lambda: core_algos.compute_rloo_outcome_advantage(rm, mask, uid), label="RLOO loop (core_algos.py:588)")
    timeit(
        lambda: core_algos.compute_rloo_vectorized_outcome_advantage(rm, mask, uid),
        label="RLOO vectorized (core_algos.py:834)",
    )

    # ---- OPO (loop only exists) ----
    timeit(lambda: core_algos.compute_opo_outcome_advantage(rm, mask, uid), label="OPO loop (core_algos.py:640)")

    # ---- GAE: loop over L ----
    values = torch.randn(N, L) * maskf
    rewards = rm
    rep = 1 if L > 8192 or N > 1024 else 2
    timeit(
        lambda: core_algos.compute_gae_advantage_return(rewards, values, maskf, 1.0, 0.95),
        repeat=rep,
        warmup=0,
        label=f"GAE loop over L={L} (core_algos.py:216)",
    )

    # ---- REINFORCE++: loop over L ----
    class Cfg:  # minimal config stand-in
        gamma = 1.0

    timeit(
        lambda: core_algos.compute_reinforce_plus_plus_outcome_advantage(rewards, maskf, config=Cfg()),
        repeat=rep,
        warmup=0,
        label=f"REINFORCE++ loop over L={L} (core_algos.py:694)",
    )

    del batch, values
