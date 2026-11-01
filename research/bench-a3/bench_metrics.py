"""Metric computation on the driver (verl/trainer/ppo/metric_utils.py) and validation-loop iteration costs."""

import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(_REPO, "research/bench-a3"))
from bench_common import gb, machine_banner, make_non_tensor_batch, make_padded_batch, nbytes, timeit  # noqa: E402

from tensordict import TensorDict  # noqa: E402

from verl import DataProto  # noqa: E402
from verl.trainer.ppo.metric_utils import (  # noqa: E402
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.core_algos import agg_loss  # noqa: E402
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))  # the repository root

machine_banner()

for N, P, L in [(1024, 2048, 8192), (4096, 2048, 8192), (4096, 2048, 32768)]:
    print(f"\n## N={N} P={P} L={L}")
    batch, uid, plen, rlen = make_padded_batch(N, P, L, with_logprob_fields=True)
    ntb = make_non_tensor_batch(N, uid)
    batch["token_level_scores"] = batch["rm_scores"]
    batch["token_level_rewards"] = batch["rm_scores"]
    batch["advantages"] = torch.randn(N, L) * batch["response_mask"]
    batch["returns"] = batch["advantages"]
    batch["values"] = torch.randn(N, L) * batch["response_mask"]
    batch["entropys"] = torch.rand(N, L) * batch["response_mask"]
    print(f"dense batch payload: {gb(nbytes(batch))}")
    td = TensorDict(dict(batch), batch_size=(N,))
    dp = DataProto(batch=td, non_tensor_batch=dict(ntb), meta_info={"global_token_num": (plen + rlen).tolist()})
    timing_raw = {"step": 100.0, "gen": 50.0, "old_log_prob": 5.0, "ref": 5.0, "adv": 1.0, "update_actor": 30.0}
    rep = 2 if L > 8192 else 3

    timeit(lambda: compute_data_metrics(dp, use_critic=True), repeat=rep, label="compute_data_metrics(use_critic=True)")
    timeit(lambda: compute_data_metrics(dp, use_critic=False), repeat=rep, label="compute_data_metrics(use_critic=False)")
    timeit(lambda: compute_timing_metrics(dp, timing_raw), repeat=rep, label="compute_timing_metrics")
    timeit(lambda: compute_throughout_metrics(dp, timing_raw, 64), repeat=rep, label="compute_throughout_metrics")
    timeit(lambda: compute_variance_proxy_metrics(dp, 1.0), repeat=rep, label="compute_variance_proxy_metrics (no sum_pi_squared: early return)")
    timeit(
        lambda: agg_loss(loss_mat=batch["entropys"], loss_mask=batch["response_mask"], loss_agg_mode="token-mean"),
        repeat=rep,
        label="agg_loss(entropys, response_mask, token-mean)  [driver entropy metric]",
    )
    # legacy fit(): global_token_num + multi_modal loop + response mask
    timeit(lambda: torch.sum(batch["attention_mask"], dim=-1).tolist(), repeat=rep, label="attention_mask.sum(-1).tolist() (global_token_num)")

    # validation: iterating a DataProto item by item (ray_trainer.py:619-621)
    n_val = min(N, 1024)
    sub = dp.slice(0, n_val)
    timeit(
        lambda: [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in sub],
        repeat=rep,
        label=f"[item... for item in DataProto] over {n_val} items (DataProto.__getitem__(int))",
    )
    timeit(
        lambda: [rm.get("ground_truth", None) for rm in sub.non_tensor_batch["reward_model"]],
        repeat=rep,
        label=f"same over the non_tensor_batch array directly ({n_val} items)",
    )
    del batch, td, dp, sub
