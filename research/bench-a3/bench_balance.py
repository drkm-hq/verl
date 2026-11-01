"""Batch balancing on the driver (legacy `_balance_batch` ray_trainer.py:1182-1250 and V1 trainer_base.py:1609-1633).

Measures:
  - attention_mask.sum(-1) over N x S
  - calculate_workload
  - get_seqlen_balanced_partitions(workload_TENSOR, dp, equal_size=True)   <- what both trainers pass today
  - get_seqlen_balanced_partitions(workload_LIST, ...)                      <- same algorithm on python ints
  - the per-partition sort with tensor indexing (ray_trainer.py:1234-1237) vs on a list
  - log_seqlen_unbalance
  - DataProto.reorder(global_idx) on the full padded batch (protocol.py:960-966): one full copy
"""

import sys

import numpy as np
import torch

import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))  # the repository root
sys.path.insert(0, os.path.join(_REPO, "research/bench-a3"))
from bench_common import gb, machine_banner, make_padded_batch, nbytes, timeit  # noqa: E402

from tensordict import TensorDict  # noqa: E402

from verl import DataProto  # noqa: E402
from verl.utils.seqlen_balancing import (  # noqa: E402
    calculate_workload,
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)

machine_banner()

for N, P, L, dp in [(1024, 2048, 8192, 8), (4096, 2048, 8192, 32), (4096, 2048, 8192, 128), (4096, 2048, 32768, 32)]:
    print(f"\n## N={N} P={P} L={L} dp_size={dp}")
    batch, uid, plen, rlen = make_padded_batch(N, P, L, with_logprob_fields=True)
    attention_mask = batch["attention_mask"]
    print(f"dense batch payload: {gb(nbytes(batch))}")

    seqlen = timeit(lambda: attention_mask.view(N, -1).sum(-1), label="attention_mask.sum(-1) over N x S")
    global_seqlen_lst = attention_mask.view(N, -1).sum(-1)
    workload_t = calculate_workload(global_seqlen_lst)
    workload_l = workload_t.tolist()

    timeit(
        lambda: get_seqlen_balanced_partitions(workload_t, k_partitions=dp, equal_size=True),
        label=f"KK partitions, TENSOR input (as trainers call it), dp={dp}",
    )
    timeit(
        lambda: get_seqlen_balanced_partitions(workload_l, k_partitions=dp, equal_size=True),
        label=f"KK partitions, LIST input, dp={dp}",
    )
    parts = get_seqlen_balanced_partitions(workload_l, k_partitions=dp, equal_size=True)
    parts_t = get_seqlen_balanced_partitions(workload_t, k_partitions=dp, equal_size=True)
    assert parts == parts_t, "tensor vs list partitions differ"

    def sort_tensor_index():
        out = []
        for partition in parts:
            p = list(partition)
            p.sort(key=lambda x: (workload_t[x], x))
            out.append(p[::2] + p[1::2][::-1])
        return out

    def sort_list_index():
        out = []
        for partition in parts:
            p = list(partition)
            p.sort(key=lambda x: (workload_l[x], x))
            out.append(p[::2] + p[1::2][::-1])
        return out

    timeit(sort_tensor_index, label="per-partition sort, tensor indexing (ray_trainer.py:1234-1237)")
    timeit(sort_list_index, label="per-partition sort, list indexing")
    assert sort_tensor_index() == sort_list_index()

    timeit(
        lambda: log_seqlen_unbalance(seqlen_list=global_seqlen_lst.tolist(), partitions=parts, prefix="x"),
        label="log_seqlen_unbalance",
    )

    # full reorder copy (legacy only; V1 reorders metadata)
    td = TensorDict(batch, batch_size=(N,))
    dp_obj = DataProto(batch=td, non_tensor_batch={"uid": uid})
    global_idx = torch.tensor([j for partition in sort_list_index() for j in partition])
    if nbytes(batch) < 12e9:
        timeit(lambda: dp_obj.reorder(global_idx), repeat=2, warmup=1, label="DataProto.reorder(global_idx) full copy")
    del dp_obj, td, batch
