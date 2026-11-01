"""DataProto / TensorDict driver-side operations at realistic size.

  - DataProto.to_tensordict() with the ~17 non-tensor keys the agent loop echoes (protocol.py:1099-1123)
  - tu.get_tensordict on a dict with N-length object arrays (tensordict_utils.py:377-455)
  - union_numpy_dict deep-equal on overlapping object keys (protocol.py:188-200)
  - DataProto.concat of W=8 agent-loop chunks (protocol.py:914-958)
  - chunk_tensordict + contiguous().consolidate() per DP chunk (protocol.py:1268-1275, tensordict_utils.py:302-375, 883-907)
  - pickle round trip of one consolidated DP chunk (what ray.put/ray.get do), and of the whole batch
  - DataProto pickle round trip via __getstate__/__setstate__ (consolidate + torch.save) (protocol.py:377-421)
  - DataProto.select_idxs(bool mask) (protocol.py:632-670)
  - NonTensorStack pickle (what the V1 agent-loop dispatch and TQ put of non-tensor fields pay)
"""

import copy
import io
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(_REPO, "research/bench-a3"))
from bench_common import gb, machine_banner, make_non_tensor_batch, make_padded_batch, nbytes, timeit  # noqa: E402

from tensordict import TensorDict  # noqa: E402
from tensordict.tensorclass import NonTensorData, NonTensorStack  # noqa: E402

from verl import DataProto  # noqa: E402
from verl.protocol import BatchData, union_numpy_dict  # noqa: E402
from verl.utils import tensordict_utils as tu  # noqa: E402
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))  # the repository root

machine_banner()

shapes = [(1024, 2048, 8192, 8), (4096, 2048, 8192, 32), (4096, 2048, 32768, 32)]
if len(sys.argv) > 1 and sys.argv[1] == "small":
    shapes = [(1024, 2048, 8192, 8)]

for N, P, L, dp in shapes:
    print(f"\n## N={N} P={P} L={L} dp={dp}")
    batch, uid, plen, rlen = make_padded_batch(N, P, L, with_logprob_fields=True)
    ntb = make_non_tensor_batch(N, uid)
    print(f"dense batch payload: {gb(nbytes(batch))}; non-tensor keys: {len(ntb)}")
    big = nbytes(batch) > 5e9
    rep = 2 if big else 3

    td = TensorDict(dict(batch), batch_size=(N,))
    dp_obj = DataProto(batch=td, non_tensor_batch=dict(ntb), meta_info={"global_token_num": (plen + rlen).tolist()})

    # ---- to_tensordict (legacy: 3-5x per step) ----
    timeit(lambda: dp_obj.to_tensordict(), repeat=rep, label=f"DataProto.to_tensordict() ({len(ntb)} non-tensor keys)")
    only_uid = DataProto(batch=td, non_tensor_batch={"uid": uid}, meta_info={})
    timeit(lambda: only_uid.to_tensordict(), repeat=rep, label="DataProto.to_tensordict() (1 non-tensor key)")
    timeit(
        lambda: NonTensorStack.from_list([NonTensorData(item) for item in ntb["reward_model"]]),
        repeat=rep,
        label="  .. NonTensorStack.from_list(N NonTensorData) for one dict key",
    )
    timeit(
        lambda: tu.get_tensordict({"a": torch.zeros(N, 1), **{k: v for k, v in ntb.items()}}),
        repeat=rep,
        label="tu.get_tensordict(dict with N-length object arrays)",
    )

    # ---- union_numpy_dict deep-equal (legacy batch.union(gen_batch_output)) ----
    overlap_keys = ["uid", "data_source", "reward_model", "extra_info"]
    left = {k: ntb[k] for k in overlap_keys}
    right = {k: copy.deepcopy(ntb[k]) for k in overlap_keys}  # distinct objects, equal values (what the agent loop returns)
    timeit(lambda: union_numpy_dict(dict(left), right), repeat=rep, label="union_numpy_dict deep-equal on 4 overlapping object keys")

    # ---- DataProto.concat of W agent-loop chunks ----
    W = 8
    chunks = dp_obj.chunk(W)
    timeit(lambda: DataProto.concat(chunks), repeat=rep, warmup=1, label=f"DataProto.concat of {W} chunks (torch.cat full copy)")

    # ---- dispatch: chunk + consolidate + pickle (legacy TensorDict dispatch path) ----
    td_full = dp_obj.to_tensordict()
    timeit(lambda: BatchData(td_full).chunk(dp), repeat=rep, warmup=1, label=f"BatchData(td).chunk({dp}) = chunk_tensordict + contiguous().consolidate()")
    pieces = BatchData(td_full).chunk(dp)
    piece = pieces[0]
    piece_bytes = sum(v.numel() * v.element_size() for v in piece.values() if isinstance(v, torch.Tensor))
    print(f"one DP chunk payload: {gb(piece_bytes)}")
    timeit(lambda: pickle.dumps(piece, protocol=pickle.HIGHEST_PROTOCOL), repeat=rep, label="pickle.dumps(consolidated chunk)  [ray.put side]")
    blob = pickle.dumps(piece, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"pickled size: {gb(len(blob))}")
    timeit(lambda: pickle.loads(blob), repeat=rep, label="pickle.loads(consolidated chunk)  [worker ray.get side]")
    # non-consolidated TensorDict pickle for comparison
    raw_piece = TensorDict({k: v for k, v in batch.items()}, batch_size=(N,)).chunk(dp)[0]
    timeit(lambda: pickle.dumps(raw_piece, protocol=pickle.HIGHEST_PROTOCOL), repeat=rep, label="pickle.dumps(non-consolidated chunk, tensors only)")
    # whole dispatch cost estimate: all dp chunks pickled (what parallel_put does, threads)
    timeit(lambda: [pickle.dumps(p, protocol=pickle.HIGHEST_PROTOCOL) for p in pieces], repeat=1, warmup=0, label=f"pickle.dumps all {dp} chunks (serial)")

    # ---- DataProto pickle round trip (__getstate__ consolidate+torch.save; __setstate__ torch.load) ----
    if not big:
        timeit(lambda: pickle.dumps(dp_obj, protocol=pickle.HIGHEST_PROTOCOL), repeat=rep, label="pickle.dumps(DataProto) via __getstate__ (consolidate + torch.save)")
        dblob = pickle.dumps(dp_obj, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"DataProto pickled size: {gb(len(dblob))}")
        timeit(lambda: pickle.loads(dblob), repeat=rep, label="pickle.loads(DataProto) via __setstate__ (torch.load)")
        del dblob

    # ---- select_idxs with a bool mask (V1 _compute_metrics) ----
    mask = np.ones(N, dtype=bool)
    mask[::97] = False
    timeit(lambda: dp_obj.select_idxs(mask), repeat=rep, warmup=1, label="DataProto.select_idxs(bool mask) full copy")

    # ---- NonTensorStack pickle (V1 agent-loop dispatch of B prompts; TQ put of non-tensor fields) ----
    stack = NonTensorStack.from_list([NonTensorData(item) for item in ntb["raw_prompt"]])
    timeit(lambda: pickle.dumps(stack, protocol=pickle.HIGHEST_PROTOCOL), repeat=rep, label="pickle.dumps(NonTensorStack of N raw_prompt dicts)")
    timeit(lambda: pickle.dumps(ntb["raw_prompt"], protocol=pickle.HIGHEST_PROTOCOL), repeat=rep, label="pickle.dumps(np object array of N raw_prompt dicts)")

    del batch, td, dp_obj, td_full, pieces, piece, chunks, blob
