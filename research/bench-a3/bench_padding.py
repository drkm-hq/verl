"""Padding <-> nested conversions on the driver (verl/workers/utils/padding.py).

  - left_right_2_no_padding(td)          padding.py:23-100   (legacy: called 3-5x per step on the full batch)
  - no_padding_2_padding(values, td)     padding.py:103-134  (legacy: 2x per log_prob call)
  - response_from_nested(nt, mask_nt)    padding.py:166-181  (V1: 4x per step)
  - response_to_nested(t, mask_nt)       padding.py:184-199  (V1: 2-4x per step)
  - NestedTensor.to_padded_tensor()      (V1: 3x per step on several fields)
and vectorized alternatives with identical outputs.
"""

import sys

import torch
import torch.nn.functional as F

import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))  # the repository root
sys.path.insert(0, os.path.join(_REPO, "research/bench-a3"))
from bench_common import gb, machine_banner, make_padded_batch, nbytes, timeit  # noqa: E402

from tensordict import TensorDict  # noqa: E402

from verl.utils import tensordict_utils as tu  # noqa: E402
from verl.workers.utils.padding import (  # noqa: E402
    left_right_2_no_padding,
    no_padding_2_padding,
    response_from_nested,
    response_to_nested,
)

machine_banner()


def vectorized_position_ids_nested(position_ids, attention_mask, cu_seqlens):
    """Same result as the python loop in left_right_2_no_padding for 1-D position ids."""
    valid = position_ids[attention_mask.bool()]
    return torch.nested.nested_tensor_from_jagged(valid, offsets=cu_seqlens)


def vectorized_no_padding_2_padding(values, prompt_lens, response_lens, max_response_len):
    """Vectorized equivalent of padding.py:120-133 (left-shift by one, right-pad to max_response_len)."""
    seq_lens = prompt_lens + response_lens
    seq_offsets = seq_lens.cumsum(0)
    N = response_lens.shape[0]
    ar = torch.arange(max_response_len)
    # source index for (i, t): seq_offset_i - resp_len_i - 1 + t  (valid while t < resp_len_i)
    src = (seq_offsets - response_lens - 1)[:, None] + ar[None, :]
    valid = ar[None, :] < response_lens[:, None]
    src = torch.where(valid, src, torch.zeros_like(src))
    out = values[src.reshape(-1)].reshape(N, max_response_len, *values.shape[1:])
    out = out * valid.reshape(N, max_response_len, *([1] * (values.ndim - 1))).to(out.dtype)
    return out


def vectorized_response_from_nested(nt, response_lens):
    values, offsets = nt.values(), nt.offsets()
    ends = offsets[1:] - 1  # exclusive end of the left-shifted response window
    starts = ends - response_lens
    # gather all windows with one index build
    total = int(response_lens.sum())
    new_offsets = torch.zeros(len(response_lens) + 1, dtype=torch.long)
    torch.cumsum(response_lens, 0, out=new_offsets[1:])
    idx = torch.repeat_interleave(starts, response_lens) + (
        torch.arange(total) - torch.repeat_interleave(new_offsets[:-1], response_lens)
    )
    return torch.nested.nested_tensor_from_jagged(values[idx], offsets=new_offsets)


def vectorized_response_to_nested(t, response_lens):
    valid = torch.arange(t.shape[1])[None, :] < response_lens[:, None]
    new_offsets = torch.zeros(len(response_lens) + 1, dtype=torch.long)
    torch.cumsum(response_lens, 0, out=new_offsets[1:])
    return torch.nested.nested_tensor_from_jagged(t[valid], offsets=new_offsets)


for N, P, L in [(1024, 2048, 8192), (4096, 2048, 8192), (4096, 2048, 32768)]:
    print(f"\n## N={N} P={P} L={L}")
    batch, uid, plen, rlen = make_padded_batch(N, P, L, with_logprob_fields=True)
    print(f"dense batch payload: {gb(nbytes(batch))}")
    rep = 2 if L > 8192 else 3

    def make_td():
        return TensorDict(dict(batch), batch_size=(N,))

    # ---- left_right_2_no_padding on the whole batch (what every legacy worker call does) ----
    timeit(lambda: left_right_2_no_padding(make_td()), repeat=rep, warmup=1, label="left_right_2_no_padding (padding.py:23) full batch")
    td = left_right_2_no_padding(make_td())
    from verl.utils.attention_utils import unpad_input  # noqa: E402

    timeit(
        lambda: unpad_input(batch["input_ids"].unsqueeze(-1), batch["attention_mask"]),
        repeat=rep,
        label="  .. unpad_input(input_ids, attention_mask) alone",
    )
    _, indices, cu_seqlens, *_ = unpad_input(batch["input_ids"].unsqueeze(-1), batch["attention_mask"])

    def pos_loop():
        am, pid = batch["attention_mask"], batch["position_ids"]
        lst = []
        for i in range(am.shape[0]):
            lst.append(pid[i][am[i].bool()])
        return torch.nested.as_nested_tensor(lst, layout=torch.jagged)

    timeit(pos_loop, repeat=rep, label="  .. position_ids python loop over N rows (padding.py:52-62)")
    timeit(
        lambda: vectorized_position_ids_nested(batch["position_ids"], batch["attention_mask"], cu_seqlens),
        repeat=rep,
        label="  .. position_ids vectorized (boolean mask + from_jagged)",
    )
    a = pos_loop()
    b = vectorized_position_ids_nested(batch["position_ids"], batch["attention_mask"], cu_seqlens)
    assert torch.equal(a.values(), b.values()) and torch.equal(a.offsets(), b.offsets())

    # ---- no_padding_2_padding: nested full-seq values -> dense (N, L) ----
    total = int(cu_seqlens[-1])
    values = torch.randn(total)
    timeit(lambda: no_padding_2_padding(values, td), repeat=rep, label="no_padding_2_padding (padding.py:103) python loop")
    out_loop = no_padding_2_padding(values, td)
    out_vec = vectorized_no_padding_2_padding(values, plen, rlen, L)
    assert torch.equal(out_loop, out_vec)
    timeit(lambda: vectorized_no_padding_2_padding(values, plen, rlen, L), repeat=rep, label="no_padding_2_padding vectorized (gather)")

    # ---- V1: response_from_nested / response_to_nested / to_padded_tensor ----
    log_probs_nt = torch.nested.nested_tensor_from_jagged(values, offsets=cu_seqlens)
    resp_offsets = torch.zeros(N + 1, dtype=torch.long)
    torch.cumsum(rlen, 0, out=resp_offsets[1:])
    resp_mask_nt = torch.nested.nested_tensor_from_jagged(torch.ones(int(rlen.sum()), dtype=torch.int64), offsets=resp_offsets)

    timeit(lambda: response_from_nested(log_probs_nt, resp_mask_nt), repeat=rep, label="response_from_nested (padding.py:166) python loop")
    r_loop = response_from_nested(log_probs_nt, resp_mask_nt)
    r_vec = vectorized_response_from_nested(log_probs_nt, rlen)
    assert torch.equal(r_loop.values(), r_vec.values()) and torch.equal(r_loop.offsets(), r_vec.offsets())
    timeit(lambda: vectorized_response_from_nested(log_probs_nt, rlen), repeat=rep, label="response_from_nested vectorized (index gather)")

    dense = r_loop.to_padded_tensor(0.0)
    timeit(lambda: r_loop.to_padded_tensor(0.0), repeat=rep, label="NestedTensor.to_padded_tensor (N x L fp32)")
    timeit(lambda: response_to_nested(dense, resp_mask_nt), repeat=rep, label="response_to_nested (padding.py:184) python loop")
    t_loop = response_to_nested(dense, resp_mask_nt)
    t_vec = vectorized_response_to_nested(dense, rlen)
    assert torch.equal(t_loop.values(), t_vec.values()) and torch.equal(t_loop.offsets(), t_vec.offsets())
    timeit(lambda: vectorized_response_to_nested(dense, rlen), repeat=rep, label="response_to_nested vectorized (mask + from_jagged)")

    del batch, td, values, log_probs_nt, dense
