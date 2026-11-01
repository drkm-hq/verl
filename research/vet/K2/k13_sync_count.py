"""K13 CPU proof: count host-sync-inducing tensor ops in transformers 5.16.1's per-layer varlen preparation.

`_flash_attention_forward` (modeling_flash_attention_utils.py:694-860) is called once per decoder layer. This
script replays its control flow between lines 766 and 800 verbatim, with the real helper functions
(`_is_packed_sequence`, `prepare_fa_kwargs_from_position_ids`, `_process_flash_attention_kwargs`) imported from the
installed 5.16.1 package, for:
  (a) what verl passes today on the default path: position_ids only (transformer_impl.py:1264-1270);
  (b) explicit FlashAttentionKwargs: cu_seq_lens_q/k + Python-int max_length_q/k (K13).
The flash-attn kernels themselves are not needed: the syncs live in the preparation. The final conversion of a
tensor-valued max_seqlen to `int` happens at the custom-op boundary (`flash_attn::_flash_attn_varlen_forward`,
flash_attn_interface.py v2.8.3:143-149 declares `max_seqlen_q: int`); it is modelled here by `int(tensor)`.

Run:
  cd <repo> && .venv-cpu/bin/python -I research/vet/K2/k13_sync_count.py
"""

from collections import Counter, defaultdict

import torch
import transformers
from torch.overrides import TorchFunctionMode
from transformers.modeling_flash_attention_utils import (
    _is_packed_sequence,
    _process_flash_attention_kwargs,
    prepare_fa_kwargs_from_position_ids,
)

SYNC_FUNCS = {
    "item", "__bool__", "__int__", "__index__", "__float__", "nonzero", "tolist", "numpy", "cpu",
}


class SyncCounter(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", str(func))
        if name in SYNC_FUNCS:
            self.counts[name] += 1
        return func(*args, **(kwargs or {}))


def one_layer(position_ids, cu_seq_lens_q=None, cu_seq_lens_k=None, max_length_q=None, max_length_k=None):
    """Lines 766-800 of _flash_attention_forward, attention_mask=None (verl's rmpad path)."""
    query_batch = position_ids.shape[0]
    total_len = position_ids.shape[1]
    is_fa_with_position_ids = _is_packed_sequence(position_ids, batch_size=query_batch)  # :766
    is_fa_with_varlen_kwargs = all(  # :767-769
        kwarg is not None for kwarg in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)
    )
    attention_mask = None
    if attention_mask is not None:  # :772
        raise AssertionError
    elif is_fa_with_varlen_kwargs or is_fa_with_position_ids:  # :796  (Tensor.__bool__ when the first is False)
        if cu_seq_lens_q is None or cu_seq_lens_k is None:  # :797
            (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
                position_ids  # :458-495: nonzero() + diff().max()
            )
        flash_kwargs = _process_flash_attention_kwargs(  # :591-691
            query_length=total_len,
            key_length=total_len,
            is_causal=True,
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
            supports_mapping=defaultdict(lambda: True),
        )
        # flash_attn::_flash_attn_varlen_forward takes `max_seqlen_q: int` (flash_attn_interface.py:149)
        return int(flash_kwargs["max_seqlen_q"]), int(flash_kwargs["max_seqlen_k"]), cu_seq_lens_q
    else:
        raise AssertionError("verl's packed batch must take the varlen branch")


def packed_position_ids(seq_lens):
    return torch.cat([torch.arange(n) for n in seq_lens]).unsqueeze(0)


def main():
    print(f"torch {torch.__version__}  transformers {transformers.__version__}")
    seq_lens = [37 + 9, 5 + 14, 20 + 3, 11 + 6]
    position_ids = packed_position_ids(seq_lens)
    cu = torch.tensor([0] + torch.tensor(seq_lens).cumsum(0).tolist(), dtype=torch.int32)
    max_len = max(seq_lens)

    with SyncCounter() as c_a:
        mq_a, mk_a, cu_a = one_layer(position_ids)
    with SyncCounter() as c_b:
        mq_b, mk_b, cu_b = one_layer(position_ids, cu, cu, max_len, max_len)

    assert torch.equal(cu_a.to(torch.int32), cu), (cu_a, cu)
    assert (mq_a, mk_a) == (mq_b, mk_b) == (max_len, max_len)
    print(f"(a) position_ids only  -> sync-inducing calls per layer: {dict(c_a.counts)}  total {sum(c_a.counts.values())}")
    print(f"(b) explicit varlen kwargs -> sync-inducing calls per layer: {dict(c_b.counts)}  total {sum(c_b.counts.values())}")
    print("cu_seqlens / max_seqlen identical between (a) and (b): True")
    per_layer_a, per_layer_b = sum(c_a.counts.values()), sum(c_b.counts.values())
    for model, layers in (("Qwen3-1.7B", 28), ("Qwen3-8B", 36), ("Qwen3-32B", 64)):
        fwd_a, fwd_b = per_layer_a * layers, per_layer_b * layers
        print(
            f"{model:11s} {layers} layers: forward-only micro-batch {fwd_a} -> {fwd_b} syncs; "
            f"training micro-batch (forward + gradient-checkpoint recompute) {2 * fwd_a} -> {2 * fwd_b}"
        )


if __name__ == "__main__":
    main()
