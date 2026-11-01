"""Tokenizer costs: validation decode loop on the driver (ray_trainer.py:654-665, v1/trainer_base.py:1105-1108)
and the agent-loop per-sample tokenizer.pad x3 (agent_loop.py:728-795), with the Qwen2.5 tokenizer (vocab ~151k)."""

import sys
import time

import torch

sys.path.insert(0, os.path.join(_REPO, "research/bench-a3"))
from bench_common import machine_banner, make_lengths, timeit  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))  # the repository root

machine_banner()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
print("tokenizer:", type(tok).__name__, "vocab", len(tok), "pad_token_id", tok.pad_token_id)

for N, P, L in [(1024, 2048, 8192), (1024, 2048, 32768)]:
    print(f"\n## N={N} P={P} L={L}")
    plen, rlen = make_lengths(N, P, L, seed=1)
    print(f"mean prompt len {plen.float().mean():.0f}, mean response len {rlen.float().mean():.0f}")
    g = torch.Generator().manual_seed(0)
    # realistic token ids: mostly common tokens (small ids decode to short strings); use a skewed distribution
    ids_resp = (torch.rand(N, L, generator=g) ** 3 * 150_000).long()
    ids_prompt = (torch.rand(N, P, generator=g) ** 3 * 150_000).long()
    ar_l = torch.arange(L)
    resp_mask = ar_l[None, :] < rlen[:, None]
    ids_resp = ids_resp * resp_mask
    ar_p = torch.arange(P)
    prompt_mask = ar_p[None, :] >= (P - plen)[:, None]
    ids_prompt = ids_prompt * prompt_mask

    # 1) validation decode loop exactly as written: decode padded rows with skip_special_tokens
    n = 256
    t0 = time.perf_counter()
    outs = [tok.decode(ids, skip_special_tokens=True) for ids in ids_resp[:n]]
    dt = time.perf_counter() - t0
    print(f"tokenizer.decode(padded response row, skip_special_tokens) x{n}: {dt * 1e3:.0f} ms total, {dt / n * 1e3:.2f} ms/row -> {dt / n * N:.1f} s for N={N}")
    t0 = time.perf_counter()
    outs = [tok.decode(ids, skip_special_tokens=True) for ids in ids_prompt[:n]]
    dt = time.perf_counter() - t0
    print(f"tokenizer.decode(padded prompt row) x{n}: {dt * 1e3:.0f} ms total, {dt / n * 1e3:.2f} ms/row -> {dt / n * N:.1f} s for N={N}")
    # 2) decode only the valid tokens (python list)
    t0 = time.perf_counter()
    outs = [tok.decode(ids_resp[i, : rlen[i]].tolist(), skip_special_tokens=True) for i in range(n)]
    dt = time.perf_counter() - t0
    print(f"tokenizer.decode(valid tokens only, list) x{n}: {dt * 1e3:.0f} ms total, {dt / n * 1e3:.2f} ms/row -> {dt / n * N:.1f} s for N={N}")
    # 3) batch_decode (Rust-parallel) on valid tokens
    lists = [ids_resp[i, : rlen[i]].tolist() for i in range(n)]
    t0 = time.perf_counter()
    outs = tok.batch_decode(lists, skip_special_tokens=True)
    dt = time.perf_counter() - t0
    print(f"tokenizer.batch_decode(valid tokens, {n} rows): {dt * 1e3:.0f} ms total, {dt / n * 1e3:.2f} ms/row -> {dt / n * N:.1f} s for N={N}")

    # 4) agent loop: tokenizer.pad x3 per sample (prompt left, response right, response_mask right)
    def pad_one(i):
        p = ids_prompt[i, P - plen[i] :].tolist()
        r = ids_resp[i, : rlen[i]].tolist()
        tok.padding_side = "left"
        a = tok.pad({"input_ids": p}, padding="max_length", max_length=P, return_tensors="pt", return_attention_mask=True)
        tok.padding_side = "right"
        b = tok.pad({"input_ids": r}, padding="max_length", max_length=L, return_tensors="pt", return_attention_mask=True)
        c = tok.pad({"input_ids": [1] * len(r)}, padding="max_length", max_length=L, return_tensors="pt", return_attention_mask=False)
        return a, b, c

    t0 = time.perf_counter()
    for i in range(n):
        pad_one(i)
    dt = time.perf_counter() - t0
    print(f"agent-loop tokenizer.pad x3 per sample: {dt / n * 1e3:.2f} ms/sample -> {dt / n * N:.1f} s per {N} samples (spread over agent-loop workers)")

    def pad_torch(i):
        p = ids_prompt[i, P - plen[i] :]
        r = ids_resp[i, : rlen[i]]
        a = torch.full((1, P), tok.pad_token_id, dtype=torch.long)
        a[0, P - len(p) :] = p
        b = torch.full((1, L), tok.pad_token_id, dtype=torch.long)
        b[0, : len(r)] = r
        return a, b

    t0 = time.perf_counter()
    for i in range(n):
        pad_torch(i)
    dt = time.perf_counter() - t0
    print(f"same padding with torch.full + slice assignment: {dt / n * 1e3:.2f} ms/sample -> {dt / n * N:.1f} s per {N} samples")
