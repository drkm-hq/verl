"""K4 vet (CPU, no GPU): sender-side bf16 cast vs receiver-side cast through the REAL
BucketedWeightSender/BucketedWeightReceiver (shared-memory path), plus the bucket/byte
arithmetic of the naive IPC path for Qwen3-1.7B and Qwen3-8B, plus the Python-side
per-parameter overhead of the cast. Run with PYTHONPATH=<clone root>.
"""
import os
import sys
import time
import uuid

import torch

import hashlib
import types
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))  # the repository root

_CLONE = os.environ.get("VERL_CLONE", os.path.join(_REPO, "verl"))
_SRC_PATH = os.path.join(_CLONE, "verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py")
_SRC = open(_SRC_PATH).read()
SRC_SHA = hashlib.sha256(_SRC.encode()).hexdigest()[:12]
# The package __init__ imports vLLM and the lazy helper import drags in uvicorn/fastapi/ray (40 s+ per
# spawned child on this Mac), so the REAL sender/receiver source is exec'd standalone with exactly two
# lines substituted: the verl.utils.device import (CPU shim) and the lazy ensure_async_iterator import.
_SUBS = [
    ("from verl.utils.device import get_device_id, get_device_name, get_torch_device, is_support_ipc",
     "from k4_shim import get_device_id, get_device_name, get_torch_device, is_support_ipc"),
    ("        from verl.workers.rollout.utils import ensure_async_iterator",
     "        from k4_shim import ensure_async_iterator"),
]
_patched = _SRC
for old, new in _SUBS:
    assert _patched.count(old) == 1, old
    _patched = _patched.replace(old, new)
bwt = types.ModuleType("bwt")
bwt.__file__ = _SRC_PATH
sys.modules["bwt"] = bwt
exec(compile(_patched, _SRC_PATH, "exec"), bwt.__dict__)

BUCKET_MB = 2048


class _CpuDev:  # torch.cpu has no empty_cache / ipc_collect
    @staticmethod
    def synchronize():
        pass

    @staticmethod
    def empty_cache():
        pass

    @staticmethod
    def ipc_collect():
        pass


def _patch_cpu():
    # macOS POSIX shm names are limited to 31 chars; verl's "verl_weights_<32 hex>" is 45.
    # Map the name identically on both sides (Linux CI does not need this).
    _create, _rebuild = bwt.create_shared_memory, bwt.rebuild_shared_memory
    bwt.create_shared_memory = lambda size, name: _create(size, "k4" + name[-20:])
    bwt.rebuild_shared_memory = lambda name, size, dtype=torch.uint8: _rebuild("k4" + name[-20:], size, dtype)


def _special_values(t: torch.Tensor) -> torch.Tensor:
    """Inject every value class that could distinguish cast orders."""
    flat = t.view(-1)
    n = flat.numel()
    k = min(n, 64)
    idx = torch.arange(k)
    specials = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1e-40, -1e-40, 3.0e38, -3.0e38,
         1.0 + 2**-8, 1.0 + 2**-9, 1.0 + 3 * 2**-9, 1.0 - 2**-9, 65504.0, 1e-5, 1.17549435e-38],
        dtype=torch.float32,
    )
    flat[idx] = specials.repeat((k + specials.numel() - 1) // specials.numel())[:k]
    return t


def make_model(seed=0):
    """Fake FSDP export: each param is W=4 shards; the sender sees the all-gathered tensor."""
    g = torch.Generator().manual_seed(seed)
    shapes = {
        "model.embed_tokens.weight": (4096, 256),
        "model.layers.0.self_attn.q_proj.weight": (512, 256),
        "model.layers.0.self_attn.k_proj.weight": (128, 256),
        "model.layers.0.input_layernorm.weight": (256,),
        "model.layers.0.mlp.gate_proj.weight": (768, 256),
        "model.layers.1.mlp.down_proj.weight": (256, 768),
        "lm_head.weight": (4096, 256),
    }
    params = {}
    for name, shape in shapes.items():
        full = torch.randn(shape, generator=g, dtype=torch.float32) * 3.0
        params[name] = _special_values(full)
    return params


def _sender(zmq_handle, seed, mode):
    params = make_model(seed)
    W = 4

    def gen():
        for name, full in params.items():
            shards = list(full.chunk(W, dim=0)) if full.dim() else [full]
            if mode == "current":  # transformer_impl.py:1033-1039 -> fp32 full_tensor, no cast
                yield name, torch.cat(shards, dim=0) if full.dim() else full
            else:  # K4(a): cast each local shard to bf16 BEFORE the all-gather
                yield name, torch.cat([s.to(torch.bfloat16) for s in shards], dim=0)

    sender = bwt.BucketedWeightSender(zmq_handle=zmq_handle, bucket_size_mb=8, use_shm=True)
    import asyncio

    asyncio.run(sender.async_send_weights(gen()))


def transfer(mode, seed=0):
    """Sender (asyncio, main thread) and receiver (thread) in one process: zmq ipc + POSIX shm are
    process-agnostic, and this avoids spawn start-up; a faulthandler watchdog dumps all stacks on a stall."""
    import faulthandler
    import threading

    faulthandler.dump_traceback_later(90, exit=True)
    handle = f"ipc:///tmp/k4-vet-{uuid.uuid4().hex[:8]}.sock"
    box = {}

    def _recv_thread():
        try:
            out, buckets, bytes_on_wire = {}, [0], [0]

            def on_bucket(weights, is_last):
                buckets[0] += 1
                for name, t in weights:
                    bytes_on_wire[0] += t.nbytes
                    dst = torch.empty(t.shape, dtype=torch.bfloat16)
                    dst.copy_(t)  # what vLLM's default weight_loader does: param.data.copy_(loaded_weight)
                    out[name] = dst.clone()

            recv = bwt.BucketedWeightReceiver(zmq_handle=handle, device=torch.device("cpu"), use_shm=True)
            recv.receive_weights(on_bucket_received=on_bucket)
            box["res"] = ({k: (tuple(v.shape), v.view(torch.int16).contiguous().numpy().tobytes()) for k, v in out.items()},
                          buckets[0], bytes_on_wire[0])
        except BaseException as e:  # noqa: BLE001
            box["err"] = repr(e)
            raise

    _patch_cpu()
    th = threading.Thread(target=_recv_thread, daemon=True)
    th.start()
    t0 = time.perf_counter()
    _sender(handle, seed, mode)
    th.join(60)
    faulthandler.cancel_dump_traceback_later()
    assert "res" in box, box.get("err", "receiver produced nothing")
    print(f"  transfer({mode}) done in {time.perf_counter() - t0:.2f}s")
    return box["res"]


def part1_bitwise():
    print("== Part 1: bitwise identity through the real sender/receiver (shm, CPU) ==")
    a, ba, bytes_a = transfer("current")
    b, bb, bytes_b = transfer("k4_cast_first")
    assert a.keys() == b.keys()
    total = 0
    import numpy as np

    for name in a:
        ea = torch.from_numpy(np.frombuffer(a[name][1], dtype=np.int16).copy()).view(a[name][0])
        eb = torch.from_numpy(np.frombuffer(b[name][1], dtype=np.int16).copy()).view(b[name][0])
        assert ea.shape == eb.shape
        same = torch.equal(ea, eb)
        total += ea.numel()
        print(f"  {name:45s} {tuple(ea.shape)!s:14s} bitwise_equal={same}")
        assert same, name
    print(f"  all {total} elements bitwise identical (int16 view, NaN payloads included)")
    print(f"  buckets: current={ba} (bytes {bytes_a/2**20:.1f} MiB)  cast-first={bb} (bytes {bytes_b/2**20:.1f} MiB)")
    # reference check without the transport: elementwise cast commutes with concatenation
    params = make_model(0)
    for name, full in params.items():
        shards = list(full.chunk(4, dim=0)) if full.dim() else [full]
        after = torch.cat(shards).to(torch.bfloat16) if full.dim() else full.to(torch.bfloat16)
        before = torch.cat([s.to(torch.bfloat16) for s in shards]) if full.dim() else full.to(torch.bfloat16)
        assert torch.equal(after.view(torch.int16), before.view(torch.int16)), name
    print("  offline check: cat(shards).to(bf16) == cat(shard.to(bf16)) bitwise for every tensor")


def qwen3_tensors(cfg):
    L, H, I, nq, nkv, hd, V, tied = cfg
    ts = [("model.embed_tokens.weight", (V, H))]
    for i in range(L):
        p = f"model.layers.{i}."
        ts += [
            (p + "self_attn.q_proj.weight", (nq * hd, H)),
            (p + "self_attn.k_proj.weight", (nkv * hd, H)),
            (p + "self_attn.v_proj.weight", (nkv * hd, H)),
            (p + "self_attn.o_proj.weight", (H, nq * hd)),
            (p + "self_attn.q_norm.weight", (hd,)),
            (p + "self_attn.k_norm.weight", (hd,)),
            (p + "mlp.gate_proj.weight", (I, H)),
            (p + "mlp.up_proj.weight", (I, H)),
            (p + "mlp.down_proj.weight", (H, I)),
            (p + "input_layernorm.weight", (H,)),
            (p + "post_attention_layernorm.weight", (H,)),
        ]
    ts.append(("model.norm.weight", (H,)))
    if not tied:
        ts.append(("lm_head.weight", (V, H)))
    return ts


def replay_packing(tensors, elem_size, bucket_bytes):
    """Exact replay of BucketedWeightSender.async_send_weights lines 127-163."""
    offset, in_bucket, buckets, direct, total = 0, 0, 0, 0, 0
    for _name, shape in tensors:
        nbytes = elem_size
        for d in shape:
            nbytes *= d
        total += nbytes
        offset = (offset + elem_size - 1) // elem_size * elem_size
        if offset + nbytes > bucket_bytes and in_bucket > 0:
            buckets += 1
            offset, in_bucket = 0, 0
        if offset + nbytes > bucket_bytes:
            direct += 1
            continue
        offset += nbytes
        in_bucket += 1
    buckets += 1  # the last bucket (sent even if empty)
    return buckets, direct, total


def part2_arithmetic():
    print("\n== Part 2: naive IPC path arithmetic (exact replay of the sender's packing rule) ==")
    cfgs = {
        "Qwen3-1.7B": (28, 2048, 6144, 16, 8, 128, 151936, True),
        "Qwen3-8B": (36, 4096, 12288, 32, 8, 128, 151936, False),
    }
    bucket = BUCKET_MB << 20
    W = 8  # trainer ranks = GPUs
    for name, cfg in cfgs.items():
        ts = qwen3_tensors(cfg)
        nparams = sum(int(torch.tensor(s).prod()) for _, s in ts)
        for label, es in (("fp32 (today)", 4), ("bf16 (K4a)", 2)):
            b, d, total = replay_packing(ts, es, bucket)
            gb = total / 1e9
            ag_recv = gb * (W - 1) / W
            t_ag_nvlink = ag_recv / 300.0  # GB/s per-rank all-gather receive, 8xH100 NVSwitch (assumption)
            t_ag_pcie = ag_recv / 20.0  # PCIe-only box (no NVLink) (assumption)
            t_copy = gb / 1300.0  # HBM copy_ into the bucket, read+write (assumption)
            t_rt = (b + d) * 0.007  # fixed lock-step round trip per bucket: sync+zmq+python (assumption 7 ms)
            print(
                f"  {name:10s} {label:12s} tensors={len(ts):4d} (= all-gathers)  bytes={gb:6.2f} GB  "
                f"buckets={b:3d} direct_sends={d}  all-gather recv/rank={ag_recv:5.2f} GB -> "
                f"NVLink {t_ag_nvlink*1e3:5.0f} ms | PCIe-only {t_ag_pcie:5.2f} s ; bucket copy {t_copy*1e3:4.0f} ms ; "
                f"lock-step RT {t_rt*1e3:4.0f} ms"
            )
        print(f"  {name:10s} params={nparams/1e9:.3f} B")


def part3_overhead():
    print("\n== Part 3: Python-side overhead per parameter of the sender-side cast (CPU, tiny tensors) ==")
    n = 2000
    xs = [torch.randn(16, dtype=torch.float32) for _ in range(n)]
    for _ in range(3):
        for x in xs:
            x.to(torch.bfloat16)
    t0 = time.perf_counter()
    for x in xs:
        x.to(torch.bfloat16)
    t_cast = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for x in xs:
        x.is_floating_point() and x.dim() >= 2
    t_guard = (time.perf_counter() - t0) / n
    print(f"  tensor.to(bf16) dispatch: {t_cast*1e6:6.1f} us/param ; dtype/ndim guard: {t_guard*1e6:5.2f} us/param")
    print(f"  x 399 params (Qwen3-8B): {t_cast*399*1e3:.2f} ms per sync of pure Python overhead (GPU: same launch cost, async kernel)")


if __name__ == "__main__":
    print(f"bucketed_weight_transfer.py sha256[:12]={SRC_SHA}; substituted lines: {len(_SUBS)} (device shim, async-iterator helper)")
    part1_bitwise()
    part2_arithmetic()
    part3_overhead()
