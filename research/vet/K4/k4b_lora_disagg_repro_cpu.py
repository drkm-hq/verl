"""K4b vet (CPU, no GPU, no Ray): reproduce which tensors the trainer exports on each
checkpoint-engine path for a LoRA (merge=False) model, driving the REAL
ActorRolloutRefWorker.update_weights and CheckpointEngineWorker.update_weights bodies
(engine_workers.py:785-886, checkpoint_engine/base.py:353-360) with a real PEFT model
through the REAL verl.utils.fsdp_utils.collect_lora_params / replace_lora_wrapper
(the non-FSDP branch, fsdp_utils.py:767-779, which has the same selection rule as the
FSDP branch at :743-765). Run with PYTHONPATH=<clone root>.
"""
import asyncio
from types import SimpleNamespace

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

import verl.workers.engine_workers as ew
from verl.checkpoint_engine.base import CheckpointEngineWorker
from verl.utils.fsdp_utils import collect_lora_params, replace_lora_wrapper
from verl.workers.engine_workers import ActorRolloutRefWorker


class Attn(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(h, h, bias=False) for _ in range(4))


class Layer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.self_attn = Attn(h)
        self.input_layernorm = nn.LayerNorm(h)


class Inner(nn.Module):
    def __init__(self, h, L):
        super().__init__()
        self.embed_tokens = nn.Embedding(64, h)
        self.layers = nn.ModuleList([Layer(h) for _ in range(L)])


class TinyLM(nn.Module):  # HF-like naming: model.layers.N.self_attn.q_proj.weight, lm_head.weight
    def __init__(self, h=16, L=2):
        super().__init__()
        self.model = Inner(h, L)
        self.lm_head = nn.Linear(h, 64, bias=False)

    def forward(self, x):
        return self.lm_head(self.model.embed_tokens(x))


class FakeFSDPEngine:
    """Mirrors FSDPEngine.get_per_tensor_param (transformer_impl.py:981-1065) for a plain
    (non-FSDP, non-DTensor) PEFT module: same branch structure, same helper calls."""

    is_param_offload_enabled = False

    def __init__(self, peft_model, merge=False):
        self.peft_model, self.merge = peft_model, merge
        self.calls = []

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
        self.calls.append(base_sync_done)
        peft_config = None
        if hasattr(self.peft_model, "peft_config") and not self.merge:  # :1003-1012
            peft_config = self.peft_model.peft_config.get("default", None)
            params = collect_lora_params(module=self.peft_model, layered_summon=layered_summon, base_sync_done=base_sync_done)
            if not base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            params = self.peft_model.state_dict()
        peft_config_dict = peft_config.to_dict() if peft_config is not None else None  # :1064
        return iter(params.items()), peft_config_dict

    def to(self, *a, **k):
        pass


class CaptureCkptEngine:
    def __init__(self):
        self.sent = None

    async def send_weights(self, weights, global_steps=None):
        self.sent = [name for name, _ in weights]
        return {}

    def receive_weights(self, global_steps=None):
        async def _w():
            for n in self.sent:
                yield n, torch.zeros(1)

        return _w()


class CaptureRollout:
    def __init__(self):
        self.updates, self.sleep_level = [], 2

    async def resume(self, tags):
        pass

    async def update_weights(self, weights, peft_config=None, base_sync_done=False, global_steps=None):
        self.updates.append(({n for n, _ in weights}, peft_config is not None, base_sync_done))


class CaptureServerAdapter:
    async def update_weights(self, weights, global_steps=None, **kwargs):
        self.names = [n async for n, _ in weights]
        self.kwargs = kwargs


def _no_gpu(monkey=ew):
    class _D:
        @staticmethod
        def empty_cache():
            pass

        @staticmethod
        def synchronize():
            pass

    monkey.get_torch_device = lambda: _D
    monkey.set_expandable_segments = lambda *_a, **_k: None
    monkey.log_gpu_memory_usage = lambda *_a, **_k: None


def build_worker(backend, peft_model, load_format="safetensors"):
    w = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    w.config = SimpleNamespace(rollout=SimpleNamespace(checkpoint_engine=SimpleNamespace(backend=backend),
                                                       load_format=load_format, free_cache_engine=False, get=lambda k, d=None: d))
    w.actor = SimpleNamespace(engine=FakeFSDPEngine(peft_model))
    w.checkpoint_engine = CaptureCkptEngine()
    w.rollout = CaptureRollout()
    w.base_sync_done = "dummy" not in load_format  # engine_workers.py:717
    w.layered_summon = False
    w.peft_merge = False  # :719, model.lora.merge default False
    # (rank is a read-only property; only the RLInsight decorator reads it and __wrapped__ bypasses it)
    return w


def receiver_decision(kwargs):
    """vllm_rollout/utils.py:266-269,293,335,366 and vllm_async_server.py:707-713 as a rule."""
    peft_config, bsd = kwargs.get("peft_config"), kwargs.get("base_sync_done", False)
    if peft_config and bsd:
        return "add_lora(TensorLoRARequest) -> list_loras()=[123] -> requests carry LoRARequest(123)"
    return "model.load_weights(base) -> list_loras()=[] -> lora_request=None -> BASE POLICY served"


def main():
    _no_gpu()
    torch.manual_seed(0)
    peft_model = get_peft_model(TinyLM(), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    all_names = [n for n, _ in peft_model.named_parameters()]
    print(f"PEFT model: {len(all_names)} params, {sum('lora_' in n for n in all_names)} adapter tensors\n")

    print("== collect_lora_params selection rule (REAL function) ==")
    for bsd in (False, True):
        ks = list(collect_lora_params(peft_model, layered_summon=False, base_sync_done=bsd).keys())
        n_lora = sum("lora_" in k for k in ks)
        print(f"  base_sync_done={bsd!s:5s}: {len(ks):2d} tensors, {n_lora} contain 'lora_'   e.g. {ks[1]}")

    print("\n== REAL ActorRolloutRefWorker.update_weights per backend (load_format=safetensors, merge=False) ==")
    rows = []
    for backend in ("nccl", "nixl", "mooncake", "naive"):
        w = build_worker(backend, peft_model)
        for step in (1, 2):
            asyncio.run(ActorRolloutRefWorker.update_weights.__wrapped__(w, global_steps=step, mode="auto"))
        if backend != "naive":
            names = w.checkpoint_engine.sent
            # rollout side: REAL CheckpointEngineWorker.update_weights forwards to the server adapter
            cw = CheckpointEngineWorker.__new__(CheckpointEngineWorker)
            cw.checkpoint_engine, cw.server_adapter = w.checkpoint_engine, CaptureServerAdapter()
            asyncio.run(CheckpointEngineWorker.update_weights.__wrapped__(cw, global_steps=2))
            kw = cw.server_adapter.kwargs
            rows.append((backend, w.actor.engine.calls, len(names), sum("lora_" in n for n in names), kw, receiver_decision(kw)))
        else:
            last_names, has_cfg, bsd = w.rollout.updates[-1]
            kw = {"peft_config": has_cfg, "base_sync_done": bsd}
            rows.append((backend, w.actor.engine.calls, len(last_names), sum("lora_" in n for n in last_names), kw, receiver_decision(kw)))
    for backend, calls, n, nl, kw, decision in rows:
        print(f"  backend={backend:9s} get_per_tensor_param(base_sync_done=...) calls={calls}  exported={n:2d} tensors, "
              f"{nl} adapter  -> server_adapter kwargs={kw}\n      => {decision}")

    bad = [r for r in rows if r[0] != "naive" and r[3] == 0]
    assert len(bad) == 3, "expected every non-naive backend to export zero adapter tensors on the clone"
    assert rows[-1][3] > 0 and rows[-1][4]["peft_config"], "naive path must ship the adapter"
    print("\nRESULT: on main@2ab35292 every non-naive backend exports the frozen base only and forwards no peft_config;"
          " the standalone replica never receives an adapter -> it samples the initial policy on every step.")


if __name__ == "__main__":
    main()
