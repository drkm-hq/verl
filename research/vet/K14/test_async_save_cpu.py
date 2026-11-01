"""K14 CPU test (gloo, 2 ranks): the real FSDPCheckpointManager sync save vs a prototype async subclass that stages
CPU copies synchronously, writes in a background thread, and publishes latest_checkpointed_iteration.txt only after
every rank's future completed (all-reduce MIN on the main thread, so no collective runs off-thread).
Checks: (1) sync save -> load is bit-exact; (2) async save -> mutate the live model (next optimizer step) -> finalize
-> load equals the PRE-mutation snapshot (the staging copy isolates the write from the race); (3) tracker written by
rank 0 with the right step, only at finalize. Prints wall times. The clone is not modified.
Run: .venv-a9/bin/python research/vet/K14/test_async_save_cpu.py
"""
import copy, os, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
import torch, torch.distributed as dist, torch.multiprocessing as mp
from omegaconf import OmegaConf

def _stage(obj):
    if torch.is_tensor(obj): return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict): return {k: _stage(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return type(obj)(_stage(v) for v in obj)
    return obj

def build_async_cls():
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
    class AsyncFSDPCheckpointManager(FSDPCheckpointManager):
        def __init__(self, *a, **k):
            super().__init__(*a, **k); self._pool = ThreadPoolExecutor(max_workers=1); self._pending = None
        def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
            self.finalize(blocking=True)                      # at most one checkpoint in flight
            os.makedirs(local_path, exist_ok=True); dist.barrier()
            ws, r = self.world_size, self.rank
            paths = [os.path.join(local_path, f"{p}_world_size_{ws}_rank_{r}.pt") for p in ("model", "optim", "extra_state")]
            staged = [_stage(self.model.state_dict()), _stage(self.optimizer.state_dict()),
                      {"lr_scheduler": None, "rng": self.get_rng_state()}]
            def _write():
                for sd, p in zip(staged, paths):
                    torch.save(sd, p + ".tmp"); os.replace(p + ".tmp", p)
                return True
            self._pending = (global_step, local_path, self._pool.submit(_write))
        def finalize(self, blocking=False):
            if self._pending is None: return True
            step, path, fut = self._pending
            if blocking: fut.result()
            done = torch.tensor([1 if fut.done() else 0], dtype=torch.int64)
            dist.all_reduce(done, op=dist.ReduceOp.MIN)        # main-thread gloo collective: every rank agrees
            if not done.item(): return False
            fut.result()
            if self.rank == 0:
                tracker = os.path.join(os.path.dirname(os.path.dirname(path)), "latest_checkpointed_iteration.txt")  # default_local_dir, as megatron_checkpoint_manager.py:1120
                with open(tracker + ".tmp", "w") as f: f.write(str(step))
                os.replace(tracker + ".tmp", tracker)
            self._pending = None; return True
    return AsyncFSDPCheckpointManager

def same(sd_a, sd_b):
    return all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a) and set(sd_a) == set(sd_b)

def worker(rank, world, port, root):
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}")
    torch.manual_seed(0)
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
    cfg = Qwen3Config(hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=64, vocab_size=1024, max_position_embeddings=256, tie_word_embeddings=False, attn_implementation="eager")
    def fresh():
        m = Qwen3ForCausalLM(cfg).float(); m.train(); return m, torch.optim.AdamW(m.parameters(), lr=1e-3)
    model, opt = fresh()
    x = torch.randint(0, 1024, (2, 64))
    def step():
        opt.zero_grad(); model(input_ids=x, labels=x).loss.backward(); opt.step()
    step(); step()
    ckpt_cfg = OmegaConf.create({"save_contents": ["model", "optimizer", "extra"], "load_contents": ["model", "optimizer", "extra"],
                                 "async_save": False, "strict": True, "save_lora_only": False})
    nbytes = sum(p.numel() * 4 for p in model.parameters()) * 3   # params + exp_avg + exp_avg_sq
    # (1) sync path, the real manager
    mgr = FSDPCheckpointManager(model, opt, None, None, checkpoint_config=ckpt_cfg)
    d_sync = os.path.join(root, "sync", "global_step_2", "actor")
    t0 = time.perf_counter(); mgr.save_checkpoint(d_sync, global_step=2); t_sync = time.perf_counter() - t0
    m2, o2 = fresh(); FSDPCheckpointManager(m2, o2, None, None, checkpoint_config=ckpt_cfg).load_checkpoint(d_sync)
    ok1 = same(model.state_dict(), m2.state_dict()) and same(_stage(opt.state_dict())["state"][0], o2.state_dict()["state"][0])
    # (2) async prototype: save, then race it with the next optimizer step, then finalize
    A = build_async_cls()
    amgr = A(model, opt, None, None, checkpoint_config=OmegaConf.merge(ckpt_cfg, {"async_save": True}))
    snap = _stage(model.state_dict()); snap_opt = _stage(opt.state_dict())
    d_async = os.path.join(root, "async", "global_step_3", "actor")
    t0 = time.perf_counter(); amgr.save_checkpoint(d_async, global_step=3); t_call = time.perf_counter() - t0
    tracker = os.path.join(root, "async", "latest_checkpointed_iteration.txt")
    early_tracker = os.path.exists(tracker)
    step()                                                           # the race: live params/optimizer change
    changed = not same(model.state_dict(), snap)
    t0 = time.perf_counter(); fin = amgr.finalize(blocking=True); t_fin = time.perf_counter() - t0
    dist.barrier()
    m3, o3 = fresh(); FSDPCheckpointManager(m3, o3, None, None, checkpoint_config=ckpt_cfg).load_checkpoint(d_async)
    ok2 = same(snap, m3.state_dict()) and same(snap_opt["state"][0], o3.state_dict()["state"][0])
    ok3 = fin and (rank != 0 or (not early_tracker and open(tracker).read().strip() == "3"))
    print(f"[rank {rank}] shard {nbytes/1e6:.1f} MB | sync save {t_sync*1e3:.0f} ms | async call {t_call*1e3:.0f} ms, finalize {t_fin*1e3:.0f} ms | "
          f"sync_bitexact={ok1} live_changed_after_call={changed} async_equals_pre_race_snapshot={ok2} tracker_only_at_finalize={ok3}", flush=True)
    res = torch.tensor([int(ok1 and ok2 and ok3 and changed)]); dist.all_reduce(res, op=dist.ReduceOp.MIN)
    if rank == 0: print("K14 CPU TEST", "PASS" if res.item() else "FAIL", flush=True)
    dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    root = tempfile.mkdtemp(prefix="k14-")
    mp.spawn(worker, args=(2, port, root), nprocs=2, join=True)
