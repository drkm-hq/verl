"""K11 CPU test: every-k / op-selective activation checkpointing on a tiny HF Qwen3 model.

Checks (fp32, CPU):
  1. loss and every parameter gradient are identical (allclose, and report max abs diff) across
     none / full (HF default, use_reentrant=False) / every-2 / SAC (save aten.mm outputs only).
  2. bytes saved for backward per mode (torch.autograd.graph.saved_tensors_hooks), i.e. the
     activation-memory ordering none > SAC > every-2 > full, and the per-layer tensors HF saves.
Run: .venv-a9/bin/python research/vet/K11/test_selective_ckpt_cpu.py
"""
import functools, sys, torch
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_layers import GradientCheckpointingLayer

torch.manual_seed(0)
cfg = Qwen3Config(hidden_size=64, intermediate_size=192, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=16, vocab_size=256, max_position_embeddings=128,
                  tie_word_embeddings=False, attn_implementation="eager")
base = Qwen3ForCausalLM(cfg).float()
base.train()
B, T = 2, 32
inp = torch.randint(0, 256, (B, T))
labels = inp.clone()

aten = torch.ops.aten
def mm_policy(ctx, op, *args, **kwargs):
    # save matmul outputs, recompute everything else (norms, SiLU, RoPE, softmax, adds)
    return CheckpointPolicy.MUST_SAVE if op in (aten.mm.default, aten.addmm.default, aten.bmm.default) \
        else CheckpointPolicy.PREFER_RECOMPUTE

def configure(model, mode):
    if mode == "none":
        model.gradient_checkpointing_disable(); return
    kw = {"use_reentrant": False}
    if mode == "sac":
        kw["context_fn"] = functools.partial(create_selective_checkpoint_contexts, mm_policy)
    if mode == "every2":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kw, every_n_layers=2)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kw)

def run(mode):
    import copy
    model = copy.deepcopy(base)
    configure(model, mode)
    flags = [m.gradient_checkpointing for m in model.modules() if isinstance(m, GradientCheckpointingLayer)]
    saved = []
    def pack(t):
        saved.append(t.numel() * t.element_size()); return t
    def unpack(t): return t
    model.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        out = model(input_ids=inp, labels=labels)
        out.loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    return out.loss.item(), grads, sum(saved), flags

ref_loss, ref_grads, ref_bytes, _ = run("none")
print(f"{'mode':8s} {'layer ckpt flags':22s} {'loss':>12s} {'max|dgrad|':>12s} {'saved bytes':>12s} {'vs none':>8s}")
ok = True
for mode in ["none", "full", "every2", "sac"]:
    loss, grads, nbytes, flags = run(mode)
    dmax = max((grads[n] - ref_grads[n]).abs().max().item() for n in grads)
    close = all(torch.allclose(grads[n], ref_grads[n], rtol=1e-5, atol=1e-6) for n in grads) and abs(loss - ref_loss) < 1e-6
    ok &= close
    print(f"{mode:8s} {str([int(f) for f in flags]):22s} {loss:12.8f} {dmax:12.3e} {nbytes:12d} {nbytes/ref_bytes:8.3f}  {'OK' if close else 'MISMATCH'}")
print("ALL GRADIENTS MATCH" if ok else "GRADIENT MISMATCH"); sys.exit(0 if ok else 1)
