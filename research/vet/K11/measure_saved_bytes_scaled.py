"""K11: bytes saved for backward per token per decoder layer, measured on CPU in bf16 at 1/8 (Qwen3-8B) and 1/4
(Qwen3-1.7B) of the real hidden/intermediate/head counts and scaled back up (parameters are excluded (FSDP re-gathers them); every per-token tensor is linear in
H, I and nh*d; only the fp32 lse (nh floats) and RoPE tables are not, and they are < 0.2 KB/token).
Per-layer cost = (saved(L=4) - saved(L=2)) / 2 / T so the embedding and LM head cancel; every2 then averages one
checkpointed and one kept layer. T x T tensors (sdpa math kernel attention probs) are reported separately because
flash-attn saves O (T,nh,d) + lse instead. SAC uses respect_saved_tensors_hooks=True so its saves are counted.
Run: .venv-a9/bin/python research/vet/K11/measure_saved_bytes_scaled.py
"""
import collections, functools, torch
from torch.utils.checkpoint import create_selective_checkpoint_contexts, CheckpointPolicy
from transformers import Qwen3Config, Qwen3ForCausalLM
torch.manual_seed(0)
# name: (scaled dims, scale factor back to the real model)
DIMS = {"Qwen3-8B (H4096 I12288 nh32 nkv8 d128)": (dict(hidden_size=512, intermediate_size=1536, num_attention_heads=4, num_key_value_heads=1, head_dim=128), 8),
        "Qwen3-1.7B (H2048 I6144 nh16 nkv8 d128)": (dict(hidden_size=512, intermediate_size=1536, num_attention_heads=4, num_key_value_heads=2, head_dim=128), 4)}
T = 64
aten = torch.ops.aten
def mm_policy(ctx, op, *a, **k):
    return CheckpointPolicy.MUST_SAVE if op in (aten.mm.default, aten.addmm.default, aten.bmm.default) else CheckpointPolicy.PREFER_RECOMPUTE
def configure(model, mode):
    if mode == "none": model.gradient_checkpointing_disable(); return
    kw = {"use_reentrant": False}
    if mode == "sac": kw.update(context_fn=functools.partial(create_selective_checkpoint_contexts, mm_policy), respect_saved_tensors_hooks=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kw, every_n_layers=2 if mode == "every2" else 1)
def measure(dims, L, mode):
    cfg = Qwen3Config(num_hidden_layers=L, vocab_size=512, max_position_embeddings=T, tie_word_embeddings=False, attn_implementation="sdpa", **dims)
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16); model.train(); configure(model, mode)
    inp = torch.randint(0, 512, (1, T)); shapes = collections.Counter(); tot = [0, 0]
    param_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}   # weights are FSDP-gathered, not activation memory
    def pack(t):
        if t.untyped_storage().data_ptr() in param_storages: return t
        b = t.numel() * t.element_size(); tot[0] += b
        if t.dim() >= 2 and t.shape[-1] == T and t.shape[-2] == T: tot[1] += b
        shapes[(tuple(t.shape), str(t.dtype).replace("torch.", ""))] += b
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(input_ids=inp, labels=inp); out.loss.backward()
    return tot[0], tot[1], shapes
for name, (dims, scale) in DIMS.items():
    print(f"== {name}: measured at 1/{scale} dims, T={T}, bf16, scaled x{scale} ==")
    print(f"{'mode':8s} {'KB/token/layer (excl TxT)':>26s} {'TxT KB/token/layer':>20s}")
    for mode in ["none", "full", "every2", "sac"]:
        t2, x2, s2 = measure(dims, 2, mode); t4, x4, s4 = measure(dims, 4, mode)
        per = (t4 - t2) / 2 / T * scale; perx = (x4 - x2) / 2 / T * scale
        print(f"{mode:8s} {(per - perx)/1024:26.1f} {perx/1024:20.1f}")
        if mode == "none":
            diff = collections.Counter()
            for k in s4: diff[k] = s4[k] - s2.get(k, 0)
            print("   saved tensors per layer (none), scaled shapes, top 14 by bytes:")
            H, I, nh, nkv, d = dims["hidden_size"], dims["intermediate_size"], dims["num_attention_heads"], dims["num_key_value_heads"], dims["head_dim"]
            for (shp, dt), b in diff.most_common(14):
                lab = str(tuple(("H" if x == H else "I" if x == I else "nh" if x == nh else "nkv" if x == nkv else "d" if x == d else "T" if x == T else x) for x in shp))
                print(f"     {lab:26s} {dt:9s} {b/2/T/1024*scale:8.2f} KB/token/layer")
