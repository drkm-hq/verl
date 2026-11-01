"""K12 vet: independent re-run of the V1 horizon bug through the real engine loss functions with nested inputs.

Covers actor (ppo_loss) and critic (value_loss), all five agg modes, three micro-batch splits, with and without
loss_scale_factor, and shows that merely injecting `max_response_len` into the TensorDict (fix candidate (i))
breaks ppo_loss with a shape mismatch, because losses.py:103 pads the other fields to the micro-batch max.
"""
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig, CriticConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss, value_loss

torch.manual_seed(0)
prompt_lens = [4, 3, 5, 2, 6, 3, 4, 5]
resp_lens = [12, 1, 3, 7, 12, 2, 9, 5]
RESPONSE_LENGTH = 16  # the configured rollout.response_length (the Dr. GRPO generation budget)
B = len(resp_lens)
prompts = [torch.randint(10, 100, (p,)) for p in prompt_lens]
responses = [torch.randint(10, 100, (r,)) for r in resp_lens]
input_ids = [torch.cat([p, r]) for p, r in zip(prompts, responses, strict=True)]
resp_mask = [torch.ones(r, dtype=torch.int64) for r in resp_lens]
resp_mask[0][4:7] = 0
old_lp = [-1.5 + 0.5 * torch.randn(r) for r in resp_lens]
adv = [torch.randn(r) for r in resp_lens]
vals = [torch.randn(r) for r in resp_lens]
rets = [torch.randn(r) for r in resp_lens]
seq_lens = torch.tensor([p + r for p, r in zip(prompt_lens, resp_lens, strict=True)])
offsets = torch.cat([torch.zeros(1, dtype=torch.long), seq_lens.cumsum(0)])
N_valid = int(sum(int(m.sum()) for m in resp_mask))
nest = lambda lst: torch.nested.as_nested_tensor(lst, layout=torch.jagged)

def make_td(extra=None):
    td = TensorDict({"prompts": nest(prompts), "responses": nest(responses), "input_ids": nest(input_ids),
                     "response_mask": nest(resp_mask), "loss_mask": nest(resp_mask), "old_log_probs": nest(old_lp),
                     "advantages": nest(adv), "values": nest(vals), "returns": nest(rets)}, batch_size=[B])
    tu.assign_non_tensor(td, dp_size=1, batch_num_tokens=N_valid, global_batch_size=B, **(extra or {}))
    return td

def actor_cfg(mode, lsf=None):
    return ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=-1, loss_agg_mode=mode,
                       loss_scale_factor=lsf, policy_loss=PolicyLossConfig(loss_mode="vanilla"))

def critic_cfg(mode, lsf=None):
    return CriticConfig(strategy="fsdp", ppo_micro_batch_size_per_gpu=-1, loss_agg_mode=mode, loss_scale_factor=lsf)

base_vals = -1.2 + 0.4 * torch.randn(int(offsets[-1]))

def leaf_for(leaf, rows):
    pieces = [leaf[offsets[i]: offsets[i + 1]] for i in rows]
    lens = torch.tensor([int(seq_lens[i]) for i in rows])
    return torch.nested.nested_tensor_from_jagged(torch.cat(pieces), torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)]))

def run(kind, mode, splits, lsf=None, extra=None):
    leaf = base_vals.clone().requires_grad_(True)
    td = make_td(extra)
    total = 0.0
    for rows in splits:
        mb = tu.index_select_tensor_dict(td, torch.tensor(rows))
        if kind == "actor":
            loss, _ = ppo_loss(actor_cfg(mode, lsf), {"log_probs": leaf_for(leaf, rows)}, mb)
        else:
            loss, _ = value_loss(critic_cfg(mode, lsf), {"values": leaf_for(leaf, rows)}, mb)
        loss.backward(); total += loss.item()
    return leaf.grad.clone(), total

SPLITS = {"whole": [list(range(B))], "split[1,2,5]": [[0], [1, 2], [3, 4, 5, 6, 7]], "split[4,4]": [[0, 1, 2, 3], [4, 5, 6, 7]], "singles": [[i] for i in range(B)]}
MODES = ["token-mean", "token-sum", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"]
print(f"B={B} resp_lens={resp_lens} valid={N_valid} configured response_length={RESPONSE_LENGTH}")
bad = 0
for kind in ("actor", "critic"):
    for mode in MODES:
        for lsf in ([None, RESPONSE_LENGTH] if mode == "seq-mean-token-sum-norm" else [None]):
            g_whole, l_whole = run(kind, mode, SPLITS["whole"], lsf)
            scale = g_whole.abs().max().item()
            for name in ("split[1,2,5]", "split[4,4]", "singles"):
                g, l = run(kind, mode, SPLITS[name], lsf)
                rel = (g - g_whole).abs().max().item() / scale
                flag = "OK " if rel < 1e-5 else "BAD"; bad += flag == "BAD"
                print(f"{flag} {kind:6s} {mode:24s} lsf={str(lsf):4s} {name:13s} loss split/whole={l / l_whole:6.3f} max rel grad err={rel:.2e}")
print(f"BAD cases: {bad}")

print("\n## fix candidate (i): inject max_response_len=response_length into the TensorDict (what left_right_2_no_padding does for V0)")
try:
    run("actor", "seq-mean-token-sum-norm", SPLITS["split[1,2,5]"], None, extra={"max_response_len": RESPONSE_LENGTH})
    print(" ppo_loss ran")
except Exception as e:
    print(f" ppo_loss FAILS: {type(e).__name__}: {str(e)[:200]}")
try:
    run("critic", "seq-mean-token-sum-norm", SPLITS["split[1,2,5]"], None, extra={"max_response_len": RESPONSE_LENGTH})
    print(" value_loss ran")
except Exception as e:
    print(f" value_loss FAILS: {type(e).__name__}: {str(e)[:200]}")

print("\n## V0 reference: with dense inputs padded to response_length, horizon = response_length for every micro-batch")
print(" (left_right_2_no_padding sets max_response_len = response_mask.shape[1] = padded width; agg_loss divides by loss_mask.shape[-1] = same width)")
