"""K3 oldlogprob-skip: CPU proof that the update pass's log-probs equal a separate
old-log-prob forward (same weights), including under micro-batching with gradient
accumulation, and that the K3 gradient (old_log_prob = log_prob.detach()) equals the
PPO gradient computed against the separately stored old_log_prob; plus the negative
control: after ONE optimizer step the next mini-batch's log-probs are no longer equal.

Run:  <repo>/.venv-a9/bin/python -I research/vet/K3/cpu_onpolicy_equivalence.py
Reads verl from the read-only clone (verl.trainer.ppo.core_algos, verl.utils.torch_functional).
"""

import sys

import torch
from omegaconf import OmegaConf
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.utils.torch_functional import logprobs_from_logits

torch.manual_seed(0)
B, P, R, V = 8, 12, 20, 512  # batch, prompt len, response len, vocab
cfg = Qwen3Config(
    vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
    attention_dropout=0.0, tie_word_embeddings=False,
)
model = Qwen3ForCausalLM(cfg).float()

# ragged responses: right-padded, response_mask marks real response tokens
input_ids = torch.randint(0, V, (B, P + R))
resp_len = torch.randint(R // 2, R + 1, (B,))
attention_mask = torch.ones(B, P + R, dtype=torch.long)
for i in range(B):
    attention_mask[i, P + resp_len[i]:] = 0
position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
response_mask = attention_mask[:, P:].clone()
advantages = torch.randn(B, R) * response_mask

actor_cfg = OmegaConf.create({
    "clip_ratio": 0.2, "clip_ratio_low": None, "clip_ratio_high": None, "clip_ratio_c": 3.0,
    "global_batch_info": {},
})


def response_log_probs(mb_idx):
    """log pi(response token | prefix) for the response positions, (len(mb_idx), R)."""
    out = model(input_ids=input_ids[mb_idx], attention_mask=attention_mask[mb_idx],
                position_ids=position_ids[mb_idx], use_cache=False)
    logits = out.logits[:, P - 1:P + R - 1, :]  # positions predicting response tokens
    labels = input_ids[mb_idx][:, P:P + R]
    return logprobs_from_logits(logits.float(), labels)


def chunks(idx, sizes):
    o, out = 0, []
    for s in sizes:
        out.append(idx[o:o + s]); o += s
    assert o == len(idx)
    return out


all_idx = torch.arange(B)

# ---- Pass A: the old_log_prob pass as verl runs it (eval, no_grad, its own micro-batching)
model.eval()
with torch.no_grad():
    old_lp = torch.cat([response_log_probs(mb) for mb in chunks(all_idx, [2, 2, 2, 2])])


def update_pass(use_stored_old, micro_sizes):
    """One mini-batch == the whole batch, micro-batched with gradient accumulation, no optimizer step."""
    model.train()
    model.zero_grad(set_to_none=True)
    lps, clipfracs, kls = [], [], []
    for mb in chunks(all_idx, micro_sizes):
        lp = response_log_probs(mb)
        old = old_lp[mb] if use_stored_old else lp.detach()  # K3: ratio == 1 by construction
        loss, m = compute_policy_loss_vanilla(
            old_log_prob=old, log_prob=lp, advantages=advantages[mb], response_mask=response_mask[mb].bool(),
            loss_agg_mode="token-mean", config=actor_cfg,
        )
        (loss * len(mb) / B).backward()  # same accumulation weighting in both variants
        lps.append(lp.detach()); clipfracs.append(m["actor/pg_clipfrac"]); kls.append(m["actor/ppo_kl"])
    grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    return torch.cat(lps), grads, max(clipfracs), max(abs(k) for k in kls)


# ---- Pass B: PPO as on main (stored old_log_prob), micro-batched 3/3/2 with grad accumulation
lp_B, grads_B, clip_B, kl_B = update_pass(use_stored_old=True, micro_sizes=[3, 3, 2])
# ---- Pass C: K3 (old_log_prob = log_prob.detach()), same micro-batching
lp_C, grads_C, clip_C, kl_C = update_pass(use_stored_old=False, micro_sizes=[3, 3, 2])

m = response_mask.bool()
d_lp = (lp_B - old_lp)[m].abs().max().item()
d_g = max((a - b).abs().max().item() for a, b in zip(grads_B, grads_C))
scale_g = max(a.abs().max().item() for a in grads_B)
print(f"[fp32] |update-pass log_prob - old_log_prob|_max over {int(m.sum())} response tokens = {d_lp:.3e}")
print(f"[fp32] main pg_clipfrac={clip_B:.3e} |ppo_kl|={kl_B:.3e}   K3 pg_clipfrac={clip_C:.3e} |ppo_kl|={kl_C:.3e}")
print(f"[fp32] |grad(main) - grad(K3)|_max = {d_g:.3e}  (max |grad| = {scale_g:.3e}, rel {d_g / scale_g:.3e})")
ok_fp32 = d_lp < 1e-5 and d_g / scale_g < 1e-5 and clip_C == 0.0 and kl_C == 0.0

# ---- Negative control: k = 2 mini-batches. Step the optimizer after mini-batch 1, then score mini-batch 2.
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train(); model.zero_grad(set_to_none=True)
mb1, mb2 = all_idx[:4], all_idx[4:]
lp1 = response_log_probs(mb1)
loss, _ = compute_policy_loss_vanilla(old_log_prob=old_lp[mb1], log_prob=lp1, advantages=advantages[mb1],
                                      response_mask=response_mask[mb1].bool(), loss_agg_mode="token-mean", config=actor_cfg)
loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
with torch.no_grad():
    lp2 = response_log_probs(mb2)
d2 = (lp2 - old_lp[mb2])[response_mask[mb2].bool()].abs().max().item()
ratio2 = torch.exp(lp2 - old_lp[mb2])[response_mask[mb2].bool()]
print(f"[k=2 control] after one AdamW step (lr 1e-4): |log_prob(mb2) - old_log_prob(mb2)|_max = {d2:.3e}, "
      f"ratio range [{ratio2.min():.4f}, {ratio2.max():.4f}]  -> second mini-batch is NOT on-policy")

# ---- bf16 variant: the #6280 effect (micro-batch shape noise) on CPU, same weights, no step
model_bf = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
model_bf.load_state_dict({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()})
model = model_bf


def bf_lp(mb):
    out = model(input_ids=input_ids[mb], attention_mask=attention_mask[mb], position_ids=position_ids[mb], use_cache=False)
    return logprobs_from_logits(out.logits[:, P - 1:P + R - 1, :].float(), input_ids[mb][:, P:P + R])


with torch.no_grad():
    model.eval(); a = torch.cat([bf_lp(mb) for mb in chunks(all_idx, [2, 2, 2, 2])])
    model.train(); b = torch.cat([bf_lp(mb) for mb in chunks(all_idx, [3, 3, 2])])
    model.train(); c = torch.cat([bf_lp(mb) for mb in chunks(all_idx, [2, 2, 2, 2])])
r_ab = torch.exp(b - a)[m]; r_ac = torch.exp(c - a)[m]
print(f"[bf16] micro-batch 2/2/2/2 (old pass) vs 3/3/2 (update pass): |dlogp|_max={(b - a)[m].abs().max():.3e}, "
      f"tokens with |ratio-1|>1e-3: {(r_ab - 1).abs().gt(1e-3).float().mean() * 100:.1f}%")
print(f"[bf16] same micro-batching 2/2/2/2 twice:                     |dlogp|_max={(c - a)[m].abs().max():.3e}, "
      f"tokens with |ratio-1|>1e-3: {(r_ac - 1).abs().gt(1e-3).float().mean() * 100:.1f}%")

print("RESULT:", "PASS" if ok_fp32 else "FAIL", "(fp32 equivalence of update-pass log-probs/gradient with the on-policy skip)")
sys.exit(0 if ok_fp32 else 1)
