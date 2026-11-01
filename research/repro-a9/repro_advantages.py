"""Advantage estimator edge cases (core_algos.py), CPU only."""

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_gpg_outcome_advantage,
    compute_grpo_outcome_advantage,
    compute_grpo_passk_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
    compute_reinforce_plus_plus_outcome_advantage,
    compute_remax_outcome_advantage,
    compute_rloo_outcome_advantage,
    compute_rloo_vectorized_outcome_advantage,
)

torch.manual_seed(0)


def last_token_reward(scores, T=8):
    r = torch.zeros(len(scores), T)
    r[:, -1] = torch.tensor(scores, dtype=torch.float32)
    return r


print("=" * 100)
print("A. GRPO: uniform fractional reward group -> advantage should be exactly 0 (core_algos.py:319-326, 352-354)")
for score, n in [(0.3, 16), (0.7, 64), (0.9, 64), (1.0, 16), (0.5, 8)]:
    r = last_token_reward([score] * n)
    m = torch.ones(n, 8)
    idx = np.array(["g0"] * n, dtype=object)
    a_loop, _ = compute_grpo_outcome_advantage(r.clone(), m, idx)
    a_vec, _ = compute_grpo_vectorized_outcome_advantage(r.clone(), m, idx)
    print(
        f"  score={score} n={n:3d}  loop adv[0]={a_loop[0, 0].item():+.5f}  vectorized adv[0]={a_vec[0, 0].item():+.5f}"
        f"   expected 0.0  {'BAD' if max(a_loop.abs().max(), a_vec.abs().max()) > 1e-6 else 'OK'}"
    )

print("=" * 100)
print("B. GRPO singleton group (n=1 or filtered): mean=0,std=1 -> advantage = raw score (core_algos.py:315-317)")
r = last_token_reward([0.7])
m = torch.ones(1, 8)
a, _ = compute_grpo_outcome_advantage(r.clone(), m, np.array(["g"], dtype=object))
av, _ = compute_grpo_vectorized_outcome_advantage(r.clone(), m, np.array(["g"], dtype=object))
print(f"  loop={a[0, 0].item():.6f} vectorized={av[0, 0].item():.6f} (raw score 0.7 / (1+1e-6))")

print("=" * 100)
print("C. RLOO singleton: scalar keeps raw score (core_algos.py:629-633), vectorized zeroes it (core_algos.py:864)")
a_s, _ = compute_rloo_outcome_advantage(r.clone(), m, np.array(["g"], dtype=object))
a_v, _ = compute_rloo_vectorized_outcome_advantage(r.clone(), m, np.array(["g"], dtype=object))
print(f"  scalar={a_s[0, 0].item():.4f} vectorized={a_v[0, 0].item():.4f}  {'BAD (disagree)' if abs(a_s[0, 0] - a_v[0, 0]) > 1e-6 else 'OK'}")

print("=" * 100)
print("D. Outcome score placed on a response_mask=0 tail token (agent_loop.py:1127-1129 puts it on the last")
print("   attention-mask token; a chat-template suffix / tool response there has response_mask=0)")
mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.float32)
on_tail = torch.tensor([[0, 0, 0, 0, 0, 1.0]])
on_action = torch.tensor([[0, 0, 0, 0, 1.0, 0]])
values = torch.zeros_like(mask)
cfg = OmegaConf.create({"gamma": 1.0, "lam": 1.0})
for name, rew in [("score on masked tail", on_tail), ("score on last generated token", on_action)]:
    gae, _ = compute_gae_advantage_return(rew, values, mask, gamma=1.0, lam=1.0)
    rpp, _ = compute_reinforce_plus_plus_outcome_advantage(rew, mask, config=cfg)
    remax, _ = compute_remax_outcome_advantage(rew, torch.zeros(1), mask, config=cfg)
    # GAE output is whitened; print the *returns* instead of whitened advantages for readability
    _, gae_ret = compute_gae_advantage_return(rew, values, mask, gamma=1.0, lam=1.0)
    grpo, _ = compute_grpo_outcome_advantage(torch.cat([rew, rew * 0]), torch.cat([mask, mask]), np.array(["g", "g"], dtype=object))
    print(f"  {name:30s} GAE returns={gae_ret[0].tolist()}")
    print(f"  {'':30s} REINFORCE++ returns-> {rpp[0].tolist()} (whitened)  ReMax adv={remax[0].tolist()}")
    print(f"  {'':30s} GRPO (sum over tokens) adv[0,0]={grpo[0, 0].item():+.4f}")

print("=" * 100)
print("E. GPG alpha = bsz / count_nonzero(scores) (core_algos.py:810-811) depends on the reward encoding")
m4 = torch.ones(8, 8)
idx = np.array(["A"] * 4 + ["B"] * 4, dtype=object)
r01 = last_token_reward([1, 0, 0, 0, 0, 0, 0, 0])
rpm = last_token_reward([1, -1, -1, -1, -1, -1, -1, -1])
a01, _ = compute_gpg_outcome_advantage(r01.clone(), m4, idx)
apm, _ = compute_gpg_outcome_advantage(rpm.clone(), m4, idx)
print(f"  {{0,1}} encoding: adv of the correct sample = {a01[0, 0].item():.3f}  (alpha=8/1=8, paper alpha=B/(B-M)=8/4=2 -> 1.5)")
print(f"  {{-1,+1}} encoding (same outcomes): adv = {apm[0, 0].item():.3f}  (alpha=8/8=1, paper -> 2*1.5=3.0)")

print("=" * 100)
print("F. grpo_passk with a singleton group raises (core_algos.py:516-519)")
try:
    compute_grpo_passk_outcome_advantage(r.clone(), m, np.array(["g"], dtype=object), config=OmegaConf.create({"norm_adv_by_std_in_grpo": True}))
    print("  no error")
except ValueError as e:
    print(f"  ValueError: {str(e)[:70]}")

print("=" * 100)
print("G. GRPO loop vs vectorized on random groups (unbiased std both): max |diff|")
scores = torch.rand(32)
r = last_token_reward(scores.tolist())
m = torch.ones(32, 8)
idx = np.array([f"g{i // 8}" for i in range(32)], dtype=object)
a1, _ = compute_grpo_outcome_advantage(r.clone(), m, idx)
a2, _ = compute_grpo_vectorized_outcome_advantage(r.clone(), m, idx)
print(f"  {(a1 - a2).abs().max().item():.2e}")

print("=" * 100)
print("H. REINFORCE++ / GAE whitening with a single valid token in the whole batch -> masked_var raises")
mask1 = torch.zeros(2, 4)
mask1[0, 0] = 1
try:
    compute_reinforce_plus_plus_outcome_advantage(torch.ones(2, 4), mask1, config=cfg)
    print("  no error")
except ValueError as e:
    print(f"  ValueError: {e}")

print("=" * 100)
print("I. GAE returns/advantages at masked (tool) positions are NOT zeroed (core_algos.py:259-263); the policy/critic")
print("   losses mask them later, so this is harmless for training but 'returns' at masked slots are garbage.")
mask = torch.tensor([[1, 1, 0, 0, 1, 1]], dtype=torch.float32)
rew = torch.tensor([[0, 0, 0, 0, 0, 1.0]])
vals = torch.tensor([[0.1, 0.2, 9.0, 9.0, 0.3, 0.4]])
adv, ret = compute_gae_advantage_return(rew, vals, mask, gamma=1.0, lam=1.0)
print(f"  returns={ret[0].tolist()}")
print(f"  whitened adv (unmasked) ={[round(x, 3) for x in adv[0].tolist()]}")
