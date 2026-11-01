"""Prototype: exact vectorized (blockwise scan) replacements for the O(L) Python loops in
core_algos.compute_gae_advantage_return (:216) and compute_reinforce_plus_plus_outcome_advantage (:694),
with the same observation-token (mask=0) carry semantics, plus parity tests and timings.

Recurrence (reverse over t) in the loop version, with m_t = response_mask[:, t]:
  nextvalues_t = m_t * v_t + (1-m_t) * nextvalues_{t+1}          (value of the next valid token after t)
  delta_t      = r_t + gamma * nextvalues_{t+1} - v_t
  g_t          = m_t * (delta_t + gamma*lam * g_{t+1}) + (1-m_t) * g_{t+1}
=> g_t = a_t g_{t+1} + b_t with a_t = (gamma*lam)^{m_t}, b_t = m_t * delta_t, i.e. a first-order linear recurrence whose
   product of coefficients between positions is (gamma*lam)^(number of valid tokens in between). Solved in blocks of
   `block` positions with the closed form g_t = q^{-S_t} * revcumsum(b_j q^{S_j}) (S = exclusive prefix count of valid
   tokens inside the block) plus a per-block carry; the cross-block carry is a loop of L/block width-N steps.
"""

import math
import sys
import time

import torch

sys.path.insert(0, os.path.join(_REPO, "research/bench-a3"))
from bench_common import machine_banner, timeit  # noqa: E402

import verl.utils.torch_functional as verl_F  # noqa: E402
from verl.trainer.ppo import core_algos  # noqa: E402
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))  # the repository root


def _linear_scan_reverse(a_is_q_pow_mask: torch.Tensor, b: torch.Tensor, q: float, block: int = 256) -> torch.Tensor:
    """Solve g_t = q^{m_t} g_{t+1} + b_t (g_L = 0) for all t, vectorized. m = a_is_q_pow_mask (0/1), float64 math."""
    N, L = b.shape
    m = a_is_q_pow_mask.to(torch.float64)
    b = b.to(torch.float64)
    if q == 1.0:
        return torch.flip(torch.cumsum(torch.flip(b, [1]), 1), [1])
    # choose a block so that q^{-block} stays far from float64 overflow
    max_block = int(600.0 / max(-math.log10(q), 1e-12))
    block = max(1, min(block, max_block))
    nb = (L + block - 1) // block
    Lp = nb * block
    if Lp != L:
        m = torch.nn.functional.pad(m, (0, Lp - L))
        b = torch.nn.functional.pad(b, (0, Lp - L))
    m = m.view(N, nb, block)
    b = b.view(N, nb, block)
    # exclusive prefix count of valid tokens inside each block
    S = torch.cumsum(m, dim=2) - m
    total = S[:, :, -1] + m[:, :, -1]  # valid tokens per block
    qS = torch.pow(q, S)
    local = torch.flip(torch.cumsum(torch.flip(b * qS, [2]), 2), [2]) / qS  # q^{-S_t} * sum_{j>=t} b_j q^{S_j}
    # carry from the next block: g_t += q^{total - S_t} * G_{next}, G = g at position 0 of the next block
    carry = torch.pow(q, total.unsqueeze(2) - S)  # (N, nb, block)
    G = torch.zeros(N, dtype=torch.float64)
    out = torch.empty_like(local)
    for blk in range(nb - 1, -1, -1):
        out[:, blk] = local[:, blk] + carry[:, blk] * G.unsqueeze(1)
        G = out[:, blk, 0]
    return out.view(N, Lp)[:, :L]


def gae_scan(token_level_rewards, values, response_mask, gamma, lam):
    with torch.no_grad():
        N, L = values.shape
        m = response_mask.to(values.dtype)
        # next valid index after t (exclusive), L if none
        ar = torch.arange(L)
        idx_valid = torch.where(response_mask.bool(), ar.expand(N, L), torch.full((N, L), L))
        # cummin over [t+1, L): flip, cummin, flip, then shift left by one
        nxt = torch.flip(torch.cummin(torch.flip(idx_valid, [1]), dim=1).values, [1])
        nxt = torch.cat([nxt[:, 1:], torch.full((N, 1), L)], dim=1)
        has_next = nxt < L
        nextvalues = torch.where(has_next, torch.gather(values, 1, nxt.clamp(max=L - 1)), torch.zeros_like(values))
        delta = token_level_rewards + gamma * nextvalues - values
        g = _linear_scan_reverse(m, m * delta, gamma * lam)
        advantages = g.to(values.dtype)
        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


def rpp_scan(token_level_rewards, response_mask, gamma):
    with torch.no_grad():
        m = response_mask.to(token_level_rewards.dtype)
        running = _linear_scan_reverse(m, m * token_level_rewards, gamma)  # running_t carried through masked tokens
        returns = (running * m).to(token_level_rewards.dtype)
        advantages = verl_F.masked_whiten(returns, response_mask) * response_mask
    return advantages, returns


class Cfg:
    def __init__(self, gamma):
        self.gamma = gamma


machine_banner()
torch.manual_seed(0)


def make_mask(N, L, obs=True):
    lens = torch.clamp((torch.rand(N) ** 0.5 * L).long(), 1, L)
    mask = (torch.arange(L)[None, :] < lens[:, None]).float()
    if obs:  # observation (tool) spans inside the response: zero out random windows
        for _ in range(3):
            s = (torch.rand(N) * L).long()
            w = (torch.rand(N) * (L // 8)).long()
            span = (torch.arange(L)[None, :] >= s[:, None]) & (torch.arange(L)[None, :] < (s + w)[:, None])
            mask = mask * (~span).float()
    return mask


print("\n## parity vs the loops (float32 inputs; diff measured on valid positions and on returns everywhere)")
for N, L in [(64, 300), (256, 2048), (512, 8192)]:
    for gamma, lam in [(1.0, 1.0), (1.0, 0.95), (0.99, 0.95), (0.9, 0.9), (0.5, 0.5)]:
        mask = make_mask(N, L)
        r = torch.randn(N, L) * mask
        v = torch.randn(N, L) * mask
        a_loop, ret_loop = core_algos.compute_gae_advantage_return(r, v, mask, gamma, lam)
        a_vec, ret_vec = gae_scan(r, v, mask, gamma, lam)
        mb = mask.bool()
        da = (a_loop - a_vec)[mb].abs().max().item()
        dr = (ret_loop - ret_vec).abs().max().item()
        scale = ret_loop.abs().max().item()
        a_rpp, ret_rpp = core_algos.compute_reinforce_plus_plus_outcome_advantage(r, mask, config=Cfg(gamma))
        a_rv, ret_rv = rpp_scan(r, mask, gamma)
        drp = (ret_rpp - ret_rv).abs().max().item()
        dap = (a_rpp - a_rv)[mb].abs().max().item()
        print(
            f"N={N:4d} L={L:5d} gamma={gamma} lam={lam}: GAE adv max|diff|={da:.2e} returns max|diff|={dr:.2e} "
            f"(max|ret|={scale:.1f}) | RF++ returns max|diff|={drp:.2e} adv max|diff|={dap:.2e}"
        )

print("\n## timings")
for N, L in [(4096, 8192), (4096, 32768)]:
    mask = make_mask(N, L)
    r = torch.randn(N, L) * mask
    v = torch.randn(N, L) * mask
    timeit(lambda: gae_scan(r, v, mask, 1.0, 0.95), repeat=3, label=f"GAE blockwise scan N={N} L={L} (gamma*lam=0.95)")
    timeit(lambda: gae_scan(r, v, mask, 1.0, 1.0), repeat=3, label=f"GAE scan N={N} L={L} (gamma*lam=1: reverse cumsum)")
    timeit(lambda: rpp_scan(r, mask, 1.0), repeat=3, label=f"RF++ scan N={N} L={L} (gamma=1)")
    timeit(lambda: rpp_scan(r, mask, 0.99), repeat=3, label=f"RF++ scan N={N} L={L} (gamma=0.99)")
    if L <= 8192:
        timeit(lambda: core_algos.compute_gae_advantage_return(r, v, mask, 1.0, 0.95), repeat=1, warmup=0, label=f"GAE loop N={N} L={L}")
        timeit(lambda: core_algos.compute_reinforce_plus_plus_outcome_advantage(r, mask, config=Cfg(1.0)), repeat=1, warmup=0, label=f"RF++ loop N={N} L={L}")
