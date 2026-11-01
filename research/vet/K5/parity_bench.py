"""K5 vet: independent parity + benchmark of the blockwise scan vs the O(L) loops in core_algos.

Scan functions copied from research/bench-a3/bench_gae_vec.py (that module benchmarks at import time) and
extended with a dtype parameter (fp32 vs fp64 math) plus a deliberately naive global weighted-cumsum variant
to show why the un-blocked "cumsum on gamma*lam-weighted terms" alternative is not viable.
Diffs are measured EVERYWHERE (valid and masked positions), with NaN==NaN treated as equal.
"""
import math, sys, time
import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo import core_algos

torch.set_num_threads(max(1, torch.get_num_threads()))

def _linear_scan_reverse(mask, b, q, block=256, dtype=torch.float64):
    N, L = b.shape
    m = mask.to(dtype); b = b.to(dtype)
    if q == 1.0:
        return torch.flip(torch.cumsum(torch.flip(b, [1]), 1), [1])
    # keep q^block >= 2^-60 (fp32) / 2^-600 (fp64) so q^{-S} never overflows and q^{S} never underflows
    budget = 60.0 if dtype == torch.float32 else 600.0
    max_block = int(budget * math.log(2) / max(-math.log(q), 1e-12))
    block = max(1, min(block, max_block))
    nb = (L + block - 1) // block; Lp = nb * block
    if Lp != L:
        m = torch.nn.functional.pad(m, (0, Lp - L)); b = torch.nn.functional.pad(b, (0, Lp - L))
    m = m.view(N, nb, block); b = b.view(N, nb, block)
    S = torch.cumsum(m, dim=2) - m
    total = S[:, :, -1] + m[:, :, -1]
    qS = torch.pow(torch.tensor(q, dtype=dtype), S)
    local = torch.flip(torch.cumsum(torch.flip(b * qS, [2]), 2), [2]) / qS
    carry = torch.pow(torch.tensor(q, dtype=dtype), total.unsqueeze(2) - S)
    G = torch.zeros(N, dtype=dtype); out = torch.empty_like(local)
    for blk in range(nb - 1, -1, -1):
        out[:, blk] = local[:, blk] + carry[:, blk] * G.unsqueeze(1)
        G = out[:, blk, 0]
    return out.view(N, Lp)[:, :L]

def _global_weighted_cumsum(mask, b, q, dtype):
    """The naive alternative: one cumsum over q^{S_t}-weighted terms across the whole L. Overflows."""
    m = mask.to(dtype); b = b.to(dtype)
    S = torch.cumsum(m, 1) - m
    qS = torch.pow(torch.tensor(q, dtype=dtype), S)
    return torch.flip(torch.cumsum(torch.flip(b * qS, [1]), 1), [1]) / qS

def gae_scan(r, v, mask, gamma, lam, dtype=torch.float64, scan=_linear_scan_reverse):
    with torch.no_grad():
        N, L = v.shape
        m = mask.to(v.dtype)
        ar = torch.arange(L)
        idx_valid = torch.where(mask.bool(), ar.expand(N, L), torch.full((N, L), L))
        nxt = torch.flip(torch.cummin(torch.flip(idx_valid, [1]), dim=1).values, [1])
        nxt = torch.cat([nxt[:, 1:], torch.full((N, 1), L)], dim=1)
        has_next = nxt < L
        nextvalues = torch.where(has_next, torch.gather(v, 1, nxt.clamp(max=L - 1)), torch.zeros_like(v))
        delta = r + gamma * nextvalues - v
        g = scan(m, m * delta, gamma * lam, dtype=dtype) if scan is _linear_scan_reverse else scan(m, m * delta, gamma * lam, dtype)
        adv = g.to(v.dtype)
        ret = adv + v
        adv = verl_F.masked_whiten(adv, mask)
    return adv, ret

def rpp_scan(r, mask, gamma, dtype=torch.float64):
    with torch.no_grad():
        m = mask.to(r.dtype)
        running = _linear_scan_reverse(m, m * r, gamma, dtype=dtype)
        ret = (running * m).to(r.dtype)
        adv = verl_F.masked_whiten(ret, mask) * mask
    return adv, ret

class Cfg:
    def __init__(self, gamma): self.gamma = gamma

def maxdiff(a, b):
    both_nan = a.isnan() & b.isnan()
    d = (a - b).abs()
    d[both_nan] = 0
    return d.max().item()

def make_mask(N, L, gen, holes=3, special=True):
    lens = torch.clamp((torch.rand(N, generator=gen) ** 0.5 * L).long(), 1, L)
    mask = (torch.arange(L)[None, :] < lens[:, None]).float()
    for _ in range(holes):
        s = (torch.rand(N, generator=gen) * L).long(); w = (torch.rand(N, generator=gen) * max(1, L // 8)).long()
        span = (torch.arange(L)[None, :] >= s[:, None]) & (torch.arange(L)[None, :] < (s + w)[:, None])
        mask = mask * (~span).float()
    if special and N >= 5:
        mask[0] = 0.0                       # all-masked row
        mask[1] = 1.0                       # all-valid row
        mask[2] = 0.0; mask[2, 0] = 1.0     # only first token valid
        mask[3] = 1.0; mask[3, : max(1, L // 3)] = 0.0   # leading observation span
        mask[4] = 1.0; mask[4, L - max(1, L // 3):] = 0.0  # trailing padding only
    return mask

if __name__ == "__main__":
    print(f"torch {torch.__version__} threads {torch.get_num_threads()}")
    print("\n## parity: max|adv diff| / max|ret diff| over ALL positions; (fp64 scan, fp32 scan); rel = ret diff / max|ret|")
    worst = {"a64": 0, "r64": 0, "a32": 0, "r32": 0, "rel32": 0, "rp32": 0, "ap32": 0}
    shapes = [(2, 1), (5, 1), (8, 2), (8, 7), (5, 255), (5, 256), (5, 257), (6, 513), (16, 300), (64, 1000), (256, 2048), (128, 8192)]
    gl = [(1.0, 1.0), (1.0, 0.95), (0.99, 0.95), (0.9, 0.9), (0.5, 0.5), (0.3, 1.0)]
    n_cases = 0
    for seed in (0, 1, 2):
        gen = torch.Generator().manual_seed(seed)
        for N, L in shapes:
            mask = make_mask(N, L, gen)
            r = torch.randn(N, L, generator=gen) * mask; v = torch.randn(N, L, generator=gen)  # values NOT masked: loop reads v at masked t too
            r[:, -1] += 5.0 * torch.rand(N, generator=gen)  # outcome-style reward at the end
            for gamma, lam in gl:
                a_loop, ret_loop = core_algos.compute_gae_advantage_return(r, v, mask, gamma, lam)
                a64, ret64 = gae_scan(r, v, mask, gamma, lam, torch.float64)
                a32, ret32 = gae_scan(r, v, mask, gamma, lam, torch.float32)
                scale = ret_loop[~ret_loop.isnan()].abs().max().item() if (~ret_loop.isnan()).any() else 1.0
                worst["a64"] = max(worst["a64"], maxdiff(a_loop, a64)); worst["r64"] = max(worst["r64"], maxdiff(ret_loop, ret64))
                worst["a32"] = max(worst["a32"], maxdiff(a_loop, a32)); worst["r32"] = max(worst["r32"], maxdiff(ret_loop, ret32))
                worst["rel32"] = max(worst["rel32"], maxdiff(ret_loop, ret32) / scale)
                ap, rp = core_algos.compute_reinforce_plus_plus_outcome_advantage(r, mask, config=Cfg(gamma))
                ap32, rp32 = rpp_scan(r, mask, gamma, torch.float32)
                worst["rp32"] = max(worst["rp32"], maxdiff(rp, rp32)); worst["ap32"] = max(worst["ap32"], maxdiff(ap, ap32))
                n_cases += 1
                if seed == 0 and (N, L) in [(1, 1), (5, 1), (8, 7), (5, 257), (128, 8192)]:
                    print(f"seed0 N={N:4d} L={L:5d} g={gamma} l={lam}: GAE adv {maxdiff(a_loop, a64):.1e}/{maxdiff(a_loop, a32):.1e} ret {maxdiff(ret_loop, ret64):.1e}/{maxdiff(ret_loop, ret32):.1e} (max|ret| {scale:.1f}) | RF++ fp32 ret {maxdiff(rp, rp32):.1e} adv {maxdiff(ap, ap32):.1e} | nan_adv_loop={a_loop.isnan().any().item()}")
    print(f"cases={n_cases}  WORST: GAE adv fp64 {worst['a64']:.2e} fp32 {worst['a32']:.2e} | ret fp64 {worst['r64']:.2e} fp32 {worst['r32']:.2e} (rel {worst['rel32']:.2e}) | RF++ fp32 ret {worst['rp32']:.2e} adv {worst['ap32']:.2e}")

    print("\n## single valid token in the whole batch: both raise verl's masked_var guard?")
    mask = torch.zeros(1, 1); mask[0, 0] = 1.0; r = torch.randn(1, 1); v = torch.randn(1, 1)
    errs = []
    for fn in (lambda: core_algos.compute_gae_advantage_return(r, v, mask, 1.0, 0.95), lambda: gae_scan(r, v, mask, 1.0, 0.95, torch.float32)):
        try: fn(); errs.append("no error")
        except ValueError as e: errs.append(f"ValueError: {e}")
    print(" loop:", errs[0], "| scan:", errs[1])
    print("\n## all-masked batch (every row masked): both NaN?")
    mask = torch.zeros(4, 16); r = torch.randn(4, 16); v = torch.randn(4, 16)
    errs = []
    for fn in (lambda: core_algos.compute_gae_advantage_return(r, v, mask, 1.0, 0.95), lambda: gae_scan(r, v, mask, 1.0, 0.95, torch.float32)):
        try: fn(); errs.append("no error")
        except ValueError as e: errs.append(f"ValueError: {e}")
    print(" loop:", errs[0], "| scan:", errs[1])

    print("\n## naive global weighted cumsum (no blocking): fp32 and fp64 at q=0.95")
    gen = torch.Generator().manual_seed(0)
    for N, L in [(8, 512), (8, 2048), (8, 8192), (8, 32768)]:
        mask = make_mask(N, L, gen, special=False); r = torch.randn(N, L, generator=gen) * mask; v = torch.randn(N, L, generator=gen)
        a_loop, ret_loop = core_algos.compute_gae_advantage_return(r, v, mask, 1.0, 0.95)
        for dt in (torch.float32, torch.float64):
            a_n, ret_n = gae_scan(r, v, mask, 1.0, 0.95, dt, scan=_global_weighted_cumsum)
            print(f"L={L:5d} {str(dt):14s} ret maxdiff={maxdiff(ret_loop, ret_n):.2e} nan/inf={(~torch.isfinite(ret_n)).any().item()}")

    print("\n## timings (median of 3 for scans, single run for loops); realistic masks with holes")
    def timeit(fn, repeat=3):
        ts = []
        for _ in range(repeat):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        ts.sort(); return ts[len(ts) // 2]
    gen = torch.Generator().manual_seed(0)
    for N, L in [(1024, 8192), (4096, 8192), (4096, 32768)]:
        mask = make_mask(N, L, gen, special=False); r = torch.randn(N, L, generator=gen) * mask; v = torch.randn(N, L, generator=gen)
        t_loop = timeit(lambda: core_algos.compute_gae_advantage_return(r, v, mask, 1.0, 0.95), repeat=1)
        t_rloop = timeit(lambda: core_algos.compute_reinforce_plus_plus_outcome_advantage(r, mask, config=Cfg(1.0)), repeat=1)
        rows = [("GAE loop (q=0.95)", t_loop), ("RF++ loop (gamma=1)", t_rloop)]
        for q, nm in [(1.0, "q=1"), (0.95, "q=0.95")]:
            for dt in (torch.float32, torch.float64):
                rows.append((f"GAE scan {nm} {str(dt)[6:]}", timeit(lambda: gae_scan(r, v, mask, 1.0, q, dt))))
        for g, nm in [(1.0, "gamma=1"), (0.99, "gamma=0.99")]:
            rows.append((f"RF++ scan {nm} fp32", timeit(lambda: rpp_scan(r, mask, g, torch.float32))))
        print(f"-- N={N} L={L}")
        for nm, t in rows:
            print(f"   {nm:32s} {t*1000:9.1f} ms   speedup vs loop {(t_loop if nm.startswith('GAE') else t_rloop)/t:6.1f}x")
