"""CPU model of GRPO sibling placement: verl's GlobalRequestLoadBalancer vs a group-affinity policy.

Read-only against the clone: imports the real router class from
<repo>/verl (falls back to an inline copy of acquire/release when
ray/cachetools are not importable; the fallback is flagged in the output).

Usage:  python3 -I research/vet/K8/sim_routing.py [--seed 0]
"""

import argparse
import random
import statistics
import sys
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))  # the repository root

CLONE = os.path.join(_REPO, "verl")
sys.path.insert(0, CLONE)
try:
    from verl.workers.rollout.router import GlobalRequestLoadBalancer  # noqa: E402

    SRC = "verl.workers.rollout.router.GlobalRequestLoadBalancer (clone @ HEAD)"
except Exception as exc:  # pragma: no cover - env without ray
    SRC = f"INLINE COPY of router.py:170-217 (import failed: {exc.__class__.__name__}: {exc})"

    class GlobalRequestLoadBalancer:  # minimal faithful copy of router.py:156-217
        def __init__(self, servers, max_cache_size=10000, full_determinism=False):
            self._servers = dict(servers)
            self._inflight_requests = {sid: 0 for sid in servers}
            self._request_id_to_server = {}
            self._full_determinism = full_determinism

        def acquire_server(self, request_id):
            if request_id in self._request_id_to_server:
                server_id = self._request_id_to_server[request_id]
                if server_id in self._inflight_requests:
                    self._inflight_requests[server_id] += 1
                    return server_id, self._servers[server_id]
                del self._request_id_to_server[request_id]
            min_count = min(self._inflight_requests.values())
            candidates = [sid for sid, c in self._inflight_requests.items() if c == min_count]
            server_id = random.choice(candidates)
            self._request_id_to_server[request_id] = server_id
            self._inflight_requests[server_id] += 1
            return server_id, self._servers[server_id]

        def release_server(self, server_id, request_id=None):
            if self._inflight_requests.get(server_id, 0) > 0:
                self._inflight_requests[server_id] -= 1


class GroupAffinityLoadBalancer(GlobalRequestLoadBalancer):
    """The K8 proposal: first sibling least-inflight, later siblings follow the group's replica
    unless that replica is more than `slack` requests above the least-loaded one."""

    def __init__(self, servers, slack=8, **kw):
        super().__init__(servers, **kw)
        self._group_to_server = {}
        self.slack = slack

    def acquire_server(self, request_id, group_id=None):
        if group_id is not None and group_id in self._group_to_server:
            sid = self._group_to_server[group_id]
            if sid in self._inflight_requests:
                min_count = min(self._inflight_requests.values())
                if self._inflight_requests[sid] - min_count <= self.slack:
                    self._request_id_to_server[request_id] = sid
                    self._inflight_requests[sid] += 1
                    return sid, self._servers[sid]
        sid, h = super().acquire_server(request_id)
        if group_id is not None:
            self._group_to_server[group_id] = sid
        return sid, h


def run(policy, R, B, n, P, resp_mean, order, workers=8, slack=8, seed=0, streaming=False):
    """Returns dict of placement statistics for one rollout step.

    order: 'grouped' = each worker submits its prompts' n siblings back-to-back
           (agent_loop_tq.py:127-134); the `workers` workers' acquire calls interleave round-robin
           (one global Ray actor serialises them, router.py:133-141).
    streaming: if True, requests arrive while others finish (async modes) -> in-flight counts drift;
           modelled by releasing a random 30% of earlier requests before each new group is submitted.
    """
    random.seed(seed)
    servers = {f"s{i}": None for i in range(R)}
    lb = GroupAffinityLoadBalancer(servers, slack=slack) if policy == "affinity" else GlobalRequestLoadBalancer(servers)
    # group length structure: hard prompts -> long siblings (correlated), lognormal group mean, per-sibling jitter
    groups = []
    for g in range(B):
        gmean = resp_mean * random.lognormvariate(0, 0.5)
        lens = [max(16, min(8 * resp_mean, int(gmean * random.lognormvariate(0, 0.3)))) for _ in range(n)]
        groups.append(lens)
    # submission order
    per_worker = [list(range(w, B, workers)) for w in range(workers)]
    seq = []  # (group, sibling)
    cursors = [0] * workers
    while any(c < len(pw) for c, pw in zip(cursors, per_worker)):
        for w in range(workers):
            if cursors[w] < len(per_worker[w]):
                g = per_worker[w][cursors[w]]
                cursors[w] += 1
                for s in range(n):
                    seq.append((g, s))
    placement = {}
    assigned = []
    for g, s in seq:
        if streaming and assigned and s == 0:
            for sid, rid in random.sample(assigned, max(1, int(0.3 * len(assigned)))):
                lb.release_server(sid, rid)
            assigned = [a for a in assigned if a[0] is not None]
        rid = f"{g}-{s}"
        if policy == "affinity":
            sid, _ = lb.acquire_server(rid, group_id=str(g))
        else:
            sid, _ = lb.acquire_server(rid)
        placement[(g, s)] = sid
        assigned.append((sid, rid))
    distinct = [len({placement[(g, s)] for s in range(n)}) for g in range(B)]
    load = {sid: 0 for sid in servers}
    for (g, s), sid in placement.items():
        load[sid] += P + groups[g][s]
    mean_load = statistics.mean(load.values())
    redundant_prefill = sum((d - 1) * P for d in distinct)
    total_prefill = B * n * P
    decode = sum(sum(gl) for gl in groups)
    return {
        "distinct_mean": statistics.mean(distinct),
        "redundant_prefill_tokens": redundant_prefill,
        "ideal_prefill_tokens": B * P,
        "naive_prefill_tokens": total_prefill,
        "decode_tokens": decode,
        "load_max_over_mean": max(load.values()) / mean_load,
        "load_cv": statistics.pstdev(load.values()) / mean_load,
    }


SHAPES = [
    # name, replicas, prompts/step, n, prompt tokens, mean response tokens, model params (B), prefill tok/s/replica, decode tok/s/replica
    ("BASELINE-A gsm8k 1.7B TP=1", 8, 128, 8, 150, 600, 1.7, 120_000, 25_000),
    ("reference 8B TP=2 P1.5k R8k", 4, 512, 8, 1500, 4000, 8, 50_000, 15_000),
    ("agentic 8B TP=2 P8k R2k", 4, 256, 8, 8000, 2000, 8, 50_000, 15_000),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slack", type=int, default=8)
    args = ap.parse_args()
    print(f"router source: {SRC}")
    print(
        f"{'shape':34s} {'policy':9s} {'stream':6s} {'distinct/grp':>12s} {'redund.prefill':>14s} {'of prefill':>10s} "
        f"{'est. gen%':>9s} {'load max/mean':>13s} {'load CV':>8s}"
    )
    for name, R, B, n, P, resp, params_b, pre_tps, dec_tps in SHAPES:
        for streaming in (False, True):
            for policy in ("baseline", "affinity"):
                r = run(policy, R, B, n, P, resp, "grouped", slack=args.slack, seed=args.seed, streaming=streaming)
                # wall estimate per replica: prefill + decode time, redundant prefill removed by affinity
                t_prefill_red = r["redundant_prefill_tokens"] / R / pre_tps
                t_total = (r["naive_prefill_tokens"] / R / pre_tps) + (r["decode_tokens"] / R / dec_tps)
                print(
                    f"{name:34s} {policy:9s} {str(streaming):6s} {r['distinct_mean']:12.2f} "
                    f"{r['redundant_prefill_tokens']:14,d} {r['redundant_prefill_tokens'] / r['naive_prefill_tokens']:10.1%} "
                    f"{t_prefill_red / t_total:9.1%} {r['load_max_over_mean']:13.3f} {r['load_cv']:8.3f}"
                )
    print(
        "\n'est. gen%' = redundant prefill time / (all prefill + decode time) per replica at the stated tok/s; "
        "it is the upper bound of the gen-time saving from affinity (KV-concurrency effects not modelled)."
    )


if __name__ == "__main__":
    main()
