"""CPU model of GRESO-style pre-rollout prompt skipping on top of verl's V1 DAPO group filter.

Pure Python (no verl import). Models the trainer-side scheduler the K7 candidate proposes:
per-prompt consecutive zero-variance count z, skip probability 1 - p_e^z (GRESO Eq. 3, p_e per class
easy=all-correct / hard=all-wrong), p_e adapted by +-delta per step toward a target zero-variance share
(paper: 25% total, 1:2 easy:hard), floor 0.05. Baseline = verl's filter_groups alone (post-generation
eviction + refill, replay_buffer.py:442-500), which pays n rollouts for every zero-variance group.

Usage: python3 -I research/vet/K7/sim_prefilter.py [--dataset 7473 --batch 128 --n 8 --epochs 6]
"""

import argparse
import math
import random
import statistics


def logit(p):
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class PromptHistory:
    """The proposed scheduler state (what the PR would checkpoint): per prompt key -> (z, last_class)."""

    def __init__(self, p_easy=0.5, p_hard=0.5, target=0.25, delta=0.01, p_min=0.05):
        self.z = {}
        self.cls = {}
        self.p = {"easy": p_easy, "hard": p_hard}
        self.target = {"easy": target / 3, "hard": 2 * target / 3}
        self.delta, self.p_min = delta, p_min

    def skip(self, key):
        z = self.z.get(key, 0)
        if z == 0:
            return False
        return random.random() < 1 - self.p[self.cls[key]] ** z

    def observe(self, key, rewards):
        if len(set(rewards)) == 1:
            self.z[key] = self.z.get(key, 0) + 1
            self.cls[key] = "easy" if rewards[0] == 1 else "hard"
        else:
            self.z[key] = 0

    def adapt(self, n_easy, n_hard, n_total):
        for c, k in (("easy", n_easy), ("hard", n_hard)):
            share = k / max(1, n_total)
            self.p[c] = max(self.p_min, min(0.95, self.p[c] - self.delta if share >= self.target[c] else self.p[c] + self.delta))


def simulate(D, B, n, epochs, prefilter, seed, drift, easy_frac, hard_frac):
    """Returns per-epoch (groups generated per effective batch, skipped prompts, zero-variance share)."""
    random.seed(seed)
    # latent pass rates: a bimodal population (saturating math RL): easy ~Beta(8,1.2), hard ~Beta(1.2,8), mid ~Beta(3,3)
    pass0 = []
    for _ in range(D):
        u = random.random()
        if u < easy_frac:
            pass0.append(random.betavariate(8, 1.2))
        elif u < easy_frac + hard_frac:
            pass0.append(random.betavariate(1.2, 8))
        else:
            pass0.append(random.betavariate(3, 3))
    pass0 = [min(0.995, max(0.005, p)) for p in pass0]
    hist = PromptHistory()
    total_steps = epochs * D // B
    step = 0
    out = []
    for ep in range(epochs):
        order = list(range(D))
        random.shuffle(order)
        cursor = 0
        gen_groups = eff_groups = skipped = zv = 0
        steps_this_epoch = 0
        while cursor < D and steps_this_epoch < D // B:
            collected = 0
            n_easy = n_hard = n_total = 0
            while collected < B and cursor < D:
                key = order[cursor]
                cursor += 1
                if prefilter and hist.skip(key):
                    skipped += 1
                    continue
                p = sigmoid(logit(pass0[key]) + drift * step / max(1, total_steps))
                rewards = [1 if random.random() < p else 0 for _ in range(n)]
                gen_groups += 1
                n_total += 1
                hist.observe(key, rewards)
                if len(set(rewards)) == 1:
                    zv += 1
                    if rewards[0] == 1:
                        n_easy += 1
                    else:
                        n_hard += 1
                    continue  # DAPO filter_groups evicts it; verl refills (replay_buffer.py:475)
                collected += 1
                eff_groups += 1
            if prefilter:
                hist.adapt(n_easy, n_hard, n_total)
            step += 1
            steps_this_epoch += 1
        out.append(
            {
                "epoch": ep + 1,
                "steps": steps_this_epoch,
                "gen_per_eff": gen_groups / max(1, eff_groups),
                "zv_share": zv / max(1, gen_groups),
                "skipped": skipped,
                "rollouts": gen_groups * n,
                "p_easy": hist.p["easy"],
                "p_hard": hist.p["hard"],
                "dataset_exhausted": cursor >= D and steps_this_epoch < D // B,
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, default=7473, help="prompts (GSM8K train = 7473)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--drift", type=float, default=2.0, help="logit drift of pass rate over the run (learning)")
    ap.add_argument("--easy", type=float, default=0.45, help="fraction of the population that is near-saturated")
    ap.add_argument("--hard", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(
        f"D={args.dataset} B={args.batch} n={args.n} epochs={args.epochs} easy={args.easy} hard={args.hard} drift={args.drift}"
    )
    base = simulate(args.dataset, args.batch, args.n, args.epochs, False, args.seed, args.drift, args.easy, args.hard)
    pre = simulate(args.dataset, args.batch, args.n, args.epochs, True, args.seed, args.drift, args.easy, args.hard)
    print(f"{'epoch':>5} {'steps':>5} | {'DAPO gen/eff':>12} {'zv%':>6} {'rollouts':>9} | {'GRESO gen/eff':>13} {'zv%':>6} {'skipped':>7} {'rollouts':>9} {'p_easy':>6} {'p_hard':>6} | {'rollout saving':>14}")
    tb = tp = 0
    for b, p in zip(base, pre):
        tb += b["rollouts"]
        tp += p["rollouts"]
        print(
            f"{b['epoch']:>5} {b['steps']:>5} | {b['gen_per_eff']:>12.2f} {b['zv_share']:>6.1%} {b['rollouts']:>9,d} | "
            f"{p['gen_per_eff']:>13.2f} {p['zv_share']:>6.1%} {p['skipped']:>7,d} {p['rollouts']:>9,d} {p['p_easy']:>6.2f} {p['p_hard']:>6.2f} | "
            f"{1 - p['rollouts'] / b['rollouts']:>14.1%}"
        )
    print(f"total rollouts: DAPO {tb:,d}  GRESO {tp:,d}  saving {1 - tp / tb:.1%}  (epoch 1 saving is by construction 0)")
    exhausted = [p["epoch"] for p in pre if p["dataset_exhausted"]]
    if exhausted:
        print(f"note: with skipping, the epoch ran out of prompts before D//B steps in epochs {exhausted} (verl would roll the dataloader over, trainer_base.py:1473-1475)")


if __name__ == "__main__":
    main()
