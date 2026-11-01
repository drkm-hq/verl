#!/usr/bin/env python3
"""Per-step timing table for a pulled run: bench/results/<run-name>/.

    python3 bench/summarize.py bench/results/<run-name> [--keys timing_s/gen,timing_s/step,...]

Reads metrics.jsonl (verl's file logger); falls back to parsing `step:N - k:v - ...`
lines in train.log. Prints every step and the mean over steps 2..N (step 1 is warm-up).
"""

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_KEYS = [
    "timing_s/gen", "timing_s/old_log_prob", "timing_s/ref", "timing_s/adv", "timing_s/update_actor",
    "timing_s/step", "perf/throughput", "perf/total_num_tokens", "response_length/mean", "critic/score/mean",
]
# second table: lengths (K/T = response / (prompt + response)), memory and the correctness columns of an A/B
AUX_KEYS = [
    "prompt_length/mean", "response_length/mean", "K/T", "actor/perf/max_memory_allocated_gb",
    "actor/perf/max_memory_reserved_gb", "actor/entropy", "actor/pg_clipfrac", "actor/ppo_kl", "actor/kl_loss",
    "training/rollout_probs_diff_mean",
]
# third table: the multi-turn / turn-level-reward columns of the gsm8k-tool preset (P4) + validation accuracy
TURN_KEYS = [
    "num_turns/mean", "num_turns/max", "critic/score/mean", "critic/rewards/mean", "critic/advantages/mean",
    "critic/advantages/max", "critic/advantages/min", "critic/returns/mean", "actor/pg_loss", "actor/grad_norm",
    "val-core/openai/gsm8k/acc/mean@1", "val-core/openai/gsm8k/reward/mean@1", "val-aux/num_turns/mean",
]


def load_steps(run_dir: Path) -> list[dict]:
    mj = run_dir / "metrics.jsonl"
    steps = []
    if mj.exists():
        for line in mj.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = {"step": rec.get("step")}
            row.update({k: v for k, v in rec.get("data", {}).items() if isinstance(v, (int, float))})
            steps.append(row)
    if steps:
        return steps
    log = run_dir / "train.log"
    if log.exists():  # console logger: "step:3 - timing_s/gen:12.3 - ..."
        for line in log.read_text(errors="replace").splitlines():
            m = re.match(r".*?step:(\d+) - (.*)$", line)
            if not m or "timing_s/step" not in line:
                continue
            row = {"step": int(m.group(1))}
            for kv in m.group(2).split(" - "):
                if ":" in kv:
                    k, _, v = kv.rpartition(":")
                    try:
                        row[k.strip()] = float(v)
                    except ValueError:
                        pass
            steps.append(row)
    return steps


def table(steps: list[dict], keys: list[str]) -> str:
    keys = [k for k in keys if any(k in s for s in steps)]
    w = [max(len(k.split("/")[-1]), 10) for k in keys]
    head = "step  " + "  ".join(k.split("/")[-1].rjust(w[i]) for i, k in enumerate(keys))
    lines = [head, "-" * len(head)]
    for s in steps:
        lines.append(f"{s['step']:<4}  " + "  ".join(
            (f"{s[k]:.3f}" if isinstance(s.get(k), float) else str(s.get(k, "-"))).rjust(w[i]) for i, k in enumerate(keys)))
    warm = [s for s in steps if (s.get("step") or 0) >= 2]
    if warm:
        lines.append("-" * len(head))
        means = []
        for i, k in enumerate(keys):
            vals = [s[k] for s in warm if isinstance(s.get(k), (int, float))]
            means.append((f"{sum(vals) / len(vals):.3f}" if vals else "-").rjust(w[i]))
        lines.append(f"mean(2..{steps[-1]['step']})  " + "  ".join(means))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    ap.add_argument("--json", action="store_true", help="also print the steps as JSON")
    a = ap.parse_args()
    steps = load_steps(a.run_dir)
    if not steps:
        print(f"no step metrics found in {a.run_dir}", file=sys.stderr)
        return 1
    meta = a.run_dir / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        print(f"run={a.run_dir.name} gpu={m.get('gpu')} image={m.get('image')} src_git={m.get('src_git')} "
              f"digest_match={m.get('tree', {}).get('digest_match')}")
        print(f"model={m.get('cfg', {}).get('model')} cfg={json.dumps(m.get('cfg', {}), sort_keys=True)}")
    summary = a.run_dir / "summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        print(f"returncode={s.get('returncode')} wall_s={s.get('wall_s')} n_steps={len(s.get('steps', []))}")
        if s.get("chaos"):
            c = s["chaos"]
            print("chaos: " + json.dumps({k: c.get(k) for k in (
                "outcome", "steps_reached", "total_training_steps", "kill_to_next_step_s", "first_step_after_kill",
                "terminated_by", "wall_cap_s", "hang_s", "returncode")}))
            if c.get("kill"):
                k = c["kill"]
                print("kill:  " + json.dumps({"step_before": k.get("step_before"), "t_rel_s": k.get("t_rel_s"),
                                              "t_utc": k.get("t_utc"), "class_name": k.get("class_name"),
                                              "killed": (k.get("killed") or {}).get("name"),
                                              "pid_gone_after_s": k.get("pid_gone_after_s"), "n_alive": k.get("n_alive"),
                                              "rc": k.get("rc")}))
    for s in steps:
        if isinstance(s.get("prompt_length/mean"), (int, float)) and isinstance(s.get("response_length/mean"), (int, float)):
            s["K/T"] = s["response_length/mean"] / (s["prompt_length/mean"] + s["response_length/mean"])
    print(table(steps, a.keys.split(",")))
    if any(k in s for k in AUX_KEYS for s in steps):
        print()
        print(table(steps, AUX_KEYS))
    if any(k in s for k in TURN_KEYS for s in steps):
        print()
        print(table(steps, TURN_KEYS))
    for name in ("rollouts", "val"):  # per-sample dumps (trainer.rollout_data_dir / validation_data_dir): diagnostics means
        diag = (json.loads(summary.read_text()).get(f"{'rollout' if name == 'rollouts' else 'validation'}_diagnostics")
                if summary.exists() else None) or {}
        if diag:
            keys = sorted({k for row in diag.values() for k in row if k != "n"})
            print(f"\n{name}/<step>.jsonl diagnostics (means over the dumped trajectories):")
            print("step  n     " + "  ".join(k.replace("/mean", "").rjust(16) for k in keys))
            for step, row in diag.items():
                print(f"{step:<5} {row['n']:<5} " + "  ".join(
                    (f"{row[k]:.4f}" if isinstance(row.get(k), (int, float)) else "-").rjust(16) for k in keys))
    if a.json:
        print(json.dumps(steps, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
