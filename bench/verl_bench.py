#!/usr/bin/env python3
"""CLI wrapper for the Modal verl harness (bench/modal_verl.py).

    python3 bench/verl_bench.py probe      [--src DIR] [--gpu H100]
    python3 bench/verl_bench.py download   [--models a,b] [--no-gsm8k]
    python3 bench/verl_bench.py build-dataset [--target-tokens 6000]   # synthetic long-prompt GSM8K -> /hf-cache/data/gsm8k-long
    python3 bench/verl_bench.py build-tool-dataset                      # verl's tool-calling GSM8K -> /hf-cache/data/gsm8k-tool
    python3 bench/verl_bench.py train      --run-name NAME [--preset smoke|baseline-a|baseline-a-k1|long-prompt|separate-async-2gpu|gsm8k-tool]
                                           [--src DIR] [--gpu-type H100] [--chaos --chaos-step 2] [...]
    python3 bench/verl_bench.py gpu-tests  [--run-name NAME] [--tests tests/] [--pytest-extra "..."] [--gpu H100:8]
    python3 bench/verl_bench.py logs APP_ID [--tail 60] [--seconds 120]
    python3 bench/verl_bench.py wait FC_ID [--seconds 240]
    python3 bench/verl_bench.py pull RUN_NAME
    python3 bench/verl_bench.py stop APP_ID

Every launch is `modal run --detach` with MODAL_PROFILE set to your Modal profile; the app id is appended to
bench/LIVE-APPS.md before this script returns. Nothing here touches other apps.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
MODAL = ROOT / ".venv-bench" / "bin" / "modal"
PY = ROOT / ".venv-bench" / "bin" / "python"
DEFAULT_SRC = ROOT / "verl"
LIVE = BENCH / "LIVE-APPS.md"
RESULTS = BENCH / "results"
HARNESS = BENCH / "modal_verl.py"
RESULTS_VOLUME = "verl-contrib-results"

PRESETS = {
    # smoke: 1xH100, Qwen2.5-0.5B, GRPO n=4, bs 16, 512/512, 3 steps, TP=1
    "smoke": dict(model="Qwen/Qwen2.5-0.5B", n_gpus=1, rollout_n=4, train_batch_size=16, max_prompt_length=512,
                  max_response_length=512, total_training_steps=3, tp=1, use_kl_loss=True, gpu="H100"),
    # BASELINE-A: 8xH100, Qwen3-1.7B, GRPO n=8, bs 128, 1024/2048, 6 steps, TP=1, use_kl_loss=True
    "baseline-a": dict(model="Qwen/Qwen3-1.7B", n_gpus=8, rollout_n=8, train_batch_size=128, max_prompt_length=1024,
                       max_response_length=2048, total_training_steps=6, tp=1, use_kl_loss=True, gpu="H100:8"),
    # P2 (old-log-prob skip): BASELINE-A with one mini-batch per step (k=1, on-policy regime)
    "baseline-a-k1": dict(model="Qwen/Qwen3-1.7B", n_gpus=8, rollout_n=8, train_batch_size=128, ppo_mini_batch_size=128,
                          max_prompt_length=1024, max_response_length=2048, total_training_steps=6, tp=1, use_kl_loss=True,
                          gpu="H100:8"),
    # P1 (loss-row-only LM head): ~6k-token synthetic prompts (build-dataset), 1k responses, K/T ~ 0.14
    "long-prompt": dict(model="Qwen/Qwen3-1.7B", n_gpus=8, rollout_n=8, train_batch_size=64, ppo_mini_batch_size=16,
                        max_prompt_length=6144, max_response_length=1024, total_training_steps=6, tp=1, use_kl_loss=True,
                        ppo_max_token_len_per_gpu=32768, data_dir="/hf-cache/data/gsm8k-long", gpu="H100:8"),
    # P3 (self-healing): V1 separate_async, 1 trainer GPU + 1 standalone vLLM replica GPU in one container
    # (mirrors tests/special_e2e/run_v1_separate_async.sh); train_batch_size = parameter_sync_step x ppo_mini_batch_size
    "separate-async-2gpu": dict(
        model="Qwen/Qwen3-0.6B", n_gpus=2, trainer_n_gpus=1, rollout_n=4, train_batch_size=16, ppo_mini_batch_size=8,
        max_prompt_length=512, max_response_length=512, total_training_steps=6, tp=1, use_kl_loss=False, gpu="H100:2",
        overrides=" ".join([
            "trainer.use_v1=True", "trainer.v1.trainer_mode=separate_async",
            "trainer.v1.separate_async.num_warmup_batches=1", "trainer.v1.separate_async.parameter_sync_step=2",
            "trainer.v1.sampler.max_off_policy_threshold=8", "trainer.v1.sampler.max_off_policy_strategy=drop",
            "transfer_queue.enable=True", "data.prompt_key=prompt", "data.truncation=left", "data.return_raw_chat=True",
            "data.gen_batch_size=1", "algorithm.rollout_correction.bypass_mode=False",
            "actor_rollout_ref.hybrid_engine=False", "actor_rollout_ref.rollout.nnodes=1",
            "actor_rollout_ref.rollout.n_gpus_per_node=1", "actor_rollout_ref.rollout.calculate_log_probs=True",
            "actor_rollout_ref.rollout.enable_chunked_prefill=True",
            "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
            "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024",
            "actor_rollout_ref.rollout.agent.num_workers=4", "actor_rollout_ref.actor.strategy=fsdp2",
            "actor_rollout_ref.actor.optim.lr_warmup_steps=-1", "actor_rollout_ref.actor.ppo_epochs=1",
            "actor_rollout_ref.actor.clip_ratio_low=0.2", "actor_rollout_ref.actor.clip_ratio_high=0.28",
            "reward.reward_manager.name=dapo", "trainer.resume_mode=disable",
        ])),
    # P4 (turn-level rewards): verl's tool-calling GSM8K through the ToolAgentLoop (vLLM async, V1 sync trainer,
    # TransferQueue), Qwen3-1.7B, 64 prompts x 8, one mini-batch. `extras` = bench/gsm8k_tool/ is copied into
    # <src>/bench_extras/ at launch (the Gsm8kTool, its tool config and the custom reward function); arms:
    #   ENABLED: --extra-overrides "actor_rollout_ref.rollout.multi_turn.turn_level_rewards=True algorithm.adv_estimator=turn_rloo"
    #   CONTROL: --extra-overrides "algorithm.adv_estimator=grpo"
    "gsm8k-tool": dict(
        model="Qwen/Qwen3-1.7B", n_gpus=8, rollout_n=8, train_batch_size=64, ppo_mini_batch_size=64,
        max_prompt_length=1024, max_response_length=1024, total_training_steps=6, tp=1, use_kl_loss=True,
        data_dir="/hf-cache/data/gsm8k-tool", gpu="H100:8", extras="gsm8k_tool",
        overrides=" ".join([
            "trainer.use_v1=True", "trainer.v1.trainer_mode=sync", "transfer_queue.enable=True",
            "data.prompt_key=prompt", "data.return_raw_chat=True",
            "actor_rollout_ref.rollout.mode=async", "actor_rollout_ref.rollout.calculate_log_probs=True",
            "actor_rollout_ref.rollout.multi_turn.enable=True", "actor_rollout_ref.rollout.multi_turn.format=hermes",
            "actor_rollout_ref.rollout.multi_turn.tool_config_path=/workspace/verl/bench_extras/gsm8k_tool_config.yaml",
            "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3",
            "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
            "reward.custom_reward_function.path=/workspace/verl/bench_extras/gsm8k_tool_reward.py",
            "reward.custom_reward_function.name=compute_score",
            "trainer.rollout_data_dir={run_dir}/rollouts", "trainer.validation_data_dir={run_dir}/val",
        ])),
}
CFG_KEYS = ("model", "n_gpus", "rollout_n", "train_batch_size", "max_prompt_length", "max_response_length",
            "total_training_steps", "tp", "use_kl_loss", "ppo_mini_batch_size", "ppo_max_token_len_per_gpu",
            "gpu_memory_utilization", "overrides", "data_dir", "trainer_n_gpus", "chaos", "chaos_step", "chaos_class",
            "chaos_wall_cap_s", "chaos_hang_s")


def tree_digest(root: Path, rels: list[str]) -> str:
    """Same function as modal_verl.tree_digest: sha256 over sorted 'path\\0sha256(content)' lines."""
    h = hashlib.sha256()
    for rel in sorted(rels):
        p = root / rel
        if p.is_file():
            h.update(f"{rel}\0{hashlib.sha256(p.read_bytes()).hexdigest()}\n".encode())
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_describe(src: Path) -> str:
    try:
        sha = subprocess.run(["git", "-C", str(src), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(src), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        branch = subprocess.run(["git", "-C", str(src), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        return f"{branch}@{sha}{'-dirty' if dirty else ''}"
    except Exception:  # noqa: BLE001
        return "no-git"


def shipped_files(src: Path) -> list[str]:
    """Paths (relative, posix) the mount ships: git-tracked + untracked-not-ignored, minus MOUNT_IGNORE-like dirs."""
    p = subprocess.run(["git", "-C", str(src), "ls-files", "--cached", "--others", "--exclude-standard"],
                       capture_output=True, text=True)
    files = p.stdout.splitlines() if p.returncode == 0 else [x.relative_to(src).as_posix() for x in src.rglob("*") if x.is_file()]
    skip = (".git/", "__pycache__/", ".venv", "docs/_build/", ".pytest_cache/", ".ruff_cache/", ".mypy_cache/", "checkpoints/", "outputs/")
    return sorted(f for f in files if (src / f).is_file() and not f.endswith(".pyc") and not any(
        f.startswith(sk) or f"/{sk}" in f"/{f}" for sk in skip))


def record(app_id: str, app_name: str, purpose: str, state: str, fc_id: str = "") -> None:
    with open(LIVE, "a") as f:
        f.write(f"    {now()} | {app_id} | {app_name} | {purpose} | {state} | {fc_id}\n")


def modal_cmd(*args: str, env_extra: dict | None = None, seconds: int | None = None) -> subprocess.CompletedProcess:
    # PYTHONUNBUFFERED: `modal app logs` is killed by the alarm; block-buffered output would be lost for quiet apps
    env = {**os.environ, "PYTHONUNBUFFERED": "1", **(env_extra or {})}
    cmd = [str(MODAL), *args]
    if seconds:  # macOS has no coreutils timeout; perl alarm does the job
        cmd = ["perl", "-e", "alarm shift; exec @ARGV", str(seconds), *cmd]
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def launch(entrypoint: str, flags: list[str], purpose: str, src: Path, gpu: str, app_name: str,
           extra_env: dict | None = None) -> str:
    src = src.resolve()
    (RESULTS / purpose).mkdir(parents=True, exist_ok=True)
    manifest = RESULTS / purpose / "manifest.txt"
    files = shipped_files(src)
    manifest.write_text("\n".join(files) + "\n")
    digest = RESULTS / purpose / "digest.txt"  # the container recomputes it after _exact_tree(): meta.json tree.digest_match
    digest.write_text(tree_digest(src, files) + "\n")
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "no-git"
    ahead = subprocess.run(["git", "-C", str(src), "rev-list", "--count", "main..HEAD"], capture_output=True, text=True).stdout.strip()
    env = {"VERL_SRC": str(src), "VERL_SRC_GIT": git_describe(src), "VERL_SRC_HEAD": head, "VERL_SRC_AHEAD": ahead or "?",
           "VERL_GPU": gpu, "VERL_APP_NAME": app_name, "VERL_MANIFEST": str(manifest), "VERL_DIGEST": str(digest),
           **(extra_env or {})}
    args = ["run", "--detach", f"{HARNESS}::{entrypoint}", *flags]
    print(f"[{now()}] MODAL_PROFILE={os.environ.get('MODAL_PROFILE', '')} {' '.join(f'{k}={v}' for k, v in env.items())} modal {' '.join(args)}")
    (RESULTS / purpose).mkdir(parents=True, exist_ok=True)
    log = RESULTS / purpose / "launch.log"
    with open(log, "w") as lf:  # streamed, so a backgrounded launch can be polled with tail
        subprocess.run(["perl", "-e", "alarm shift; exec @ARGV", "1500", str(MODAL), *args],
                       env={**os.environ, **env}, stdout=lf, stderr=subprocess.STDOUT, text=True)
    out = log.read_text()
    m = re.search(r"APP_ID=(\S+) FC_ID=(\S+)", out)
    if not m:
        print(out[-3000:])
        print("LAUNCH FAILED: no APP_ID in output", file=sys.stderr)
        sys.exit(1)
    app_id, fc_id = m.group(1), m.group(2)
    record(app_id, app_name, purpose, "running", fc_id)
    print(f"APP_ID={app_id} FC_ID={fc_id} recorded in {LIVE}")
    return app_id


def cmd_probe(a) -> None:
    launch("probe_launch", ["--run-name", a.run_name], a.run_name, Path(a.src), a.gpu, "verl-contrib-probe")


def cmd_download(a) -> None:
    flags = ["--models", a.models, "--run-name", a.run_name] + (["--no-gsm8k"] if a.no_gsm8k else [])
    launch("download_launch", flags, a.run_name, Path(a.src), "H100", "verl-contrib-download")


def ship_extras(src: Path, extras: str, purpose: str) -> None:
    """Copy bench/<extras>/* into <src>/bench_extras/ (untracked, so the manifest ships them; the tree reads -dirty).
    Only for throwaway `wt/bench-*` worktrees or the main clone's siblings named so: never a author's worktree."""
    import shutil
    src_dir = BENCH / extras
    if not src_dir.is_dir():
        sys.exit(f"extras dir {src_dir} does not exist")
    if not src.name.startswith("bench-") and not os.environ.get("VERL_BENCH_EXTRAS_ANYWHERE"):
        sys.exit(f"--ship-extras writes into {src}/bench_extras; use a detached wt/bench-<name> worktree "
                 f"(or set VERL_BENCH_EXTRAS_ANYWHERE=1)")
    dst = src / "bench_extras"
    dst.mkdir(exist_ok=True)
    shipped = {}
    for f in sorted(src_dir.iterdir()):
        if f.is_file() and not f.name.startswith("."):
            shutil.copy2(f, dst / f.name)
            shipped[f"bench_extras/{f.name}"] = hashlib.sha256(f.read_bytes()).hexdigest()
    (RESULTS / purpose).mkdir(parents=True, exist_ok=True)
    (RESULTS / purpose / "extras.json").write_text(json.dumps({"from": str(src_dir), "files": shipped}, indent=2) + "\n")
    print(f"shipped extras into {dst}: {list(shipped)}")


def cmd_build_tool_dataset(a) -> None:
    launch("build_tool_dataset_launch", ["--run-name", a.run_name], a.run_name, Path(a.src), "H100", "verl-contrib-build-dataset")


def cmd_build_dataset(a) -> None:
    flags = ["--run-name", a.run_name, "--tokenizer-model", a.tokenizer_model, "--target-tokens", str(a.target_tokens)]
    launch("build_long_dataset_launch", flags, a.run_name, Path(a.src), "H100", "verl-contrib-build-dataset")


def cmd_train(a) -> None:
    cfg = dict(PRESETS.get(a.preset, {}))
    preset_gpu = cfg.pop("gpu", None)
    extras = a.ship_extras or cfg.pop("extras", None)
    cfg.pop("extras", None)
    if extras:
        ship_extras(Path(a.src).resolve(), extras, a.run_name)
    for k in CFG_KEYS:
        v = getattr(a, k)
        if v is not None:
            cfg[k] = v
    if a.extra_overrides:  # appended after the preset's own overrides (a later key wins)
        cfg["overrides"] = (cfg.get("overrides", "") + " " + a.extra_overrides).strip()
    cfg["preset"] = a.preset or ""
    n = int(cfg["n_gpus"])
    if a.gpu:
        gpu = a.gpu
    elif preset_gpu and a.gpu_type == "H100":
        gpu = preset_gpu
    else:
        gpu = a.gpu_type if n == 1 else f"{a.gpu_type}:{n}"
    flags = ["--run-name", a.run_name]
    for k, v in cfg.items():
        if isinstance(v, bool):
            flags.append(f"--{k.replace('_', '-')}" if v else f"--no-{k.replace('_', '-')}")
        else:
            flags += [f"--{k.replace('_', '-')}", str(v)]
    mem = int(a.memory_gb * 1024) if a.memory_gb else (384 * 1024 if n >= 8 else 128 * 1024 if n >= 2 else 96 * 1024)
    cpu = a.cpu or (64 if n >= 8 else 32 if n >= 2 else 16)
    launch("train_launch", flags, a.run_name, Path(a.src), gpu, "verl-contrib-train",
           extra_env={"VERL_MEMORY_MB": str(mem), "VERL_CPU": str(cpu)})


def cmd_gpu_tests(a) -> None:
    # `=` form: a value such as "-s -rs" would otherwise be parsed as options by Modal's CLI
    flags = ["--run-name", a.run_name, f"--tests={a.tests}", f"--pytest-extra={a.pytest_extra}"]
    if a.no_torchrun:
        flags.append("--no-torchrun")
    launch("gpu_tests_launch", flags, a.run_name, Path(a.src), a.gpu, "verl-contrib-gpu-tests",
           extra_env={"VERL_MEMORY_MB": str(256 * 1024), "VERL_CPU": "32"})


def cmd_logs(a) -> None:
    p = modal_cmd("app", "logs", a.app_id, "--tail", str(a.tail), seconds=a.seconds)
    out = (p.stdout or "") + (p.stderr or "")
    lines = out.splitlines()
    print("\n".join(lines[-a.tail:]))


def cmd_wait(a) -> None:
    code = ("import modal, sys; fc = modal.FunctionCall.from_id(sys.argv[1]);\n"
            "try:\n    print('RESULT', fc.get(timeout=float(sys.argv[2])))\n"
            "except TimeoutError:\n    print('STILL_RUNNING')\n")
    env = {**os.environ}
    p = subprocess.run([str(PY), "-c", code, a.fc_id, str(a.seconds)], env=env, capture_output=True, text=True)
    print((p.stdout or "") + (p.stderr or "")[-2000:])


def cmd_pull(a) -> None:
    dest = RESULTS / a.run_name
    dest.mkdir(parents=True, exist_ok=True)
    p = modal_cmd("volume", "get", "--force", RESULTS_VOLUME, f"{a.run_name}/", str(RESULTS), seconds=600)
    print((p.stdout or "")[-2000:] + (p.stderr or "")[-2000:])
    print(f"pulled to {dest}: {sorted(x.name for x in dest.iterdir())}")


def cmd_stop(a) -> None:
    if not a.app_id.startswith("ap-"):
        sys.exit("stop takes an app id (ap-...), never a name")
    if a.app_id not in LIVE.read_text():
        sys.exit(f"{a.app_id} is not in {LIVE}; refusing to stop an app we did not launch")
    p = modal_cmd("app", "stop", "-y", a.app_id, seconds=120)
    print((p.stdout or "") + (p.stderr or ""))
    record(a.app_id, "-", "-", "stopped")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("probe", cmd_probe), ("download", cmd_download), ("build-dataset", cmd_build_dataset),
                     ("build-tool-dataset", cmd_build_tool_dataset), ("train", cmd_train), ("gpu-tests", cmd_gpu_tests)]:
        s = sub.add_parser(name)
        s.add_argument("--src", default=str(DEFAULT_SRC), help="verl checkout to ship (default: the main clone)")
        s.set_defaults(fn=fn)
        if name == "probe":
            s.add_argument("--run-name", default=f"probe-{datetime.now().strftime('%Y%m%d-%H%M')}")
            s.add_argument("--gpu", default="H100")
        elif name == "download":
            s.add_argument("--models", default="Qwen/Qwen2.5-0.5B,Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-8B")
            s.add_argument("--no-gsm8k", action="store_true")
            s.add_argument("--run-name", default="download")
        elif name == "build-tool-dataset":
            s.add_argument("--run-name", default="build-gsm8k-tool")
        elif name == "build-dataset":
            s.add_argument("--run-name", default="build-gsm8k-long")
            s.add_argument("--tokenizer-model", default="Qwen/Qwen3-1.7B")
            s.add_argument("--target-tokens", type=int, default=6000)
        elif name == "train":
            s.add_argument("--run-name", required=True)
            s.add_argument("--preset", choices=sorted(PRESETS), default=None)
            s.add_argument("--model"); s.add_argument("--n-gpus", type=int); s.add_argument("--rollout-n", type=int)
            s.add_argument("--train-batch-size", type=int); s.add_argument("--max-prompt-length", type=int)
            s.add_argument("--max-response-length", type=int); s.add_argument("--total-training-steps", type=int)
            s.add_argument("--tp", type=int); s.add_argument("--ppo-mini-batch-size", type=int)
            s.add_argument("--ppo-max-token-len-per-gpu", type=int); s.add_argument("--gpu-memory-utilization", type=float)
            s.add_argument("--use-kl-loss", dest="use_kl_loss", action="store_true", default=None)
            s.add_argument("--no-use-kl-loss", dest="use_kl_loss", action="store_false")
            s.add_argument("--overrides", default=None, help="hydra overrides, one string (REPLACES the preset's)")
            s.add_argument("--extra-overrides", default=None, help="hydra overrides appended to the preset's (later key wins)")
            s.add_argument("--data-dir", default=None, help="parquet dir with train/test.parquet (default /hf-cache/data/gsm8k)")
            s.add_argument("--trainer-n-gpus", type=int, default=None, help="trainer.n_gpus_per_node when < --n-gpus (separate rollout)")
            s.add_argument("--chaos", dest="chaos", action="store_true", default=None, help="kill one actor after --chaos-step")
            s.add_argument("--chaos-step", type=int, default=None); s.add_argument("--chaos-class", default=None)
            s.add_argument("--chaos-wall-cap-s", type=int, default=None); s.add_argument("--chaos-hang-s", type=int, default=None)
            s.add_argument("--gpu-type", default="H100", help="H100 | H200 | H100! (strict)")
            s.add_argument("--gpu", default=None, help="full modal gpu spec, overrides --gpu-type, e.g. H200:8")
            s.add_argument("--memory-gb", type=float, default=None); s.add_argument("--cpu", type=float, default=None)
            s.add_argument("--ship-extras", default=None, help="bench/<dir> copied into <src>/bench_extras before launch (preset gsm8k-tool: gsm8k_tool)")
        else:
            s.add_argument("--run-name", default="gpu-tests")
            s.add_argument("--tests", default="tests/")
            s.add_argument("--pytest-extra", default="")
            s.add_argument("--no-torchrun", action="store_true")
            s.add_argument("--gpu", default="H100:8")
    s = sub.add_parser("logs"); s.add_argument("app_id"); s.add_argument("--tail", type=int, default=60)
    s.add_argument("--seconds", type=int, default=90); s.set_defaults(fn=cmd_logs)
    s = sub.add_parser("wait"); s.add_argument("fc_id"); s.add_argument("--seconds", type=int, default=240); s.set_defaults(fn=cmd_wait)
    s = sub.add_parser("pull"); s.add_argument("run_name"); s.set_defaults(fn=cmd_pull)
    s = sub.add_parser("stop"); s.add_argument("app_id"); s.set_defaults(fn=cmd_stop)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
