"""verl GPU benchmark harness on Modal (workspace profile `<profile>`).

Launch through bench/verl_bench.py (it sets the env vars below, records app ids
in LIVE-APPS.md and pulls results). Direct use:

    VERL_SRC=/path/to/verl VERL_GPU=H100:8 MODAL_PROFILE=<profile> \
        .venv-bench/bin/modal run --detach bench/modal_verl.py::train_launch --run-name x ...

Every remote function materialises the verl uv environment from the image's
baked uv cache (`uv sync --frozen --all-packages --extra ...`) and then runs
the shipped source tree at /workspace/verl exactly the way verl's CI does.

The module is re-imported inside the container at /root/modal_verl.py, so
nothing at module level may depend on the Mac filesystem.
"""

import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import modal
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))  # the repository root

# ---- configuration (env vars are set by bench/verl_bench.py at launch) ----
IMAGE_TAG = os.environ.get("VERL_IMAGE", "verlai/verl:uv.cu130.dev5")
ADD_PYTHON = os.environ.get("VERL_ADD_PYTHON") or None  # "3.12" fallback if the image's python is unusable
DEFAULT_SRC = os.path.join(_REPO, "verl")
SRC = os.environ.get("VERL_SRC", DEFAULT_SRC)  # local checkout shipped into the container
SRC_GIT = os.environ.get("VERL_SRC_GIT", "unknown")  # git describe of SRC, computed by the CLI wrapper
SRC_HEAD = os.environ.get("VERL_SRC_HEAD", "unknown")  # full HEAD sha of SRC at launch (worktrees move over time)
SRC_AHEAD = os.environ.get("VERL_SRC_AHEAD", "?")  # `git rev-list --count main..HEAD` of SRC at launch
APP_NAME = os.environ.get("VERL_APP_NAME", "verl-contrib-bench")
GPU = os.environ.get("VERL_GPU", "H100")  # e.g. H100, H100:8, H200:8, H100!:8 (strict)
TIMEOUT_S = int(os.environ.get("VERL_TIMEOUT_S", str(3 * 3600)))
CPU = float(os.environ.get("VERL_CPU", "16"))
MEMORY_MB = int(os.environ.get("VERL_MEMORY_MB", str(96 * 1024)))

REMOTE_SRC = "/workspace/verl"  # where the image keeps its own copy of the repo; ours is mounted over it
HF_VOL_PATH = "/hf-cache"
RES_VOL_PATH = "/results"
HF_HOME = f"{HF_VOL_PATH}/hf_home"
GSM8K_DIR = f"{HF_VOL_PATH}/data/gsm8k"
UV_EXTRAS_TRAIN = ["--extra", "vllm", "--extra", "fsdp"]
UV_EXTRAS_TESTS = ["--extra", "sglang", "--extra", "megatron", "--extra", "ci"]
UV_RUN_BASE = ["uv", "run", "--frozen", "--all-packages"]
MOUNT_IGNORE = [
    ".git", "**/.git", "**/__pycache__", "**/*.pyc", ".venv", ".venv*", "**/.venv", "**/.venv*",
    "docs/_build", "**/.pytest_cache", "**/.ruff_cache", "**/.mypy_cache", "checkpoints", "outputs",
]

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("verl-contrib-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("verl-contrib-results", create_if_missing=True)
VOLUMES = {HF_VOL_PATH: hf_cache, RES_VOL_PATH: results}

# Modal needs a `python` on PATH to detect the interpreter version; the image only ships `python3`
# (Ubuntu 24.04 apt python 3.12). PY_MODE=symlink keeps the image's own interpreter (what verl CI
# uses under uv); PY_MODE=add_python installs Modal's standalone 3.12 to /usr/local instead (shadows
# /usr/bin/python3 on PATH, so uv would build .venv from it).
PY_MODE = os.environ.get("VERL_PY_MODE", "symlink")
if PY_MODE == "add_python":
    image = modal.Image.from_registry(IMAGE_TAG, add_python=ADD_PYTHON or "3.12")
else:
    image = modal.Image.from_registry(
        IMAGE_TAG, setup_dockerfile_commands=["RUN ln -sf /usr/bin/python3 /usr/local/bin/python"])
MANIFEST = os.environ.get("VERL_MANIFEST", "")  # local file listing every shipped path (written by the wrapper)
REMOTE_MANIFEST = "/workspace/verl-manifest.txt"
DIGEST = os.environ.get("VERL_DIGEST", "")  # sha256 over the manifest's files, computed by the wrapper at launch
REMOTE_DIGEST = "/workspace/verl-digest.txt"
BENCH_SCRIPTS = ("build_gsm8k_long.py", "chaos_kill.py")  # bench/ helpers that run inside the container (not verl)
GSM8K_LONG_DIR = f"{HF_VOL_PATH}/data/gsm8k-long"  # built by build_long_dataset (synthetic ~6k-token prompts)
GSM8K_TOOL_DIR = f"{HF_VOL_PATH}/data/gsm8k-tool"  # built by build_tool_dataset (verl's tool-calling GSM8K, P4)
if modal.is_local() and os.path.isdir(SRC):
    # Mounted at container start (copy=False): no image rebuild per source tree. The mount is a
    # file-level overlay on the image's own copy of the repo, so _exact_tree() removes what we did not ship.
    image = image.add_local_dir(SRC, remote_path=REMOTE_SRC, ignore=MOUNT_IGNORE)
    if MANIFEST and os.path.isfile(MANIFEST):
        image = image.add_local_file(MANIFEST, remote_path=REMOTE_MANIFEST)
    if DIGEST and os.path.isfile(DIGEST):
        image = image.add_local_file(DIGEST, remote_path=REMOTE_DIGEST)
    for _name in BENCH_SCRIPTS:
        _local = os.path.join(os.path.dirname(os.path.abspath(__file__)), _name)
        if os.path.isfile(_local):
            image = image.add_local_file(_local, remote_path=f"/root/{_name}")


def tree_digest(root: Path, rels: list[str]) -> str:
    """sha256 over sorted 'path\\0sha256(content)' lines; the wrapper computes it locally, _exact_tree() in the container."""
    h = hashlib.sha256()
    for rel in sorted(rels):
        p = root / rel
        if p.is_file():
            h.update(f"{rel}\0{hashlib.sha256(p.read_bytes()).hexdigest()}\n".encode())
    return h.hexdigest()


def _exact_tree(fp) -> dict:
    """Delete files under REMOTE_SRC that are not in the shipped manifest (the image's stale copy, its .git)."""
    import shutil
    mf = Path(REMOTE_MANIFEST)
    if not mf.exists():
        _log(fp, "no manifest shipped; image-resident stale files (if any) were NOT removed")
        return {"manifest": False}
    keep = set(ln.strip() for ln in mf.read_text().splitlines() if ln.strip())
    root = Path(REMOTE_SRC)
    removed_files = removed_dirs = 0
    for top in (".git",):
        if (root / top).exists():
            shutil.rmtree(root / top, ignore_errors=True)
            removed_dirs += 1
    for path in sorted(root.rglob("*"), key=lambda x: -len(str(x))):
        rel = path.relative_to(root).as_posix()
        if rel.split("/")[0] in (".venv",) or path.is_dir():
            continue
        if rel not in keep:
            path.unlink(missing_ok=True)
            removed_files += 1
    for path in sorted(root.rglob("*"), key=lambda x: -len(str(x))):
        if path.is_dir() and path.name != ".venv" and not any(path.iterdir()):
            path.rmdir()
            removed_dirs += 1
    present = sum(1 for rel in keep if (root / rel).exists())
    out = {"manifest": True, "shipped": len(keep), "present": present, "removed_files": removed_files, "removed_dirs": removed_dirs}
    if Path(REMOTE_DIGEST).exists():  # proves the container runs exactly the tree the wrapper hashed at launch
        out["digest"] = tree_digest(root, sorted(keep))
        out["digest_local"] = Path(REMOTE_DIGEST).read_text().strip()
        out["digest_match"] = out["digest"] == out["digest_local"]
    _log(fp, f"exact tree: {json.dumps(out)}")
    return out


# ---- helpers (run inside the container) ----
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(fp, msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    if fp is not None:
        fp.write(line + "\n")
        fp.flush()


def _sh(cmd, fp=None, cwd=REMOTE_SRC, env=None, check=False, timeout=None) -> subprocess.CompletedProcess:
    """Run a command, tee its merged output to fp and to stdout, return the CompletedProcess."""
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    _log(fp, f"$ {' '.join(shlex.quote(c) for c in cmd)}")
    p = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       timeout=timeout)
    out = p.stdout or ""
    print(out, end="", flush=True)
    if fp is not None:
        fp.write(out)
        fp.flush()
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")
    return p


def _base_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env.update({
        "HF_HOME": HF_HOME,
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VIRTUAL_ENV": f"{REMOTE_SRC}/.venv",
        "UV_PROJECT_ENVIRONMENT": f"{REMOTE_SRC}/.venv",
        "PYTHONUNBUFFERED": "1",
    })
    if extra:
        env.update(extra)
    return env


def _uv_sync(extras: list[str], fp, env: dict) -> float:
    """Materialise /workspace/verl/.venv from the baked uv cache. Returns seconds taken."""
    t0 = time.time()
    _sh(["uv", "sync", "--frozen", "--all-packages", *extras], fp=fp, env=env, check=True)
    dt = time.time() - t0
    _log(fp, f"uv sync done in {dt:.1f}s")
    return dt


def _uv_run(extras: list[str]) -> list[str]:
    return [*UV_RUN_BASE, *extras]


def _resolve_model(model: str) -> str:
    """HF id -> local snapshot dir in HF_HOME (downloaded by `download`); a directory path is returned as is."""
    if os.path.isdir(model):
        return model
    d = Path(HF_HOME) / "hub" / ("models--" + model.replace("/", "--"))
    ref = (d / "refs" / "main").read_text().strip()
    snap = d / "snapshots" / ref
    if not (snap / "config.json").exists():
        raise FileNotFoundError(f"{model}: no snapshot at {snap}; run the download function first")
    return str(snap)


def _run_dir(run_name: str) -> Path:
    d = Path(RES_VOL_PATH) / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _committer(stop: threading.Event, every: float = 60.0) -> None:
    """Commit the results volume periodically so partial logs are visible while a run is in flight."""
    while not stop.wait(every):
        try:
            results.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"[committer] commit failed: {exc}", flush=True)


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


# ---- 1. image probe ----
@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=TIMEOUT_S, retries=0, cpu=CPU, memory=MEMORY_MB)
def probe(run_name: str = "probe", src_git: str = SRC_GIT, gpu: str = GPU) -> dict:
    """Print what the image looks like, time `uv sync`, and check the verl/vllm/torch imports."""
    rd = _run_dir(run_name)
    fp = open(rd / "probe.log", "a")
    env = _base_env()
    _log(fp, f"probe start image={IMAGE_TAG} gpu={gpu} src_git={src_git}")
    tree = _exact_tree(fp)
    for cmd in ["uname -a", "ls /", "which uv python3 python pip", "uv --version", "python3 --version",
                "nvidia-smi", "df -h /dev/shm / /root", "free -g", "nproc", f"ls {REMOTE_SRC}",
                f"cat {REMOTE_SRC}/verl/version/version", f"ls -la {REMOTE_SRC}/.venv",
                f"ls {HF_VOL_PATH}", f"ls {RES_VOL_PATH}", "env"]:
        _sh(cmd, fp=fp, env=env)
    # Overlay semantics: files that exist only in the image's copy of the repo would be stale.
    _sh(f"git -C {REMOTE_SRC} status --short", fp=fp, env=env)  # expect: not a git repo (we do not ship .git)
    _sh(f"find {REMOTE_SRC} -maxdepth 1 -newer {REMOTE_SRC}/uv.lock", fp=fp, env=env)
    sync_s = _uv_sync(UV_EXTRAS_TRAIN, fp, env)
    chk = ("import verl, vllm, torch; print('IMPORT_OK', verl.__version__, vllm.__version__, torch.__version__, "
           "torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)")
    p = _sh([*_uv_run(UV_EXTRAS_TRAIN), "python3", "-c", chk], fp=fp, env=env)
    _sh([*_uv_run(UV_EXTRAS_TRAIN), "python3", "-c", "import verl, sys; print('verl file:', verl.__file__); "
         "print('python:', sys.executable)"], fp=fp, env=env)
    _sh(["uv", "pip", "freeze", "--python", f"{REMOTE_SRC}/.venv/bin/python"], fp=fp, env=env)
    out = {"import_ok": "IMPORT_OK" in (p.stdout or ""), "uv_sync_s": sync_s, "image": IMAGE_TAG, "gpu": gpu,
           "src_git": src_git, "finished": _now()}
    _write_json(rd / "probe.json", out)
    fp.close()
    results.commit()
    return out


@app.local_entrypoint()
def probe_launch(run_name: str = "probe"):
    fc = probe.spawn(run_name, SRC_GIT, GPU)
    print(f"APP_ID={app.app_id} FC_ID={fc.object_id} RUN_NAME={run_name}", flush=True)


# ---- 2. one-time downloads (models + GSM8K parquet) onto the hf-cache volume ----
@app.function(image=image, gpu=None, volumes=VOLUMES, timeout=TIMEOUT_S, retries=0, cpu=8, memory=32 * 1024)
def download(models: str = "Qwen/Qwen2.5-0.5B,Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-8B",
             gsm8k: bool = True, run_name: str = "download") -> dict:
    rd = _run_dir(run_name)
    fp = open(rd / "download.log", "a")
    env = _base_env()
    Path(HF_HOME).mkdir(parents=True, exist_ok=True)
    _uv_sync(UV_EXTRAS_TRAIN, fp, env)
    out = {"models": {}, "gsm8k": None}
    for m in [x.strip() for x in models.split(",") if x.strip()]:
        t0 = time.time()
        code = ("from huggingface_hub import snapshot_download; import sys; "
                f"p = snapshot_download({m!r}, allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.py','*.jinja']); "
                "print('SNAPSHOT', p)")
        p = _sh([*_uv_run(UV_EXTRAS_TRAIN), "python3", "-c", code], fp=fp, env=env)
        snap = [ln.split(" ", 1)[1] for ln in (p.stdout or "").splitlines() if ln.startswith("SNAPSHOT ")]
        out["models"][m] = {"path": snap[-1] if snap else None, "ok": p.returncode == 0, "s": round(time.time() - t0, 1)}
        hf_cache.commit()
    if gsm8k:
        Path(GSM8K_DIR).mkdir(parents=True, exist_ok=True)
        p = _sh([*_uv_run(UV_EXTRAS_TRAIN), "python3", "examples/data_preprocess/gsm8k.py", "--local_save_dir", GSM8K_DIR],
                fp=fp, env=env)
        _sh(f"ls -la {GSM8K_DIR}", fp=fp, env=env)
        out["gsm8k"] = {"ok": p.returncode == 0, "dir": GSM8K_DIR}
        hf_cache.commit()
    _sh(f"du -sh {HF_HOME}/hub/* {GSM8K_DIR}", fp=fp, env=env)
    out["finished"] = _now()
    _write_json(rd / "download.json", out)
    fp.close()
    hf_cache.commit()
    results.commit()
    return out


@app.local_entrypoint()
def download_launch(models: str = "Qwen/Qwen2.5-0.5B,Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-8B",
                    gsm8k: bool = True, run_name: str = "download"):
    fc = download.spawn(models, gsm8k, run_name)
    print(f"APP_ID={app.app_id} FC_ID={fc.object_id} RUN_NAME={run_name}", flush=True)


# ---- 2b. synthetic long-prompt GSM8K (bench/build_gsm8k_long.py) onto the hf-cache volume ----
@app.function(image=image, gpu=None, volumes=VOLUMES, timeout=TIMEOUT_S, retries=0, cpu=16, memory=64 * 1024)
def build_long_dataset(run_name: str = "build-gsm8k-long", tokenizer_model: str = "Qwen/Qwen3-1.7B",
                       target_tokens: int = 6000, dst: str = GSM8K_LONG_DIR) -> dict:
    rd = _run_dir(run_name)
    fp = open(rd / "build.log", "a")
    env = _base_env()
    _log(fp, f"build_long_dataset start tokenizer={tokenizer_model} target={target_tokens} dst={dst}")
    _uv_sync(UV_EXTRAS_TRAIN, fp, env)
    tok = _resolve_model(tokenizer_model)
    t0 = time.time()
    p = _sh([*_uv_run(UV_EXTRAS_TRAIN), "python3", "/root/build_gsm8k_long.py", "--src", GSM8K_DIR, "--dst", dst,
             "--tokenizer", tok, "--target-tokens", str(target_tokens)], fp=fp, env=env)
    hf_cache.commit()
    out = {"ok": p.returncode == 0 and "BUILD_OK" in (p.stdout or ""), "returncode": p.returncode, "dst": dst,
           "tokenizer": tok, "build_s": round(time.time() - t0, 1), "finished": _now()}
    bj = Path(dst) / "build.json"
    if bj.exists():
        out["build"] = json.loads(bj.read_text())
        (rd / "build.json").write_text(bj.read_text())
    _sh(f"ls -la {dst}", fp=fp, env=env)
    _write_json(rd / "summary.json", out)
    fp.close()
    results.commit()
    return out


@app.local_entrypoint()
def build_long_dataset_launch(run_name: str = "build-gsm8k-long", tokenizer_model: str = "Qwen/Qwen3-1.7B",
                              target_tokens: int = 6000, dst: str = GSM8K_LONG_DIR):
    fc = build_long_dataset.spawn(run_name, tokenizer_model, target_tokens, dst)
    print(f"APP_ID={app.app_id} FC_ID={fc.object_id} RUN_NAME={run_name}", flush=True)


# ---- 2c. verl's tool-calling GSM8K (examples/data_preprocess/gsm8k_tool_agent_loop.py) onto the hf-cache volume ----
TOOL_DATASET_CHECK = """
import hashlib, json, sys
import pandas as pd
out = {}
for split in ("train", "test"):
    path = f"{sys.argv[1]}/{split}.parquet"
    df = pd.read_parquet(path)
    row = df.iloc[0]
    kw = row["extra_info"]["tools_kwargs"]
    out[split] = {"rows": int(len(df)), "columns": list(df.columns), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                  "agent_name": str(row["agent_name"]), "data_source": str(row["data_source"]),
                  "prompt_roles": [m["role"] for m in row["prompt"]], "tools_kwargs_keys": list(kw.keys()),
                  "ground_truth_matches_create_kwargs": bool(row["reward_model"]["ground_truth"] == kw["calc_gsm8k_reward"]["create_kwargs"]["ground_truth"]),
                  "system_prompt_head": str(row["prompt"][0]["content"])[:120]}
print("BUILD_CHECK " + json.dumps(out))
"""


@app.function(image=image, gpu=None, volumes=VOLUMES, timeout=TIMEOUT_S, retries=0, cpu=8, memory=32 * 1024)
def build_tool_dataset(run_name: str = "build-gsm8k-tool", dst: str = GSM8K_TOOL_DIR, src_git: str = SRC_GIT,
                       src_head: str = SRC_HEAD) -> dict:
    """Run verl's own preprocess script (system prompt asking for `calc_gsm8k_reward` calls, agent_name=tool_agent,
    tools_kwargs carrying the ground truth) into the volume; verify the schema and record sha256s in build.json."""
    rd = _run_dir(run_name)
    fp = open(rd / "build.log", "a")
    env = _base_env()
    _log(fp, f"build_tool_dataset start dst={dst}")
    _uv_sync(UV_EXTRAS_TRAIN, fp, env)
    Path(dst).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = _sh([*_uv_run(UV_EXTRAS_TRAIN), "python3", "examples/data_preprocess/gsm8k_tool_agent_loop.py",
             "--local_save_dir", dst], fp=fp, env=env)
    q = _sh([*_uv_run(UV_EXTRAS_TRAIN), "python3", "-c", TOOL_DATASET_CHECK, dst], fp=fp, env=env)
    check = [ln[len("BUILD_CHECK "):] for ln in (q.stdout or "").splitlines() if ln.startswith("BUILD_CHECK ")]
    out = {"ok": p.returncode == 0 and q.returncode == 0 and bool(check), "returncode": p.returncode, "dst": dst,
           "script": "examples/data_preprocess/gsm8k_tool_agent_loop.py", "src_git": src_git, "src_head": src_head,
           "build_s": round(time.time() - t0, 1), "finished": _now(), "check": json.loads(check[-1]) if check else None}
    _write_json(Path(dst) / "build.json", out)
    hf_cache.commit()
    _sh(f"ls -la {dst}", fp=fp, env=env)
    _write_json(rd / "build.json", out)
    _write_json(rd / "summary.json", out)
    fp.close()
    results.commit()
    return out


@app.local_entrypoint()
def build_tool_dataset_launch(run_name: str = "build-gsm8k-tool", dst: str = GSM8K_TOOL_DIR):
    fc = build_tool_dataset.spawn(run_name, dst, SRC_GIT, SRC_HEAD)
    print(f"APP_ID={app.app_id} FC_ID={fc.object_id} RUN_NAME={run_name}", flush=True)


# ---- 3. GRPO training run ----
def grpo_overrides(cfg: dict, model_path: str) -> dict:
    """Ordered hydra overrides for a run config (mirrors examples/grpo_trainer/run_qwen3_8b_fsdp.sh).

    cfg['overrides'] ("a.b=c d.e=f", also "+k=v" / "~k") replace a key in place or extend the list, so the
    recorded command carries every key exactly once and the preset's final values are explicit.
    """
    bs = int(cfg["train_batch_size"])
    mini = int(cfg.get("ppo_mini_batch_size") or 0) or max(1, bs // 4)
    max_tok = int(cfg.get("ppo_max_token_len_per_gpu") or 0) or 24576
    kl = "True" if cfg.get("use_kl_loss", True) else "False"
    data_dir = cfg.get("data_dir") or GSM8K_DIR
    trainer_gpus = int(cfg.get("trainer_n_gpus") or 0) or int(cfg["n_gpus"])
    items = [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files={data_dir}/train.parquet",
        f"data.val_files={data_dir}/test.parquet",
        f"data.train_batch_size={bs}",
        f"data.max_prompt_length={int(cfg['max_prompt_length'])}",
        f"data.max_response_length={int(cfg['max_response_length'])}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={mini}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={max_tok}",
        f"actor_rollout_ref.actor.use_kl_loss={kl}",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={int(cfg.get('tp') or 1)}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={float(cfg.get('gpu_memory_utilization') or 0.6)}",
        f"actor_rollout_ref.rollout.n={int(cfg['rollout_n'])}",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={max_tok}",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={max_tok}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "trainer.critic_warmup=0",
        "trainer.balance_batch=True",
        "trainer.logger=[console,file]",
        "trainer.project_name=verl-contrib-bench",
        f"trainer.experiment_name={cfg['run_name']}",
        "trainer.nnodes=1",
        f"trainer.n_gpus_per_node={trainer_gpus}",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        "trainer.save_freq=-1",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={int(cfg['total_training_steps'])}",
        f"trainer.default_local_dir=/tmp/ckpt/{cfg['run_name']}",
        "ray_kwargs.ray_init.runtime_env.py_executable=uv -v run --frozen --all-packages --extra vllm --extra fsdp",
    ]
    ov: dict = {}
    run_dir = f"{RES_VOL_PATH}/{cfg['run_name']}"  # "{run_dir}" in an override value = this run's results directory
    for item in items + shlex.split((cfg.get("overrides") or "").strip()):
        k, eq, v = item.partition("=")
        ov[k] = v.replace("{run_dir}", run_dir) if eq else None  # "~key" (hydra delete) has no value
    return ov


def grpo_command(cfg: dict, model_path: str) -> list[str]:
    """The exact verl.trainer.main_ppo command line for a run config."""
    return [*_uv_run(UV_EXTRAS_TRAIN), "python3", "-m", "verl.trainer.main_ppo",
            *(k if v is None else f"{k}={v}" for k, v in grpo_overrides(cfg, model_path).items())]


def parse_metrics_jsonl(path: Path) -> list[dict]:
    """verl FileLogger lines -> [{step, timing_s/*, perf/*, ...}] (numbers only)."""
    steps = []
    if not path.exists():
        return steps
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        data = rec.get("data", {})
        row = {"step": rec.get("step")}
        for k, v in data.items():
            if isinstance(v, (int, float)) and k.startswith((
                    "timing_s/", "perf/", "training/", "critic/", "actor/", "num_turns/", "val-core/", "val-aux/",
                    "response_length/", "prompt_length/", "rollout/", "reward/", "off_policy/", "adv_estimator/")):
                row[k] = v
        steps.append(row)
    return steps


def _chaos_arm(class_name: str, trigger: Path, ready: Path, env: dict) -> subprocess.Popen:
    """Start bench/chaos_kill.py in the trainer's venv: it connects to the running Ray cluster now and
    ray.kill()s one <class_name> actor as soon as `trigger` exists (see chaos_kill.py)."""
    return subprocess.Popen([*_uv_run(UV_EXTRAS_TRAIN), "python3", "/root/chaos_kill.py", "--class-name", class_name,
                             "--wait-for-file", str(trigger), "--ready-file", str(ready)],
                            cwd=REMOTE_SRC, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _chaos_collect(proc: subprocess.Popen, class_name: str, fp) -> dict:
    """Wait for the armed killer to report; parse its CHAOS_KILL line."""
    info = {"class_name": class_name}
    try:
        out, _ = proc.communicate(timeout=240)
        info["rc"] = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        info["rc"] = "timeout"
    for ln in (out or "").splitlines():
        _log(fp, f"[chaos-killer] {ln}")
        if ln.startswith("CHAOS_KILL "):
            info.update(json.loads(ln[len("CHAOS_KILL "):]))
    info["t_unix"] = info.get("kill_unix") or time.time()
    info["t_utc"] = datetime.fromtimestamp(info["t_unix"], timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return info


def _run_trainer(argv: list[str], env: dict, rd: Path, cfg: dict, fp) -> tuple[int, float, dict | None]:
    """Run the trainer (tee to train.log + container stdout). With cfg['chaos']: after step cfg['chaos_step']
    completes, kill one cfg['chaos_class'] actor; cap the run (wall cap, and a hang cap after the kill) and
    record outcome: completed | crashed | hung | exited_early, steps reached and kill-to-next-step time."""
    import signal
    t0 = time.time()
    holder: dict = {}
    proc = subprocess.Popen(argv, cwd=REMOTE_SRC, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)

    def pump():
        with open(rd / "train.log", "ab") as lf:
            for raw in iter(proc.stdout.readline, b""):
                lf.write(raw)
                lf.flush()
                sys.stdout.buffer.write(raw)
                sys.stdout.flush()
        holder["rc"] = proc.wait()

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    chaos = None
    if cfg.get("chaos"):
        chaos = {"step": int(cfg.get("chaos_step") or 2), "class_name": cfg.get("chaos_class") or "AgentLoopWorkerTQ",
                 "wall_cap_s": int(cfg.get("chaos_wall_cap_s") or 1200), "hang_s": int(cfg.get("chaos_hang_s") or 600),
                 "kill": None, "terminated_by": None, "kill_to_next_step_s": None, "first_step_after_kill": None}
    total = int(cfg["total_training_steps"])
    last_seen = 0
    killer = None
    trigger, ready = Path(f"/tmp/chaos-{cfg['run_name']}.trigger"), Path(f"/tmp/chaos-{cfg['run_name']}.ready")
    while th.is_alive():
        th.join(0.5 if chaos else 2.0)
        if chaos is None:
            continue
        now = time.time()
        done = [s["step"] for s in parse_metrics_jsonl(rd / "metrics.jsonl") if isinstance(s.get("step"), int)]
        reached = max(done) if done else 0
        if reached != last_seen:
            last_seen = reached
            _log(fp, f"[chaos] step {reached} completed at +{now - t0:.1f}s")
            if chaos["kill"] and reached > chaos["kill"]["step_before"] and chaos["kill_to_next_step_s"] is None:
                chaos["kill_to_next_step_s"] = round(now - chaos["kill"]["t_unix"], 1)
                chaos["first_step_after_kill"] = reached
        if chaos["kill"] is None and killer is None and reached >= chaos["step"] - 1:
            killer = _chaos_arm(chaos["class_name"], trigger, ready, env)  # connects to Ray now, fires on the trigger
            chaos["armed_at_step"], chaos["armed_rel_s"] = reached, round(now - t0, 1)
            _log(fp, f"[chaos] killer armed at step {reached} (+{now - t0:.1f}s), fires after step {chaos['step']}")
        if chaos["kill"] is None and killer is not None and reached >= chaos["step"]:
            trigger.touch()
            _log(fp, f"[chaos] step {reached} >= {chaos['step']}: trigger (killer connected={ready.exists()})")
            chaos["kill"] = _chaos_collect(killer, chaos["class_name"], fp)
            chaos["kill"]["step_before"] = reached
            chaos["kill"]["t_rel_s"] = round(chaos["kill"]["t_unix"] - t0, 1)
            chaos["kill"]["trigger_rel_s"] = round(now - t0, 1)
            _log(fp, f"[chaos] kill result: {json.dumps(chaos['kill'])}")
        why = None
        if now - t0 > chaos["wall_cap_s"]:
            why = f"wall_cap_s={chaos['wall_cap_s']}"
        elif chaos["kill"] and chaos["kill_to_next_step_s"] is None and now - chaos["kill"]["t_unix"] > chaos["hang_s"]:
            why = f"hang_s={chaos['hang_s']}: no step completed after the kill"
        if why:
            _log(fp, f"[chaos] terminating the trainer process group: {why}")
            chaos["terminated_by"] = why
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            th.join(60)
            if th.is_alive():
                _sh("ray stop --force", fp=fp, env=env)
                th.join(60)
            break
    rc = holder.get("rc", proc.poll() if proc.poll() is not None else -9)
    wall = time.time() - t0
    if chaos is not None:
        done = [s["step"] for s in parse_metrics_jsonl(rd / "metrics.jsonl") if isinstance(s.get("step"), int)]
        chaos["steps_reached"] = max(done) if done else 0
        chaos["total_training_steps"] = total
        chaos["wall_s"] = round(wall, 1)
        chaos["returncode"] = rc
        if chaos["terminated_by"]:
            chaos["outcome"] = "hung"
        elif rc != 0:
            chaos["outcome"] = "crashed"
        elif chaos["steps_reached"] >= total:
            chaos["outcome"] = "completed"
        else:
            chaos["outcome"] = "exited_early"
        _log(fp, f"[chaos] outcome: {json.dumps(chaos)}")
    return rc, wall, chaos


def _stream(argv: list[str], env: dict, cwd: str, log_path: Path) -> int:
    """Run argv, tee merged stdout/stderr to log_path and to the container stdout (Modal logs)."""
    with open(log_path, "ab") as lf:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for raw in iter(proc.stdout.readline, b""):
            lf.write(raw)
            lf.flush()
            sys.stdout.buffer.write(raw)
            sys.stdout.flush()
        return proc.wait()


def summarize_rollout_dumps(dump_dir: Path) -> dict:
    """Per-step means of the numeric per-sample columns in trainer.rollout_data_dir / validation_data_dir dumps
    (`<step>.jsonl`, one row per trajectory: score + every reward_extra_info key, e.g. the gsm8k-tool diagnostics)."""
    out: dict = {}
    if not dump_dir.is_dir():
        return out
    for path in sorted(dump_dir.glob("*.jsonl"), key=lambda x: int(x.stem) if x.stem.isdigit() else -1):
        sums: dict = {}
        n = 0
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            for k, v in rec.items():
                if isinstance(v, bool):
                    v = float(v)
                if isinstance(v, (int, float)) and k != "step":
                    sums[k] = sums.get(k, 0.0) + float(v)
        if n:
            out[path.stem] = {"n": n, **{f"{k}/mean": round(v / n, 6) for k, v in sorted(sums.items())}}
    return out


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=TIMEOUT_S, retries=0, cpu=CPU, memory=MEMORY_MB)
def train(cfg: dict) -> dict:
    import socket
    run_name = cfg["run_name"]
    rd = _run_dir(run_name)
    fp = open(rd / "harness.log", "a")
    stop = threading.Event()
    threading.Thread(target=_committer, args=(stop,), daemon=True).start()
    env = _base_env({"VERL_FILE_LOGGER_PATH": str(rd / "metrics.jsonl"), "HF_HUB_OFFLINE": "1"})
    meta = {"cfg": cfg, "image": IMAGE_TAG, "gpu": cfg.get("gpu", GPU), "src_git": cfg.get("src_git", SRC_GIT),
            "started": _now(), "hostname": socket.gethostname()}
    _log(fp, f"train start {json.dumps(meta)}")
    tree = _exact_tree(fp)
    p = _sh("nvidia-smi", fp=None, env=env)
    (rd / "nvidia-smi-before.txt").write_text(p.stdout or "")
    _sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv", fp=fp, env=env)
    meta["uv_sync_s"] = _uv_sync(UV_EXTRAS_TRAIN, fp, env)
    p = _sh(["uv", "pip", "freeze", "--python", f"{REMOTE_SRC}/.venv/bin/python"], fp=None, env=env)
    (rd / "pip-freeze.txt").write_text(p.stdout or "")
    model_path = _resolve_model(cfg["model"])
    _log(fp, f"model {cfg['model']} -> {model_path}")
    argv = grpo_command(cfg, model_path)
    cmdline = " ".join(shlex.quote(a) for a in argv)
    (rd / "command.txt").write_text(f"cd {REMOTE_SRC} && VERL_FILE_LOGGER_PATH={rd}/metrics.jsonl HF_HOME={HF_HOME} "
                                    f"HF_HUB_OFFLINE=1 {cmdline}\n")
    meta.update({"command": cmdline, "model_path": model_path, "tree": tree})
    _write_json(rd / "meta.json", meta)
    results.commit()
    _log(fp, f"launching: {cmdline}")
    rc, wall, chaos = _run_trainer(argv, env, rd, cfg, fp)
    _log(fp, f"trainer exited rc={rc} after {wall:.1f}s")
    p = _sh("nvidia-smi", fp=None, env=env)
    (rd / "nvidia-smi-after.txt").write_text(p.stdout or "")
    steps = parse_metrics_jsonl(rd / "metrics.jsonl")
    summary = {**meta, "returncode": rc, "wall_s": round(wall, 1), "finished": _now(), "chaos": chaos, "steps": steps,
               "rollout_diagnostics": summarize_rollout_dumps(rd / "rollouts"),
               "validation_diagnostics": summarize_rollout_dumps(rd / "val")}
    _write_json(rd / "summary.json", summary)
    fp.close()
    stop.set()
    results.commit()
    out = {"returncode": rc, "wall_s": round(wall, 1), "n_steps": len(steps), "run_name": run_name}
    if chaos:
        out["chaos"] = {k: chaos.get(k) for k in ("outcome", "steps_reached", "kill_to_next_step_s", "terminated_by")}
    return out


@app.local_entrypoint()
def train_launch(run_name: str, model: str = "Qwen/Qwen2.5-0.5B", n_gpus: int = 1, rollout_n: int = 4,
                 train_batch_size: int = 16, max_prompt_length: int = 512, max_response_length: int = 512,
                 total_training_steps: int = 3, tp: int = 1, use_kl_loss: bool = True, ppo_mini_batch_size: int = 0,
                 ppo_max_token_len_per_gpu: int = 24576, gpu_memory_utilization: float = 0.6, overrides: str = "",
                 data_dir: str = "", trainer_n_gpus: int = 0, preset: str = "", chaos: bool = False, chaos_step: int = 2,
                 chaos_class: str = "AgentLoopWorkerTQ", chaos_wall_cap_s: int = 1200, chaos_hang_s: int = 600):
    cfg = dict(run_name=run_name, model=model, n_gpus=n_gpus, rollout_n=rollout_n, train_batch_size=train_batch_size,
               max_prompt_length=max_prompt_length, max_response_length=max_response_length,
               total_training_steps=total_training_steps, tp=tp, use_kl_loss=use_kl_loss,
               ppo_mini_batch_size=ppo_mini_batch_size, ppo_max_token_len_per_gpu=ppo_max_token_len_per_gpu,
               gpu_memory_utilization=gpu_memory_utilization, overrides=overrides, data_dir=data_dir,
               trainer_n_gpus=trainer_n_gpus, preset=preset, chaos=chaos, chaos_step=chaos_step, chaos_class=chaos_class,
               chaos_wall_cap_s=chaos_wall_cap_s, chaos_hang_s=chaos_hang_s, gpu=GPU, src=SRC, src_git=SRC_GIT,
               src_head=SRC_HEAD, src_commits_ahead_of_main=SRC_AHEAD)
    fc = train.spawn(cfg)
    print(f"APP_ID={app.app_id} FC_ID={fc.object_id} RUN_NAME={run_name}", flush=True)


# ---- 4. GPU unit tests, the way .github/workflows/gpu_unit_tests.yml runs them ----
PYTEST_IGNORES = [
    "--ignore-glob=*on_npu.py", "--ignore-glob=*test_special_*.py", "--ignore-glob=*on_cpu.py",
    "--ignore-glob=*test_vllm*", "--ignore-glob=*_sglang*", "--ignore-glob=*_hf_rollout*", "--ignore-glob=tests/models/",
    "--ignore-glob=tests/special*", "--ignore-glob=tests/experimental", "--ignore-glob=tests/workers/reward_model",
    "--ignore-glob=*test_shared_memory*", "--ignore-glob=tests/workers/rollout/rollout_trtllm",
    "--ignore-glob=*test_bucketed_weight_transfer*",
]


def _link_models(fp) -> list[str]:
    """verl's CI runners keep models at ~/models/<org>/<name>; link every cached HF snapshot there."""
    hub = Path(HF_HOME) / "hub"
    linked = []
    for d in sorted(hub.glob("models--*--*")):
        org, name = d.name[len("models--"):].split("--", 1)
        ref = d / "refs" / "main"
        if not ref.exists():
            continue
        snap = d / "snapshots" / ref.read_text().strip()
        dst = Path("/root/models") / org / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(snap)
        linked.append(f"{org}/{name}")
    _log(fp, f"linked /root/models: {linked}")
    return linked


def parse_junit(path: Path) -> dict:
    import xml.etree.ElementTree as ET
    out = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "failed": []}
    if not path.exists():
        return out
    root = ET.parse(path).getroot()
    for suite in root.iter("testsuite"):
        for k in ("tests", "failures", "errors", "skipped"):
            out[k] += int(suite.get(k, 0))
    for case in root.iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            out["failed"].append(f"{case.get('classname')}::{case.get('name')}")
    out["passed"] = out["tests"] - out["failures"] - out["errors"] - out["skipped"]
    return out


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=TIMEOUT_S, retries=0, cpu=CPU, memory=MEMORY_MB)
def gpu_tests(run_name: str = "gpu-tests", tests: str = "tests/", pytest_extra: str = "", torchrun: bool = True,
              src_git: str = SRC_GIT, gpu: str = GPU) -> dict:
    rd = _run_dir(run_name)
    fp = open(rd / "harness.log", "a")
    stop = threading.Event()
    threading.Thread(target=_committer, args=(stop,), daemon=True).start()
    env = _base_env({"HF_HUB_ENABLE_HF_TRANSFER": "0"})
    meta = {"image": IMAGE_TAG, "gpu": gpu, "src_git": src_git, "src_head": SRC_HEAD, "src_commits_ahead_of_main": SRC_AHEAD,
            "started": _now(), "tests": tests, "pytest_extra": pytest_extra}
    _log(fp, f"gpu_tests start {json.dumps(meta)}")
    tree = _exact_tree(fp)
    p = _sh("nvidia-smi", fp=None, env=env)
    (rd / "nvidia-smi-before.txt").write_text(p.stdout or "")
    meta["tree"] = tree
    meta["uv_sync_s"] = _uv_sync(UV_EXTRAS_TESTS, fp, env)
    _sh([*_uv_run(UV_EXTRAS_TESTS), "uv", "pip", "install", "mlflow"], fp=fp, env=env)  # workflow step: cannot be locked
    meta["models_linked"] = _link_models(fp)
    p = _sh(["uv", "pip", "freeze", "--python", f"{REMOTE_SRC}/.venv/bin/python"], fp=None, env=env)
    (rd / "pip-freeze.txt").write_text(p.stdout or "")
    argv = [*_uv_run(UV_EXTRAS_TESTS), "pytest", "-s", *PYTEST_IGNORES, f"--junitxml={rd}/pytest.xml",
            *shlex.split(pytest_extra), *shlex.split(tests)]
    (rd / "command.txt").write_text(" ".join(shlex.quote(a) for a in argv) + "\n")
    results.commit()
    t0 = time.time()
    rc = _stream(argv, env, REMOTE_SRC, rd / "pytest.log")
    meta["pytest"] = {"returncode": rc, "wall_s": round(time.time() - t0, 1), **parse_junit(rd / "pytest.xml")}
    _log(fp, f"pytest rc={rc} {json.dumps(meta['pytest'])}")
    _write_json(rd / "summary.json", meta)
    results.commit()
    if torchrun:
        for name, cmd, extra_env in [
            ("linear_cross_entropy_tp", ["torchrun", "--standalone", "--nnodes=1", "--nproc-per-node=8",
                                         "tests/utils/test_special_linear_cross_entropy_tp.py"], {"LOW_MEMORY": "True"}),
            ("megatron_kl_loss_tp", ["torchrun", "--standalone", "--nnodes=1", "--nproc-per-node=2",
                                     "tests/utils/test_special_megatron_kl_loss_tp.py"], {}),
        ]:
            t0 = time.time()
            rc = _stream([*_uv_run(UV_EXTRAS_TESTS), *cmd], {**env, **extra_env}, REMOTE_SRC, rd / f"torchrun-{name}.log")
            meta[f"torchrun_{name}"] = {"returncode": rc, "wall_s": round(time.time() - t0, 1)}
            _log(fp, f"torchrun {name} rc={rc}")
    p = _sh("nvidia-smi", fp=None, env=env)
    (rd / "nvidia-smi-after.txt").write_text(p.stdout or "")
    meta["finished"] = _now()
    _write_json(rd / "summary.json", meta)
    fp.close()
    stop.set()
    results.commit()
    return meta


@app.local_entrypoint()
def gpu_tests_launch(run_name: str = "gpu-tests", tests: str = "tests/", pytest_extra: str = "", torchrun: bool = True):
    fc = gpu_tests.spawn(run_name, tests, pytest_extra, torchrun, SRC_GIT, GPU)
    print(f"APP_ID={app.app_id} FC_ID={fc.object_id} RUN_NAME={run_name}", flush=True)
