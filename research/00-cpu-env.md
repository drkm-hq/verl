# verl CPU test harness on this Mac (Apple M5 Max, macOS, no CUDA)

Status: DONE 2026-10-28. Harness works: the safe regression set (section 7) is
1484 passed / 64 skipped / 0 failed across six capped groups; clone untouched.

Purpose: let other jobs run verl's CPU unit tests (`tests/**/*_on_cpu.py`) against
code changes in the clone, the way `.github/workflows/cpu_unit_tests.yml` does, without
a GPU and without touching the clone's git state.

## 0. Layout

| Path | What |
|---|---|
| `<repo>/verl` | the clone: verl-project/verl, `main`, 0.10.0.dev, commit `2ab35292`. Never edited by the harness; `git status --porcelain` must stay empty. |
| `<repo>/.venv-cpu` | Python 3.12.14 venv (created with uv) holding torch CPU + verl's `verl-core` deps + test extras. |
| `<repo>/requirements-cpu-macos.txt` | the exact pins that were installed (derived from `uv.lock`, see section 1). |
| `<repo>/pytest-cpu.ini` | the CI `pytest.ini` (`python_files = *_on_cpu.py`), kept outside the clone, plus `cache_dir` outside the clone. |
| `<repo>/scripts/timeout.py` | `timeout(1)` replacement (no GNU coreutils on this Mac); kills the whole process group, exits 124 on expiry. |
| `<repo>/data/{gsm8k,geo3k}` | preprocessed parquet datasets (section 3). |
| `<repo>/research/logs/*.log`, `research/junit/*.xml` | per-directory pytest output and junit reports from the baseline run (section 4). |

Tools: `uv` 0.12.9 at `/opt/homebrew/bin/uv`. No `timeout`/`gtimeout` binary exists
(`brew list coreutils` says not installed); use `scripts/timeout.py`.

## 1. Environment install (replayable)

Proven on 2026-10-28. Every command is absolute-path so it can be replayed from anywhere.
The pins come from verl's own `uv.lock` (the thing CI actually tests), exported for the
`cpu` extra, so this venv is the CI venv minus four packages that cannot exist on macOS
arm64: `megatron-core` (git source, imports `triton`), `triton` (no macOS wheels),
`nvtx` (CUDA), `torchcodec` (needs FFmpeg shared libs; only used for video samples).
`uv sync --extra cpu` itself refuses on macOS because `[tool.uv].environments` is Linux-only.

```bash
# 0. venv (already existed; this is how to recreate it)
/opt/homebrew/bin/uv venv --python 3.12 <repo>/.venv-cpu

# 1. export the lock's cpu-extra pins (read-only on the clone; --frozen never rewrites uv.lock)
cd <repo>/verl
/opt/homebrew/bin/uv export --frozen --extra cpu --no-hashes --no-emit-project --no-header --no-annotate \
  -o <repo>/research/uv-lock-cpu-export.txt
# -> <repo>/requirements-cpu-macos.txt is that file with the
#    " ; <linux markers>" stripped, "+cpu" local tags removed, fastapi deduped to 0.137.0
#    (the lock forks it per arch), and megatron-core/triton/nvtx/torchcodec dropped.

# 2. torch trio from the PyTorch CPU index (2.13.0 CPU exists for macOS arm64)
/opt/homebrew/bin/uv pip install --python <repo>/.venv-cpu/bin/python \
  --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0" "torchvision==0.28.0" "torchaudio==2.11.0"

# 3. everything else, exact lock pins, from PyPI (all 140 resolve to wheels on macOS arm64; ~3 min)
/opt/homebrew/bin/uv pip install --python <repo>/.venv-cpu/bin/python \
  -r <repo>/requirements-cpu-macos.txt

# 4. verl itself, editable, no deps (its metadata would otherwise ask for the GPU-only extras)
/opt/homebrew/bin/uv pip install --python <repo>/.venv-cpu/bin/python \
  --no-deps -e <repo>/verl
#    This writes <repo>/verl/verl.egg-info/ inside the clone.
#    It is covered by the clone's .gitignore (*.egg-info/), so `git status --porcelain` stays empty.

# 5. proof
env TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 <repo>/.venv-cpu/bin/python \
  -c "import verl, verl.protocol, verl.trainer.ppo.core_algos, verl.utils.seqlen_balancing; print('ok', verl.__version__)"
# -> ok 0.10.0.dev
```

### Installed versions (from `importlib.metadata`)

| package | version | note |
|---|---|---|
| python | 3.12.14 | |
| torch / torchvision / torchaudio | 2.13.0 / 0.28.0 / 2.11.0 | CPU builds from download.pytorch.org/whl/cpu; same pins as the `cpu` extra |
| transformers / tokenizers | 5.16.1 / 0.23.2 | project-wide override pin |
| ray | 2.55.1 | `ray[default]` |
| tensordict | 0.10.0 | |
| datasets / pyarrow / pandas / numpy | 5.0.0 / 24.0.0 / 3.0.3 / 2.4.6 | |
| peft / accelerate | 0.19.1 / 1.14.0 | |
| hydra-core / omegaconf | 1.3.3 / 2.3.1 | |
| transferqueue | 0.1.11 | installed with its real metadata (CI installs it deps-free); resolved without conflicts |
| pytest / pytest-asyncio / pytest-rerunfailures | 9.1.0 / 1.4.0 / 16.3 | |
| pre-commit / py-spy | 4.6.0 / 0.4.2 | |
| qwen-vl-utils / msgspec / pyzmq / wandb | 0.0.14 / 0.21.1 / 27.1.0 / 0.27.2 | |
| verl | 0.10.0.dev0 (editable, from the clone) | |

Not installed (and why): megatron-core, triton (no macOS build; every `*_on_cpu.py`
that imports `megatron` fails at collection here even though CI runs it), nvtx, torchcodec,
flash-attn, cupy, apex, transformer-engine, vllm, sglang, deep-ep (GPU-only; the tests that need
them `pytest.importorskip` and are skipped on CI too).

## 2. Lint: pre-commit / ruff rules (from `verl/.pre-commit-config.yaml` and `verl/pyproject.toml`)

Hook environments were built with (no git hooks installed, nothing written in the clone):

```bash
cd <repo>/verl
<repo>/.venv-cpu/bin/pre-commit install-hooks   # envs land in ~/.cache/pre-commit
```

The first attempt stalled for >4 min cloning github.com/astral-sh/ruff-pre-commit (network);
the retry completed in under 4 min. If it stalls again, just rerun it (it resumes).

Run the formatter/linter on the files you touched (this is what matters for a PR):

```bash
cd <repo>/verl
export PATH=<repo>/.venv-cpu/bin:$PATH   # local hooks call bare `python3`
export RUFF_CACHE_DIR=<repo>/.ruff_cache  # .ruff_cache is NOT gitignored in verl
export MYPY_CACHE_DIR=<repo>/.mypy_cache  # same for .mypy_cache
pre-commit run ruff        --files <file>...   # ruff check --fix --show-fixes --output-format=full
pre-commit run ruff-format --files <file>...   # ruff format
pre-commit run mypy        --files <file>...   # mypy v1.17.0; ignore_errors=true except the 5 modules below
```

Do NOT run `pre-commit run --all-files` or a bare `pre-commit run --files` here: the config
also has `pass_filenames: false` local hooks (`autogen-trainer-cfg` regenerates
`verl/trainer/config/_generated_*.yaml` in place, plus license/docstring/structure
checks, and `compileall`), which touch or scan the whole clone. Run those only on purpose.

Rules (ruff pinned to **v0.12.2** by the hook; `tool.ruff` in pyproject.toml):

| setting | value |
|---|---|
| `line-length` | **120** |
| `exclude` | `scripts/legacy_model_merger.py` (ruff); `*.ipynb` (ruff hook only) |
| `lint.select` | `E` (pycodestyle), `F` (Pyflakes), `UP` (pyupgrade), `B` (flake8-bugbear), `I` (isort), `G` (flake8-logging-format) |
| `lint.ignore` | `F405`, `F403` (star imports), `E731` (lambda assignment), `B007` (unused loop var), `UP032` (f-string), `G004` (f-string in `.log()`), `UP045` (`X \| None`), `UP035` (deprecated import) |
| `lint.isort` | `known-first-party = ["verl"]` |
| ruff hook args | `--fix --show-fixes --output-format=full` |
| mypy | `pretty`, `ignore_missing_imports`, `explicit_package_bases`, `follow_imports=skip`, `ignore_errors=true` globally; `ignore_errors=false` for `verl.trainer.config.algorithm`, `verl.trainer.ppo.core_algos`, `verl.trainer.ppo.reward`, `verl.workers.reward_manager`, `verl.workers.reward_manager.*` |

Other local hooks in the config (all `pass_filenames: false`): `autogen-trainer-cfg`
(`scripts/generate_trainer_config.sh`), `check-docs-time-info`, `check-docstrings`,
`check-license` (Apache header on every file), `check-device-api-usage`,
`check-dataproto-usage`, `validate-structure`, `check-naming-conventions` (no `veRL`,
no `Sglang`), `check-example-naming`, `check-uv-gpu-only`, `compileall`.

## 3. Datasets (outside the clone)

The CI step passes `--local_dataset_path` only because its runner has the raw HF datasets
on disk; that flag is the *source*, not the output. The output flag is `--local_save_dir`
(default `~/data/<name>`), and the CPU tests hardcode `~/data/gsm8k/` and
`~/data/geo3k/train.parquet` (`tests/utils/dataset/test_rl_dataset_on_cpu.py`). So the
harness writes under `verl-contrib/data` and symlinks the two `~/data` entries to it:

```bash
cd <repo>/verl
export TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 HF_HUB_ENABLE_HF_TRANSFER=0
<repo>/.venv-cpu/bin/python examples/data_preprocess/gsm8k.py \
  --local_save_dir <repo>/data/gsm8k      # openai/gsm8k from HF, ~10 s
<repo>/.venv-cpu/bin/python examples/data_preprocess/geo3k.py \
  --local_save_dir <repo>/data/geo3k      # hiyouga/geometry3k from HF
mkdir -p ~/data
ln -s <repo>/data/gsm8k ~/data/gsm8k
ln -s <repo>/data/geo3k ~/data/geo3k
```

Result: `data/gsm8k/train.parquet` (6.5 MB, 7473 rows), `data/gsm8k/test.parquet`
(1.2 MB, 1319 rows), `data/geo3k/train.parquet` (41 MB), `data/geo3k/test.parquet` (12 MB).
Both scripts finished in well under a minute each (HF reachable, unauthenticated).

## 4. Running the CPU unit tests

Runner (wraps the CI command; nothing is written inside the clone):
`<repo>/scripts/run-cpu-tests.sh <label> <cap_seconds> <pytest args...>`.
It expands to exactly:

```bash
cd <repo>/verl
export PATH=<repo>/.venv-cpu/bin:$PATH      # as under CI's `uv run`
env TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
  <repo>/.venv-cpu/bin/python <repo>/scripts/timeout.py <cap> \
  <repo>/.venv-cpu/bin/python -m pytest \
    -c <repo>/pytest-cpu.ini --rootdir <repo>/verl \
    -s --asyncio-mode=auto -rfEs --continue-on-collection-errors \
    --junitxml <repo>/research/junit/<label>.xml \
    <pytest args...>  > <repo>/research/logs/<label>.log 2>&1
rm -rf <repo>/verl/test_data <repo>/verl/outputs   # only if the run created them
```

Differences from CI, all deliberate: no `-x` (see every failure), `-rfEs` (summary of
fails/errors/skips), junit + log files, the wall-clock cap, the explicit PATH (section 5),
the post-run cleanup (section 8), and `--continue-on-collection-errors` (needed here
because the `megatron`-importing files cannot be collected without megatron-core; CI has
it so CI does not need the flag). Without that flag pytest reports "Interrupted: N errors
during collection" and runs nothing in that directory.

Summarise a run: `<repo>/.venv-cpu/bin/python <repo>/scripts/junit-summary.py <repo>/research/junit/<label>.xml`.

### Baseline results on commit 2ab35292 (2026-10-28)

| directory | passed | failed | errors | skipped | wall | verdict |
|---|---|---|---|---|---|---|
| `tests/` top-level (`test_base_config`, `test_protocol`, `test_protocol_v2`) | 79 | 0 | 0 | 0 | 10 s | clean |
| `tests/trainer` | 438 | 0 | 0 | 1 | 126 s | clean |
| `tests/single_controller` (Ray) | 21 | 0 | 0 | 0 | 188 s | clean; Ray 2.55.1 starts/stops fine, no stray raylet/gcs afterwards |
| `tests/checkpoint_engine` | 30 | 0 | 0 | 0 | 13 s | clean |
| `tests/models` | 26 | 0 | 0 | 0 | 11 s | clean |
| `tests/tools` | 42 | 0 | 0 | 0 | 8 s | clean |
| `tests/model_merger` | 23 | 0 | 0 | 0 | 5 s | clean |
| `tests/plugin` | 0 | 0 | 0 | 0 | 0 s | no `*_on_cpu.py` files; pytest exit 5 "no tests ran" (same on CI) |
| `tests/experimental` | 74 | 0 | 0 | 0 | 81 s | clean after the `~/models` fix (section 5); before it: 61 passed, 13 errors in `test_rate_limited_reward_manager_on_cpu` (missing `~/models/Qwen/Qwen2.5-0.5B-Instruct`) |
| `tests/utils/test_*_on_cpu.py` (32 top-level files) | 405 | 0 | 0 | 10 | 26 s | clean once the venv's bin is first on PATH (section 5); before: `test_config_on_cpu::TestPrintCfgCommand::test_command_with_override` failed because it runs bare `python3 scripts/print_cfg.py` |
| `tests/utils/dataset` | 17 | 0 | 0 | 0 | 29 s | clean after the `~/models` fix (before: 10 failures, all missing tokenizers) |
| `tests/utils/{ckpt,megatron,veomni}` | 47 | 0 | 2 | 2 | 17 s | 2 errors = megatron collection (`test_megatron_checkpoint_manager_on_cpu`, `test_muon_layerwise_bridge_ddp_on_cpu`) |
| `tests/utils/reward_score/reward_score/test_sandbox_fusion_on_cpu.py` | 4 | 0 | 0 | 11 | 16 s | 11 skips need `SANDBOX_FUSION_URL` (unset on CI too) |
| `tests/utils/reward_score/test_sandbox_on_cpu.py` | 2 | 2 | 0 | 2 | 242 s | `test_prime_math` (0.0 != 1.0) and `test_prime_code` (0.6 != 1.0): macOS-only, multiprocessing `spawn` (section 6); `test_parallelism` alone takes 134 s. Combined with the rest of `tests/utils` this file pushes the directory past the 300 s cap |
| **`tests/utils` total** | **475** | **2** | **2** | **25** | ~330 s | sum of the four rows above (415+17+51+15+6 = 504 tests, same count as the one-shot run) |
| `tests/workers` | 278 | 1 | 4 | 40 | 70 s | 1 fail = float32 tolerance (5.72e-6 > atol 5e-6) in `test_chunked_topk_log_probs_on_cpu` on Apple CPU BLAS; 4 errors = megatron collection; before the `~/models` fix also `test_model_config_on_cpu::test_target_modules_raises_on_invalid_type` |

## 5. CI image assets the tests hardcode, and the PATH detail

**`~/models/<org>/<name>`.** CI's image carries HF model snapshots under `~/models`; 24 CPU
tests load tokenizers/processors from those exact paths (`tests/utils/dataset/*`,
`tests/workers/config/test_model_config_on_cpu.py`,
`tests/experimental/reward_loop/test_rate_limited_reward_manager_on_cpu.py`). Weights are
never needed, so the harness holds tokenizer/processor-only snapshots (`*.json`, `*.jinja`,
`*.txt`, `*.model`, `*.tiktoken`; 114 MB total) under
`<repo>/models/<org>/<name>` and symlinks each from
`~/models/<org>/<name>` (`~/models` already existed with unrelated content, which was left
alone). Replay:

```bash
<repo>/.venv-cpu/bin/python - <<'EOF'
import pathlib
from huggingface_hub import snapshot_download
ROOT = pathlib.Path("<repo>/models"); HOME = pathlib.Path.home() / "models"
for rid in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B",
            "Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen3-VL-2B-Instruct", "openai/gpt-oss-20b",
            "deepseek-ai/deepseek-coder-1.3b-instruct"]:
    dst = ROOT / rid
    snapshot_download(rid, local_dir=str(dst), allow_patterns=["*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken"],
                      ignore_patterns=["*.safetensors*", "*.bin", "*.pt", "*.pth", "*.gguf", "*.msgpack", "*.h5", "*.onnx"])
    link = HOME / rid
    if not (link.exists() or link.is_symlink()):
        link.parent.mkdir(parents=True, exist_ok=True); link.symlink_to(dst)
EOF
```

**PATH.** `tests/utils/test_config_on_cpu.py` shells out to bare `python3 scripts/print_cfg.py`.
Under CI's `uv run` the venv's `bin` is first on PATH; here it must be put there explicitly
(`PATH=<repo>/.venv-cpu/bin:$PATH`), otherwise the system
`python3` runs it and fails with `No module named 'hydra'`. The runner exports this.

**Datasets** at `~/data/gsm8k` and `~/data/geo3k`: section 3.

## 6. Every failing / erroring test on this Mac, with cause

All of these are environment causes; none is a verl bug found by this run.

| test | cause |
|---|---|
| `tests/utils/ckpt/test_megatron_checkpoint_manager_on_cpu.py` (collection error) | missing GPU-only dep: imports `megatron.core`; megatron-core needs `triton`, neither installs on macOS. Runs on CI. |
| `tests/utils/megatron/test_muon_layerwise_bridge_ddp_on_cpu.py` (collection error) | same (megatron) |
| `tests/workers/test_megatron_ddp_config_on_cpu.py` (collection error) | same (megatron) |
| `tests/workers/test_megatron_distillation_only_on_cpu.py` (collection error) | same (megatron) |
| `tests/workers/test_megatron_liger_fused_kernels_on_cpu.py` (collection error) | same (megatron) |
| `tests/workers/test_megatron_value_head_cp_layout_on_cpu.py` (collection error) | same (megatron) |
| `tests/utils/reward_score/test_sandbox_on_cpu.py::test_prime_math` (0.0 != 1.0) | macOS-only: `verl.utils.py_functional.timeout_limit` runs the grader in a `multiprocessing.Process`; macOS uses the `spawn` start method (Linux CI forks) and the log shows 7x `Can't pickle <function are_equal_under_sympy ...>: it's not the same object as verl.utils.reward_score.prime_math.are_equal_under_sympy`, so every sympy comparison scores 0. |
| `tests/utils/reward_score/test_sandbox_on_cpu.py::test_prime_code` (0.6..0.8 != 0.9/1.0) | macOS-only: `prime_code/utils.py` runs each case in a spawned `multiprocessing.Process` under a 5-10 s alarm; spawn re-imports torch/verl per case, so cases expire and score 0 (which case varies with load: 0.8 under parallel load, 0.6 alone). |
| `tests/workers/test_chunked_topk_log_probs_on_cpu.py::test_numerical_equivalence[2-32-152064-64-dtype2-5e-06]` | float32 tolerance: max abs diff 5.72e-6 vs `atol=5e-6` on Apple's CPU BLAS; the other parametrisations pass. Treat as macOS-only numeric noise unless a change touches `chunked_topk_log_probs`. |

Fixed by the harness setup (would fail on a bare Mac): the 24 `~/models/...` tokenizer
tests (section 5) and `test_config_on_cpu::TestPrintCfgCommand::test_command_with_override`
(PATH, section 5). Skips (25 in utils, 40 in workers, 1 in trainer, 10 in utils top-level)
are `pytest.importorskip` of vllm/sglang/megatron/torchtitan/veomni/npu backends or
`SANDBOX_FUSION_URL`; CI skips most of them too.

## 7. Safe regression set (what to run against a code change)

`<repo>/scripts/run-cpu-regression.sh` runs the whole CPU
suite minus the eight known-bad items above (six megatron files ignored, `test_sandbox_on_cpu.py`
ignored, the one tolerance parametrisation deselected). Six groups, each capped at 300 s,
~12 min sequentially; pass group names to run a subset, e.g. the ones your change touches:

```bash
<repo>/scripts/run-cpu-regression.sh                 # all six groups
<repo>/scripts/run-cpu-regression.sh trainer utils   # subset
```

| group | what | time alone |
|---|---|---|
| `small` | `tests/test_base_config_on_cpu.py tests/test_protocol_on_cpu.py tests/test_protocol_v2_on_cpu.py tests/checkpoint_engine tests/models tests/tools tests/model_merger` | ~1 min |
| `utils` | `tests/utils` minus the two megatron files and `test_sandbox_on_cpu.py` | ~1.5 min |
| `trainer` | `tests/trainer` (core_algos, losses, metrics, v1 trainer) | ~2 min |
| `workers` | `tests/workers` minus the four megatron files, minus the one tolerance param | ~1.5 min |
| `experimental` | `tests/experimental` (agent loop, reward loop, fully async) | ~1.5 min |
| `single_controller` | `tests/single_controller` (starts real Ray clusters) | ~3 min |

Proof run of the script (2026-10-28, groups launched in parallel pairs, so times are
a little above "alone"):

| group | passed | skipped | failed | wall | exit |
|---|---|---|---|---|---|
| `small` | 200 | 0 | 0 | 25 s | 0 |
| `utils` | 473 | 23 | 0 | 88 s | 0 |
| `trainer` | 438 | 1 | 0 | 116 s | 0 |
| `workers` | 278 | 40 (+1 deselected) | 0 | 103 s | 0 |
| `experimental` | 74 | 0 | 0 | 65 s | 0 |
| `single_controller` | 21 | 0 | 0 | 170 s | 0 |
| **total** | **1484** | **64** | **0** | | `regression set exit=0` |

Targeted runs use the runner directly, e.g.
`<repo>/scripts/run-cpu-tests.sh core 120 tests/trainer/ppo/test_core_algos_on_cpu.py -k gae`
(label, cap in seconds, then any pytest args). Output lands in
`research/logs/<label>.log` and `research/junit/<label>.xml`; summarise with
`scripts/junit-summary.py`.

Safe-to-use test files, by area of verl they exercise (all green here):
`tests/test_protocol_on_cpu.py`, `tests/test_protocol_v2_on_cpu.py` (DataProto);
`tests/trainer/ppo/test_core_algos_on_cpu.py`, `test_loss_aggregation_on_cpu.py`,
`test_dynamic_policy_losses_on_cpu.py`, `test_value_loss_normalization_on_cpu.py`,
`test_score_centering_*_on_cpu.py`, `test_metric_utils_on_cpu.py` (PPO algorithms/metrics);
`tests/trainer/ppo/v1/*_on_cpu.py` (v1 trainer, replay buffer, streaming feed);
`tests/utils/test_padding_on_cpu.py`, `test_linear_cross_entropy_on_cpu.py`,
`test_experimental_torch_functional_on_cpu.py`, `test_model_on_cpu.py`,
`test_config_on_cpu.py`, `tests/utils/dataset/*_on_cpu.py` (data pipeline);
`tests/workers/config/*_on_cpu.py`, `tests/workers/reward_manager/*_on_cpu.py`,
`tests/workers/rollout/*_on_cpu.py` (the vllm/sglang ones self-skip);
`tests/single_controller/*_on_cpu.py` (Ray worker groups);
`tests/experimental/**/*_on_cpu.py`; `tests/checkpoint_engine`, `tests/models`,
`tests/tools`, `tests/model_merger`.

Not testable here: anything importing `megatron`, `vllm`, `sglang`, `flash_attn`,
`torchtitan`, `veomni` (skipped or collection error), `test_sandbox_on_cpu.py`,
`tests/special_*` (never part of the CPU job), and `tests/plugin` (no CPU tests).

## 8. Housekeeping (keep the clone pristine)

- Inside the clone the only artefacts are gitignored: `verl.egg-info/` (editable install)
  and `__pycache__/`. `git -C <repo>/verl status --porcelain`
  was empty after every step of this setup.
- `./test_data/` (written by `tests/utils/dataset` fixtures, not gitignored) and
  `./outputs/<date>/<time>/` (hydra run dirs from `test_config_on_cpu.py`'s
  `python3 scripts/print_cfg.py` subprocess; gitignored via `**/outputs`): the runner deletes
  both after each run unless they pre-existed; delete them yourself after a manual pytest
  invocation (`rm -rf <repo>/verl/test_data <repo>/verl/outputs`).
- Caches live outside the clone: `<repo>/.pytest_cache`
  (via `pytest-cpu.ini`), `.ruff_cache` / `.mypy_cache` (via `RUFF_CACHE_DIR` /
  `MYPY_CACHE_DIR`, section 2), `~/.cache/pre-commit`, `~/.cache/huggingface`.
- Home-directory additions made for the tests: `~/data/{gsm8k,geo3k}` and
  `~/models/{Qwen/*,deepseek-ai/*,openai/*}` are symlinks into
  `<repo>/{data,models}`; nothing pre-existing was touched.
- Never `pre-commit install` in the clone and never `pre-commit run --all-files` (section 2).
- Process leak: on macOS `tests/utils/reward_score/test_sandbox_on_cpu.py` strands
  `multiprocessing` spawn workers (`python -B -c "from multiprocessing.spawn import
  spawn_main ..."`, PPID 1) after the session ends: 150 and 149 workers were left by the
  two runs of this file during the baseline, each as one orphaned process group (killed
  afterwards with `kill -TERM -- -<pgid>`). The regression set skips that file. After running
  it by hand, check with `pgrep -fl 'verl-contrib/.venv-cpu'` and kill the leftover
  process group (`ps -axo pid,pgid,command | grep .venv-cpu`, then `kill -TERM -- -<pgid>`).
  The runner's `timeout.py` only group-kills on cap expiry.

## 9. Appendix: first-pass numbers before the fixes

Earlier whole-`tests/utils` run (before the `~/models`/PATH fixes, with six directories
running in parallel): 464 passed, 13 failed, 2 errors, 25 skipped in 254 s. The 13 were:
10 x missing `~/models/...` tokenizers, 1 x `python3` without hydra on PATH, and
`test_sandbox_on_cpu::test_prime_code` (0.8 != 0.9) / `test_prime_math` (0.0 != 1.0),
which were re-tested in isolation (sections 4 and 6): the root cause is the macOS
`spawn` start method, and CPU contention only changes which prime_code case expires.

The first `tests/utils` / `tests/workers` attempts (without `--continue-on-collection-errors`)
ran 0 tests: pytest stopped at "Interrupted: 2 (resp. 4) errors during collection".
The runner now always passes the flag.

Note: `tests/utils/dataset/*` fixtures write `./test_data/` relative to the cwd, i.e. inside
the clone (not gitignored). The runner deletes it after each run (unless it pre-existed);
if you run pytest by hand, `rm -rf <repo>/verl/test_data` afterwards.
