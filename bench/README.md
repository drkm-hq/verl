# verl GPU benchmark harness on Modal

Runs verl (verl-project/verl) GRPO training and GPU unit tests on Modal H100/H200
from ANY local verl checkout, captures per-step timings to durable storage, and
pulls them to `bench/results/<run-name>/`. Built for A/B runs of local branches
against main.

Status log at the bottom of this file.

**State (2026-10-29 06:30Z): harness validated on main@2ab35292 (probe, smoke, BASELINE-A 31.15 / 31.35 s per step on
two runs, GPU unit tests 503 / 0 / 8). Proof presets added and measured on main: `baseline-a-k1` (P2), `long-prompt`
(P1, synthetic 6k-token GSM8K on the hf-cache volume), `separate-async-2gpu` + `--chaos` (P3). All branch arms measured
on their final shas - see "A/B results": P1 training side -11.9 % and -4.65 GB per rank on 6k-token prompts (2 runs per
arm, +-0.4 %), no regression on BASELINE-A; P2 old_log_prob 3.4-4.7 s -> 0.000 on 5/5 ON runs, training side -10.5 %
(fast nodes) / -12.1 % (same-window slow-node pair) vs OFF; P3 main crashes 3/3 after an agent-loop worker is killed,
TTL-only 0/1, TTL + relaunch completes 5/5 (incl. the final p3b@f6846af0); GPU unit tests 503 / 0 / 8 on every branch
(p0, p1 x2, p2 x2, p3, p3b x2, p4, p4b). Volumes `verl-contrib-hf-cache` (now also `/data/gsm8k-long`) and
`verl-contrib-results` kept; every app stopped by id when its run finished.**

## Layout

| path | role |
|------|------|
| `bench/modal_verl.py` | the Modal app: image, volumes, functions `probe`, `download`, `train`, `gpu_tests` + `*_launch` entrypoints |
| `bench/verl_bench.py` | CLI wrapper: sets env, `modal run --detach`, records app ids, `logs`/`wait`/`pull`/`stop` |
| `bench/summarize.py` | per-step timing table + mean of steps 2..N for a pulled run |
| `bench/follow.py` | bounded log follower: `python3 bench/follow.py <app-id> --seconds 230 --until 'step:1 '` (prints only new lines) |
| `bench/results/<run>/` | pulled artifacts: train.log, metrics.jsonl, meta.json, summary.json, command.txt, nvidia-smi-*.txt, pip-freeze.txt |
| `.venv-bench/` | python 3.12 venv with the `modal` client (1.6.3) |

## Environment rules

- Every modal command runs with `MODAL_PROFILE` set to your own Modal profile (the wrapper passes it through).
- App names are `verl-contrib-<purpose>`. In a shared workspace stop apps by id (`verl_bench.py stop <app-id>`),
  never by name; `stop` refuses an id that is not in the launch registry it keeps.
- Volumes (created with create_if_missing, keep them): `verl-contrib-hf-cache` (mounted at
  `/hf-cache`: `HF_HOME=/hf-cache/hf_home`, GSM8K parquet at `/hf-cache/data/gsm8k/{train,test}.parquet`)
  and `verl-contrib-results` (mounted at `/results`, one directory per run name).
- The verl clone at `<repo>/verl` is read-only; the harness only ships it.

## Image path

`modal.Image.from_registry("verlai/verl:uv.cu130.dev5", add_python=None)` - verl's own CI image
(CUDA 13.0, Ubuntu 24.04, uv 0.11.16, the full uv cache for every backend baked at
`/root/.cache/uv`, no pre-built venv). Our checkout is mounted over the image's repo copy at
`/workspace/verl` (`add_local_dir`, copy=False, so no image rebuild per branch; `.git`,
`__pycache__`, `.venv*`, `docs/_build` are not shipped). Each function first runs
`uv sync --frozen --all-packages --extra vllm --extra fsdp` (offline, from the baked cache) and
then runs verl exactly as CI does:
`uv run --frozen --all-packages --extra vllm --extra fsdp python3 -m verl.trainer.main_ppo ...`
with `ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp"`.

Lock mismatch fallback: `--frozen` fails if the shipped `uv.lock`/`pyproject.toml` differ from what
the image's cache can satisfy (a branch that changes deps). Then run the same commands without
`--frozen` (`uv sync --all-packages --extra ...` resolves/downloads the delta online) - set
`UV_RUN_BASE` in modal_verl.py accordingly, or pin the branch's lock to main's.

## Commands

```
cd <repo>
PY=.venv-bench/bin/python

# 1. image probe (1xH100, ~5 min of GPU time)
$PY bench/verl_bench.py probe --run-name probe-1 [--src /path/to/verl]

# 2. one-time downloads onto the hf-cache volume (no GPU)
$PY bench/verl_bench.py download            # Qwen2.5-0.5B, Qwen3-0.6B, Qwen3-1.7B, Qwen3-8B + GSM8K parquet

# 3. smoke: 1xH100, Qwen2.5-0.5B, GRPO n=4, bs 16, 512/512, 3 steps, TP=1
$PY bench/verl_bench.py train --preset smoke --run-name smoke-main-1 [--src /path/to/verl]

# 4. BASELINE-A: 8xH100, Qwen3-1.7B, GRPO n=8, bs 128, 1024/2048, 6 steps, TP=1, use_kl_loss=True
$PY bench/verl_bench.py train --preset baseline-a --run-name baseline-a-main-1 [--src /path/to/branch]
#    H200 alternative: add --gpu H200:8 ; strict H100 (no H200 substitution): --gpu 'H100!:8'

# 5. GPU unit tests (8xH100, the gpu_unit_tests.yml selection; add --tests "tests/utils tests/workers tests/trainer" for the subset)
$PY bench/verl_bench.py gpu-tests --run-name gpu-tests-main-1

# 6. proof presets (see "Presets" below): the synthetic long-prompt dataset is built once onto the hf-cache volume
$PY bench/verl_bench.py build-dataset --run-name build-gsm8k-long-1          # CPU only, ~2 min; /hf-cache/data/gsm8k-long
$PY bench/verl_bench.py train --preset long-prompt --run-name long-prompt-main-1 [--gpu H200:8]
$PY bench/verl_bench.py train --preset baseline-a-k1 --run-name baseline-a-k1-main-1
$PY bench/verl_bench.py train --preset separate-async-2gpu --run-name sep2-main-1
$PY bench/verl_bench.py train --preset separate-async-2gpu --chaos --run-name sep2-chaos-main-1   # kill after step 2
#    --extra-overrides "a.b=c" appends to the preset's overrides (a later key wins); --overrides replaces them

# polling / results
$PY bench/verl_bench.py logs <app-id> --tail 60 --seconds 90
$PY bench/verl_bench.py wait <fc-id> --seconds 240        # prints RESULT {...} when the function returned
$PY bench/verl_bench.py pull <run-name>                   # modal volume get verl-contrib-results <run-name>/ bench/results/
python3 bench/summarize.py bench/results/<run-name>
$PY bench/verl_bench.py stop <app-id>                     # by app id
```

Any `train` parameter can be given explicitly (`--model`, `--n-gpus`, `--rollout-n`, `--train-batch-size`,
`--max-prompt-length`, `--max-response-length`, `--total-training-steps`, `--tp`, `--use-kl-loss/--no-use-kl-loss`,
`--ppo-mini-batch-size` (default bs/4), `--ppo-max-token-len-per-gpu` (default 24576),
`--gpu-memory-utilization` (default 0.6), `--overrides "a.b=c d.e=f"` free-form hydra overrides).
`--src` is the checkout to ship; its `git describe` (branch@sha[-dirty]) is recorded in meta.json.

## Launch registry format (the registry file itself is not included here)

One line per launch, appended by `verl_bench.py` the moment the app id is known
(and again with state `stopped` by `stop`):

    <UTC time> | <app-id ap-...> | <app name verl-contrib-...> | <run-name / purpose> | <state> | <fc-id>

`stop` refuses any id that is not in the file. Volumes are listed at the top of the file.

## Known gotchas

- macOS has no `timeout`; the wrapper uses `perl -e 'alarm N; exec @ARGV'` to bound `modal app logs`.
- `modal run --detach` with a spawned function: the local entrypoint returns in seconds, the app
  keeps running; a function that crashes at import crash-loops - read the logs within a minute of
  every launch and stop the app by id if it loops.
- First launch builds the Modal image from the 24 GB `verlai/verl:uv.cu130.dev5` (pull + unpack
  on Modal's author, ~10-15 min). Later launches reuse it (keyed by image tag + mount definition).
- The results volume is committed every 60 s by a background thread while a run is in flight, so
  `pull` during a run gives partial `train.log`/`metrics.jsonl`.

## Probe results (2026-10-29, app <modal-app>, results/probe-1/)

- Image `verlai/verl:uv.cu130.dev5` builds on Modal in ~4 min (pull + unpack 24 GB); first attempt
  failed with "unable to determine the version of Python installed in the Image" because the image
  has `python3` but no `python`; fixed with `setup_dockerfile_commands=["RUN ln -sf /usr/bin/python3 /usr/local/bin/python"]`
  (`VERL_PY_MODE=add_python` switches to Modal's standalone 3.12 if ever needed).
- Container: gVisor kernel, 16 vCPU as requested, `free -g` reports 1 TB, `/dev/shm` 1.8 TB (tmpfs, no Ray
  object-store warning), uv 0.11.16 at /usr/bin/uv, python3 3.12.3 (Ubuntu), `UV_CACHE_DIR=/root/.cache/uv`,
  `UV_LINK_MODE=copy`, `VIRTUAL_ENV=/workspace/verl/.venv` (not pre-built), GPU `NVIDIA H100 80GB HBM3`.
- `uv sync --frozen --all-packages --extra vllm --extra fsdp` from the baked cache: **29.6 s** (download
  container: 44 s). Import check: `IMPORT_OK verl 0.10.0.dev vllm 0.32.0 torch 2.13.0+cu130 cuda=True ngpu=1 cuda 13.0`,
  `verl.__file__ = /workspace/verl/verl/__init__.py` (our mounted tree), python = /workspace/verl/.venv/bin/python3.
- Mount semantics: `add_local_dir` is a file-level overlay on the image's own `/workspace/verl` (which even
  carries its `.git`); files deleted in our checkout would survive as stale copies. The harness therefore ships a
  manifest (`results/<run>/manifest.txt`, git-tracked + untracked-unignored files of `--src`) and `_exact_tree()`
  deletes everything else (and the image's `.git`) before `uv sync`; the counts land in meta.json["tree"].

## Smoke (results/smoke-main-2/, app <modal-app>, 2026-10-29 02:13Z)

1xH100 80GB, Qwen2.5-0.5B, GRPO n=4, train_batch_size 16 (ppo_mini 4), prompt 512 / response 512,
3 steps, TP=1, use_kl_loss=True, main@2ab35292. rc=0, trainer wall 241.7 s (incl. Ray/vLLM start,
FlashInfer JIT, CUDA-graph capture); `uv sync` 29.6 s; exact-tree cleanup removed 15 stale image files.

| step | gen | old_log_prob | ref | adv | update_actor | step | tokens | tok/s/GPU |
|------|-----|--------------|-----|-----|--------------|------|--------|-----------|
| 1 (warm-up) | 8.07 | 4.53 | 1.82 | 0.05 | 2.07 | 35.78 | 28008 | 783 |
| 2 | 2.06 | 0.36 | 0.44 | 0.04 | 1.47 | 4.83 | 28023 | 5805 |
| 3 | 2.07 | 1.78 | 0.38 | 0.04 | 1.25 | 5.98 | 29017 | 4850 |
| mean(2..3) | 2.06 | 1.07 | 0.41 | 0.04 | 1.36 | 5.41 | 28520 | 5327 |

Shutdown noise to ignore: after the last step the driver prints a `Traceback` from torch
`_dynamo.utils.dump_compile_times` -> `RuntimeError: DataLoader worker ... killed by signal` (an
atexit hook racing the DataLoader teardown); the exit code is 0 and metrics.jsonl is complete.

## BASELINE-A (the fixed reference config for every A/B run)

`python3 bench/verl_bench.py train --preset baseline-a --run-name baseline-a-<tree>-<n> [--src <checkout>]`

| parameter | value |
|-----------|-------|
| GPU | Modal `H100:8` (8x NVIDIA H100 80GB HBM3, one container); `--gpu H200:8` is the documented alternative, `--gpu 'H100!:8'` forbids H200 substitution |
| model | Qwen/Qwen3-1.7B (HF snapshot 70d244cc, bf16) |
| data | GSM8K train parquet from `examples/data_preprocess/gsm8k.py` (val unused: `val_before_train=False`, `test_freq=-1`) |
| algorithm | GRPO (`algorithm.adv_estimator=grpo`, `use_kl_in_reward=False`), `use_kl_loss=True`, `kl_loss_coef=0.001`, `kl_loss_type=low_var_kl`, `entropy_coeff=0`, lr 1e-6 |
| batch | `train_batch_size=128` prompts, `rollout.n=8` (1024 trajectories/step), `ppo_mini_batch_size=32` (4 optimizer steps per training step) |
| lengths | `max_prompt_length=1024`, `max_response_length=2048`, `filter_overlong_prompts=True`, `truncation=error` |
| actor (FSDP) | `use_remove_padding=True`, `enable_gradient_checkpointing=True`, `use_dynamic_bsz=True`, `ppo_max_token_len_per_gpu=24576`, no param/optimizer offload; ref: dynamic bsz 24576, param_offload=True |
| rollout (vLLM 0.32.0) | `mode=async`, `tensor_model_parallel_size=1`, `gpu_memory_utilization=0.6`, `log_prob_use_dynamic_bsz=True` (24576) |
| trainer | `balance_batch=True`, `logger=[console,file]`, `total_training_steps=6` (step 1 is warm-up: FlashInfer JIT + CUDA graphs + first weight sync), `save_freq=-1`, `critic_warmup=0`, critic disabled |
| env | image `verlai/verl:uv.cu130.dev5` (torch 2.13.0+cu130, vllm 0.32.0, transformers 5.16.1, flash-attn 2.8.3), `uv run --frozen --all-packages --extra vllm --extra fsdp`, Ray workers via `py_executable="uv -v run --frozen ..."` |

The full command line of every run is in `results/<run>/command.txt`; compare A/B runs with
`python3 bench/summarize.py results/<run>` (mean of steps 2..6).

### BASELINE-A measured on main@2ab35292 (results/baseline-a-main-1/, app <modal-app>, 2026-10-29 02:24Z)

8x NVIDIA H100 80GB HBM3 (driver 610.57.04), scheduled ~5 min after launch; `uv sync` 25.8 s; trainer wall
516.0 s for 6 steps (incl. ~3.5 min of Ray/vLLM/FSDP start-up); rc=0. Reward stays ~0 (Qwen3-1.7B base on
GSM8K with 6 steps) - this is a timing baseline, not a quality run.

| step | gen | old_log_prob | ref | adv | update_actor | step | tokens/step | tok/s/GPU | resp len |
|------|-----|--------------|-----|-----|--------------|------|-------------|-----------|----------|
| 1 (warm-up) | 20.42 | 8.50 | 7.54 | 1.05 | 9.63 | 70.85 | 1,332,619 | 2351 | 1217 |
| 2 | 12.40 | 3.28 | 3.19 | 0.53 | 8.47 | 29.97 | 1,316,396 | 5491 | 1199 |
| 3 | 12.41 | 3.50 | 4.15 | 0.53 | 8.65 | 31.49 | 1,357,637 | 5390 | 1241 |
| 4 | 12.34 | 3.77 | 2.65 | 0.51 | 8.31 | 29.84 | 1,308,032 | 5478 | 1195 |
| 5 | 12.35 | 3.44 | 4.26 | 0.52 | 9.31 | 31.87 | 1,359,497 | 5332 | 1245 |
| 6 | 14.35 | 3.47 | 2.90 | 0.96 | 8.81 | 32.59 | 1,387,297 | 5320 | 1272 |
| **mean(2..6)** | **12.77** | 3.49 | 3.43 | 0.61 | **8.71** | **31.15** | 1,345,772 | 5402 | 1230 |

Step-to-step spread on main: gen 12.3-14.3 s, update_actor 8.3-9.3 s, step 29.8-32.6 s (about +-5%);
an A/B claim below ~5% needs repeated runs (`--run-name baseline-a-<tree>-2`, `-3`).

## GPU unit tests on main@2ab35292 (results/gpu-tests-main-1/, app <modal-app>, 8xH200, 2026-10-29 02:23Z)

Exactly the `gpu_unit_tests.yml` job: `uv run --frozen --all-packages --extra sglang --extra megatron --extra ci`
(`uv sync` 38.7 s), `uv pip install mlflow`, then
`pytest -s <the workflow's 13 --ignore-glob filters> tests/` (without CI's `-x`, so every test runs and the
counts are complete; `--junitxml` added for parsing), then the two torchrun tests. Ran on H200:8 because
H100:8 was busy with BASELINE-A; the selection does not depend on the GPU model.

- pytest: **529 collected, 497 passed, 24 failed, 8 skipped, 8 min 51 s** (rc=1). Full set, so no subset needed.
- torchrun `tests/utils/test_special_linear_cross_entropy_tp.py` (8 ranks, LOW_MEMORY=True): rc=0, 39.7 s.
- torchrun `tests/utils/test_special_megatron_kl_loss_tp.py` (2 ranks): rc=0, 33.8 s.
- The 24 failures are environmental, two root causes:
  1. 5 tests load models from local dirs verl's CI runners pre-populate:
     `tests/checkpoint_engine/test_correctness_on_gpu.py::test_nccl_checkpoint_engine[2-6-False-False]`
     (`/root/models/Qwen/Qwen3-8B-Base`) and `tests/utils/test_activation_offload.py::test_activation_offloading[fsdp-2|fsdp-4|fsdp2-2|fsdp2-4]`
     (`/root/models/Qwen/Qwen2.5-0.5B-Instruct`): `OSError: Repo id must be in the form 'repo_name'...`.
  2. 19 cascades: the first crash left Ray initialised in the pytest process, so every later `ray.init()` test
     failed with `RuntimeError: Maybe you called ray.init twice by accident?`:
     the other 3 `test_nccl_checkpoint_engine[2-6-*]` cases and all of `tests/single_controller/`
     (test_colocated_workers, test_colocated_workers_fused, test_data_transfer, test_device_mesh_register,
     test_driverfunc_to_worker, test_high_level_scheduling_api, test_nested_worker, test_ray_collectives,
     test_rvdz, test_split_resource_pool x4, test_worker_group_basics, test_worker_group_torch x2).
  Fix in the harness: `download --models ...` the models the tests expect and link them under `/root/models/`
  before pytest (see the gpu-tests-main-2 entry below once re-run).
- verl's tests load models from `~/models/<org>/<name>` (what the CI runners pre-populate). `gpu_tests`
  links every snapshot in the hf-cache volume to `/root/models/<org>/<name>` before pytest; add models with
  `download --models a,b --no-gsm8k --run-name download-N`. The tests reference (selected set needs only the
  first three): Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen3-8B-Base, Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-0.5B,
  Qwen/Qwen3-0.6B, HuggingFaceTB/SmolLM2-135M-Instruct, deepseek-ai/deepseek-coder-1.3b-instruct (+ VL / 30B /
  Skywork reward models used only by ignored test dirs).
- `git_describe` marks the tree `-dirty` when `git status --porcelain` is non-empty, including untracked files
  (smoke-main-2 saw a transient `test_data/test_video.json` left by a CPU-test run in the clone); the shipped
  manifest (`results/<run>/manifest.txt`) is the exact file list that ran.
- `pytest -x` (CI) stops at the first failure; the harness drops it so counts are complete, but one test that
  crashes after `ray.init()` makes every later Ray test fail with "ray.init twice" - read the first failure.

## Status log (UTC)

- 2026-10-29 01:50 probe-1 attempt 1 (<modal-app>): image built in 236 s, `modal run` refused the image (no `python`).
- 02:01 probe-1 attempt 2 (<modal-app>): python symlink fix; IMPORT_OK on 1xH100, uv sync 29.6 s.
- 02:08 download (<modal-app>): Qwen2.5-0.5B, Qwen3-0.6B, Qwen3-1.7B, Qwen3-8B (46 s) + GSM8K parquet.
- 02:11 smoke-main-1 (<modal-app>): failed before training, `snapshot_download(local_files_only=True)`
  rejected the pattern-filtered snapshot; model resolution now reads the HF cache layout directly.
- 02:13 smoke-main-2 (<modal-app>): 3 steps OK, mean(2..3) gen 2.06 / update_actor 1.36 / step 5.41 s.
- 02:19 baseline-a-main-1 (<modal-app>): queued ~5 min for H100:8, 6 steps OK, wall 516 s,
  mean(2..6) gen 12.77 / update_actor 8.71 / step 31.15 s.
- 02:23 gpu-tests-main-1 (<modal-app>, H200:8): 497 passed / 24 failed / 8 skipped (missing ~/models + Ray cascade).
- 02:36 download-2 (<modal-app>): the five test models.
- 02:39 gpu-tests-main-2 (<modal-app>, H100:8): re-run with /root/models links - see below.
- 03:37 build-gsm8k-long-1 (<modal-app>): WRONG output (transformers 5.x `apply_chat_template(tokenize=True)`
  returns a BatchEncoding, `len()` was 2, every prompt grew to ~27k words, 31 min); fixed count + `num_proc=16`.
- 03:37 baseline-a-k1-main-1 (<modal-app>, H100:8): 6 steps OK, mean(2..6) old_log_prob 3.72 / update_actor 9.11 / step 31.40.
- 03:38 sep2-main-1 (<modal-app>, H100:2): separate_async preset OK, 6 steps, step 3.78 s, wall 276 s.
- 03:43 sep2-chaos-main-1 (<modal-app>): crashed (ActorDiedError) rc=1; the kill landed after step 4 because the
  killer side process took 9 s to start -> pre-armed killer. 04:02 sep2-chaos-main-2 (<modal-app>): crashed, kill still
  10 s late (GCS listing after the trigger) -> handle resolved while arming. 04:21 sep2-chaos-main-3 (<modal-app>):
  kill 0.0 s after step 2, crashed rc=1 7 s later. Three main chaos runs, three crashes.
- 03:54 baseline-a-p1-on-1 (<modal-app>, p1@4bcbbeb9) and gpu-tests-p1-1 (<modal-app>, p1@4bcbbeb9:
  503 passed / 0 failed / 8 skipped, torchrun 2/2). 04:06 baseline-a-k1-p2-on-1 / -off-1 (<modal-app> /
  <modal-app>, p2@1e45a508). 04:12 baseline-a-main-2 (<modal-app>): step 31.35 s (main-1: 31.15).
- 04:13 build-gsm8k-long-2 (<modal-app>): correct dataset in 108 s (train 5977-6024 tokens, mean 5998.6).
- 04:17 long-prompt trio launched on H200:8 (<modal-app>, <modal-app>, <modal-app>):
  never scheduled in 15 min; stopped in the queue and relaunched on H100:8 at 04:25 (long-prompt-main-1 <modal-app>,
  long-prompt-p1-on-2 <modal-app> from p1@b32c777e (9 commits), long-prompt-p1-off-1 <modal-app>).
- 04:20 P3 arms (p3@5e6f6b74, H100:2): sep2-chaos-p3-on-1 <modal-app> COMPLETED (rc=0, 6 steps, step 5 13.9 s with
  dead_workers_relaunched=1, prompts_redispatched=4), sep2-chaos-p3-off-1 <modal-app> crashed like main,
  sep2-p3-on-1 <modal-app> rc=0.
- 04:30 repeats: sep2-main-2 <modal-app>, sep2-p3-on-2 <modal-app>, sep2-chaos-p3-on-2
  <modal-app>, gpu-tests-p2-1 <modal-app>, baseline-a-k1-p2-on-2 <modal-app>,
  baseline-a-k1-p2-off-2 <modal-app>, baseline-a-k1-main-2 <modal-app>, baseline-a-p1-on-2
  <modal-app> (p1@b32c777e), gpu-tests-p3-1 <modal-app>, gpu-tests-p4-1 <modal-app>,
  gpu-tests-p0-1 <modal-app> (p0@6684917e).
- 04:45 final-sha arms: baseline-a-k1-p2-on-3 / -on-4 <modal-app> / <modal-app> (p2@5cb89c52; both on
  slow nodes: ref +50 %, gen +15 %), gpu-tests-p2-2 <modal-app> (5cb89c52: 503/0/8), gpu-tests-p1-2
  <modal-app> (p1@b32c777e), gpu-tests-p4b-1 <modal-app> (d8c16498, stopped in the queue when
  the final sha landed; relaunched 05:07 as <modal-app> from p4b@97a43c92).
- 04:52 long-prompt-main-2 <modal-app>; long-prompt-p1-on-3 <modal-app> (b32c777e, stopped in the
  queue after p1 moved) -> relaunched 04:57 as <modal-app> from p1@a11ca00b: 6.22 / 5.83 / 23.73 s, 11.92 GB
  (reproduces on-2 within 0.5 %).
- 05:00 gpu-tests results: p0-1 503/0/8 (26:32), p3-1 503/0/8 (33:18), p4-1 503/0/8 (32:42), p2-1 503/0/8 (25:23), all torchrun
  2/2 rc=0, no new failures on any branch.
- 05:05 same-window control pair for P2's slow-node runs: baseline-a-k1-p2-off-3 <modal-app> and -on-5
  <modal-app> (p2@2b259b2d, final HEAD): both slow nodes again (801 / 711 s wall); ON vs OFF in the same
  window: train sum 15.60 vs 17.74 s (-12.1 %), old_log_prob 0 vs 4.70 s.
- 05:28 P3 final split arms (p3b@e43e784c contains p3a@7f43e43e): sep2-chaos-p3b-on-1 <modal-app> and -on-2
  <modal-app> COMPLETED (step 5 13.4 / 14.3 s, relaunched=1, redispatched=4); sep2-chaos-p3b-off-1
  <modal-app> crashed; sep2-chaos-p3a-ttl-1 <modal-app> (TTL only) crashed; sep2-p3b-on-1
  <modal-app> rc=0 step 4.17 s; gpu-tests-p3b-1 <modal-app>.
- 05:30 long-prompt-main-2: 8.03 / 6.59 / 25.80 s, 16.56 GB (main-1 within 0.5 %). gpu-tests-p1-2 (b32c777e) 503/0/8 (33:06),
  gpu-tests-p4b-1 (97a43c92) 503/0/8 (26:37), gpu-tests-p2-2 (5cb89c52) 503/0/8 (25:18); all torchrun 2/2.
- 06:00 p3b final product commit f6846af0 landed: sep2-chaos-p3b-on-final <modal-app> COMPLETED (kill right after
  step 2; step 5 6.1 s relaunched=1 / orphaned=4, step 6 10.8 s inflight_reissued=8; wall 273 s); gpu-tests-p3b-2
  <modal-app> from f6846af0. gpu-tests-p3b-1 (e43e784c): pytest rc=0, 0 failures.

## Presets

Every preset is a `train` config (`bench/verl_bench.py` `PRESETS`); the full command line of each run is in
`results/<run>/command.txt`. Common to all: GRPO, `use_kl_in_reward=False`, `use_remove_padding`, gradient
checkpointing, dynamic bsz, lr 1e-6, vLLM async rollout TP=1 `gpu_memory_utilization=0.6`, `balance_batch`,
`val_before_train=False`, `test_freq=-1`, `save_freq=-1`, console+file logger; `trainer.use_v1=True` is verl's default.

| preset | GPUs | model | data | batch / mini / n | prompt / resp | steps | token budget | what differs from BASELINE-A |
|--------|------|-------|------|------------------|---------------|-------|--------------|------------------------------|
| `smoke` | H100 x1 | Qwen2.5-0.5B | gsm8k | 16 / 4 / 4 | 512 / 512 | 3 | 24576 | 1 GPU sanity run |
| `baseline-a` | H100 x8 | Qwen3-1.7B | gsm8k | 128 / 32 / 8 | 1024 / 2048 | 6 | 24576 | the reference (use_kl_loss=True) |
| `baseline-a-k1` (P2) | H100 x8 | Qwen3-1.7B | gsm8k | 128 / **128** / 8 | 1024 / 2048 | 6 | 24576 | `actor.ppo_mini_batch_size=128`: one optimizer step per step (k=1, strictly on-policy) |
| `long-prompt` (P1) | H200 x8 (H100 queue was saturated; both arms on H200) | Qwen3-1.7B | **gsm8k-long** (~6000-token prompts) | 64 / 16 / 8 | **6144 / 1024** | 6 | **32768** | `data_dir=/hf-cache/data/gsm8k-long`; K/T = response/(prompt+response) ~ 0.14 instead of 0.94 |
| `gsm8k-tool` (P4) | H100 x8 | Qwen3-1.7B | **gsm8k-tool** (verl's `gsm8k_tool_agent_loop.py`: system prompt asking for `calc_gsm8k_reward` calls, `agent_name=tool_agent`, ground truth in `tools_kwargs`) | 64 / **64** / 8 | 1024 / 1024 | 6 | 24576 | V1 `trainer_mode=sync` + TransferQueue (V1 forces `transfer_queue.enable=True`), vLLM async, `multi_turn.enable=True` `format=hermes` `max_assistant_turns=3` `tool_config_path=/workspace/verl/bench_extras/gsm8k_tool_config.yaml`, `agent.default_agent_loop=tool_agent`, `reward.custom_reward_function.path=/workspace/verl/bench_extras/gsm8k_tool_reward.py` (`compute_score`), `trainer.rollout_data_dir={run_dir}/rollouts`, `trainer.validation_data_dir={run_dir}/val`; ENABLED arm `--extra-overrides "actor_rollout_ref.rollout.multi_turn.turn_level_rewards=True algorithm.adv_estimator=turn_rloo"`, CONTROL `"algorithm.adv_estimator=grpo"` |
| `separate-async-2gpu` (P3) | H100 x2 (1 trainer + 1 standalone vLLM) | Qwen3-0.6B | gsm8k | 16 / 8 / 4 (`parameter_sync_step=2`) | 512 / 512 | 6 | 24576 | V1 `trainer_mode=separate_async`, `hybrid_engine=False`, TQ, `checkpoint_engine.backend=nccl`, `agent.num_workers=4`, fsdp2, `use_kl_loss=False`, `reward_manager=dapo`, `data.truncation=left`, `gen_batch_size=1` (mirrors `tests/special_e2e/run_v1_separate_async.sh`); `trainer.n_gpus_per_node=1`, `rollout.nnodes=1`, `rollout.n_gpus_per_node=1` |

`long-prompt` data: `bench/build_gsm8k_long.py` (run in the container by `build-dataset`, ~2 min with 16 processes)
prepends to every GSM8K user turn a header plus a slice of Project Gutenberg #1342 (Pride and Prejudice, sha256
`3f6bb9d6...`, cached on the volume as `filler.txt`) starting at word `(row * 7919) % 127360`, sized per row until the
Qwen3 chat-templated prompt is 6000 +- 8 tokens (train: 7473 rows, 5977-6024 tokens, mean 5998.6, none over 6144;
test: 1319 rows); the original question + verl instruction stays last, schema identical to
`examples/data_preprocess/gsm8k.py` (`data_source`, `prompt`, `ability`, `reward_model.ground_truth`, `extra_info`).
Different per-row prefixes keep vLLM prefix caching from short-circuiting prefill across rows (the n=8 rollouts of one
prompt still share it, as in any GRPO run). Provenance and parquet sha256s: `results/build-gsm8k-long-2/build.json`.

### `--chaos` (separate-async-2gpu)

`--chaos [--chaos-step 2] [--chaos-class AgentLoopWorkerTQ] [--chaos-wall-cap-s 1200] [--chaos-hang-s 600]`:
`train()` watches `metrics.jsonl`; when step N-1 completes it arms `bench/chaos_kill.py` in the trainer's venv
(connects to the job's Ray cluster with `ray.init(address="auto")`, joins `ray.util.list_named_actors(all_namespaces=True)`
with `ray._private.state.actors()` to find the alive `AgentLoopWorkerTQ` actors - named `agent_loop_worker_<i>_<hex>` -
and resolves the first one's handle; the GCS listing takes ~8 s, hence the pre-arming); when step N completes the
harness touches a trigger file and the killer runs `ray.kill(handle, no_restart=True)` within ~0.1 s, then verifies the
pid is gone. The harness then caps the run: `wall_cap_s` from launch, and `hang_s` after the kill with no new step
(SIGKILL of the trainer's process group, `ray stop --force`). `summary.json["chaos"]` records `outcome`
(`completed` rc=0 and all steps | `crashed` rc!=0 | `hung` cap fired | `exited_early`), `steps_reached`,
`kill.{step_before,t_rel_s,trigger_to_kill_s,killed.name,pid_gone_after_s,survivors}`, `kill_to_next_step_s`,
`first_step_after_kill`, `terminated_by`. `python3 bench/summarize.py results/<run>` prints them.

## A/B recipe for a local branch

```
cd <repo>; PY=.venv-bench/bin/python
# the branch must be checked out in its own worktree/clone (the harness ships a directory, not a ref)
$PY bench/verl_bench.py train --preset baseline-a --src /path/to/verl-branch --run-name baseline-a-<branch>-1
$PY bench/verl_bench.py train --preset baseline-a --run-name baseline-a-main-2          # paired main run, same hour
$PY bench/verl_bench.py gpu-tests --src /path/to/verl-branch --run-name gpu-tests-<branch>-1
# ... poll with `logs <app-id>` / `python3 bench/follow.py <app-id>`; then
$PY bench/verl_bench.py pull baseline-a-<branch>-1 && python3 bench/summarize.py bench/results/baseline-a-<branch>-1
```

Rules of evidence: compare mean(2..6) of `timing_s/gen`, `timing_s/update_actor`, `timing_s/step` and
`perf/throughput` against a main run from the same session (Modal hosts differ: the actual GPU/driver is in
`nvidia-smi-before.txt`, the host in meta.json); the manifest + `src_git` in meta.json prove which tree ran;
`pip-freeze.txt` proves the environment is the locked one (identical unless the branch changes `uv.lock`).
Extra hydra overrides for a branch's new flag: `--extra-overrides "actor_rollout_ref.actor.my_flag=True"`
(appended to the preset's overrides; a later key wins; `--overrides` would replace the preset's list).

### Branch arms (the exact commands; main arms already in `results/`)

```
cd <repo>; PY=.venv-bench/bin/python; WT=<repo>/wt
# P1 loss-row-only LM head (main arms: long-prompt-main-1 on H200:8, baseline-a-main-1/-2 on H100:8)
$PY bench/verl_bench.py train --preset long-prompt --gpu H200:8 --src $WT/p1 --run-name long-prompt-p1-on-1 \
    --extra-overrides "actor_rollout_ref.actor.fsdp_config.response_only_lm_head=True"     # ref follows via oc.select
$PY bench/verl_bench.py train --preset long-prompt --gpu H200:8 --src $WT/p1 --run-name long-prompt-p1-off-1 \
    --extra-overrides "actor_rollout_ref.actor.fsdp_config.response_only_lm_head=False"
$PY bench/verl_bench.py train --preset baseline-a --src $WT/p1 --run-name baseline-a-p1-on-1 \
    --extra-overrides "actor_rollout_ref.actor.fsdp_config.response_only_lm_head=True"
$PY bench/verl_bench.py gpu-tests --src $WT/p1 --run-name gpu-tests-p1-1
# P2 old-log-prob skip (main arm: baseline-a-k1-main-1, H100:8)
$PY bench/verl_bench.py train --preset baseline-a-k1 --src $WT/p2 --run-name baseline-a-k1-p2-on-1 \
    --extra-overrides "actor_rollout_ref.actor.skip_old_log_prob_when_on_policy=True"
$PY bench/verl_bench.py train --preset baseline-a-k1 --src $WT/p2 --run-name baseline-a-k1-p2-off-1 \
    --extra-overrides "actor_rollout_ref.actor.skip_old_log_prob_when_on_policy=False"
$PY bench/verl_bench.py gpu-tests --src $WT/p2 --run-name gpu-tests-p2-1
# P3 self-healing (main arms: sep2-main-{1,2}, sep2-chaos-main-{1,2,3}, H100:2); final split: p3b (relaunch) stacked on p3a (TTL)
ON="trainer.v1.sampler.inflight_ttl_s=60 trainer.v1.sampler.max_inflight_reissues=3 actor_rollout_ref.rollout.agent.relaunch_dead_workers=True"
OFF="trainer.v1.sampler.inflight_ttl_s=null actor_rollout_ref.rollout.agent.relaunch_dead_workers=False"
$PY bench/verl_bench.py train --preset separate-async-2gpu --chaos --src $WT/p3b --run-name sep2-chaos-p3b-on-final --extra-overrides "$ON"
$PY bench/verl_bench.py train --preset separate-async-2gpu --chaos --src $WT/p3b --run-name sep2-chaos-p3b-off-1 --extra-overrides "$OFF"
$PY bench/verl_bench.py train --preset separate-async-2gpu --src $WT/p3b --run-name sep2-p3b-on-1 --extra-overrides "$ON"
$PY bench/verl_bench.py train --preset separate-async-2gpu --chaos --src $WT/p3a --run-name sep2-chaos-p3a-ttl-1 \
    --extra-overrides "trainer.v1.sampler.inflight_ttl_s=60 trainer.v1.sampler.max_inflight_reissues=3"   # TTL only
# (the pre-split arms sep2-chaos-p3-{on,off}-1, sep2-p3-on-1 used --src $WT/p3 with the ON/OFF strings minus max_inflight_reissues)
$PY bench/verl_bench.py gpu-tests --src $WT/p0 --run-name gpu-tests-p0-1   # likewise p1, p2, p3, p3b, p4, p4b
```

## A/B results (2026-10-29, image verlai/verl:uv.cu130.dev5, main@2ab35292; branch sha per row)

Reading rules: the TRAINING-SIDE phases (`timing_s/old_log_prob`, `ref`, `update_actor`, their sum) are the metrics;
`timing_s/step` and `gen` are reported separately because `gen` varies ~10 % between identical runs on this harness
(main BASELINE-A: 12.77 / 12.39 s; flag-off arms 10.5 / 10.9 s) while the two main BASELINE-A step times agree within 0.3 %
(31.15 / 31.35 s). Means are over steps 2..6; "peak mem" is `actor/perf/max_memory_allocated_gb` (per rank). Every run's
tree is proven by `meta.json` (`src_git`, `src_head`, `tree.digest_match=True`) and `command.txt`. Per-branch copies with
the discussion: `packages/p<N>/bench/RESULTS.md`.

### P1 loss-row-only LM head (`actor_rollout_ref.actor.fsdp_config.response_only_lm_head`)

`long-prompt` preset, 8x H100, K/T = 0.096 (prompt 5999 / response ~636 tokens; GSM8K responses under a 1024 cap):

| run | tree | old_log_prob | ref | update_actor | train sum | step | gen | peak mem GB |
|-----|------|--------------|-----|--------------|-----------|------|-----|-------------|
| long-prompt-main-1 | main@2ab35292 | 8.098 | 6.595 | 25.928 | 40.62 | 62.24 | 16.99 | 16.56 |
| long-prompt-p1-off-1 | p1@b32c777e OFF (9 commits) | 8.055 | 6.552 | 26.000 | 40.61 | 61.87 | 16.61 | 16.57 |
| long-prompt-p1-on-2 | p1@b32c777e ON (9 commits) | **6.158** | **5.857** | **23.577** | **35.59** | 54.20 | 15.96 | **11.90** |
| ON vs OFF (same tree) | | **-23.6 %** | **-10.6 %** | **-9.3 %** | **-12.4 %** | -12.4 % | -4 % | **-4.67 GB (-28 %)** |
| long-prompt-main-2 | main@2ab35292 | 8.030 | 6.593 | 25.801 | 40.42 | 62.09 | 17.06 | 16.56 |
| long-prompt-p1-on-3 | p1@a11ca00b ON (10 commits, final HEAD) | **6.224** | **5.825** | **23.730** | **35.78** | 53.96 | 15.57 | **11.92** |
| main mean +- spread (main-1, main-2) | | 8.06 +- 0.03 | 6.59 +- 0.00 | 25.86 +- 0.06 | 40.52 +- 0.10 | 62.16 +- 0.07 | 17.02 +- 0.04 | 16.56 |
| p1 ON mean +- spread (on-2, on-3) | | 6.19 +- 0.03 | 5.84 +- 0.02 | 23.65 +- 0.08 | 35.68 +- 0.10 | 54.08 +- 0.12 | 15.77 +- 0.20 | 11.91 |
| **ON vs main (2 runs each)** | | **-23.2 %** | **-11.4 %** | **-8.5 %** | **-11.9 %** | -13.0 % | -7 % | **-4.65 GB (-28 %)** |

OFF equals main on every column (the branch is a no-op when disabled); correctness columns ON vs OFF: entropy 0.134 /
0.134, pg_clipfrac 0.002 / 0.002, ppo_kl 0.000, kl_loss 0.002, rollout_probs_diff 0.005, reward 0.130 / 0.127.

BASELINE-A no-regression (K/T = 0.94, head is 17 % of forward FLOPs, expected -1 %):

| run | tree | old_log_prob | ref | update_actor | train sum | step | gen | peak mem GB |
|-----|------|--------------|-----|--------------|-----------|------|-----|-------------|
| baseline-a-main-1 | main@2ab35292 | 3.491 | 3.430 | 8.709 | 15.63 | 31.15 | 12.77 | 14.70 |
| baseline-a-main-2 | main@2ab35292 | 3.547 | 3.384 | 8.922 | 15.85 | 31.35 | 12.39 | 14.62 |
| baseline-a-p1-on-1 | p1@4bcbbeb9 ON (4 commits) | 3.484 | 3.059 | 8.069 | 14.61 | 29.23 | 10.51 | 14.51 |
| baseline-a-p1-on-2 | p1@b32c777e ON (9 commits) | 3.379 | 3.036 | 8.086 | 14.50 | 28.54 | 10.47 | 14.56 |
| **main mean +- spread** | | 3.52 +- 0.03 | 3.41 +- 0.02 | 8.82 +- 0.11 | 15.74 +- 0.11 | 31.25 +- 0.10 | 12.58 +- 0.19 | 14.66 |
| **p1 ON mean +- spread** | | 3.43 +- 0.05 | 3.05 +- 0.01 | 8.08 +- 0.01 | 14.56 +- 0.05 | 28.89 +- 0.35 | 10.49 +- 0.02 | 14.54 |

Correctness columns identical across the three: entropy 0.203 / 0.202 / 0.204, pg_clipfrac 0, ppo_kl 0.000, kl_loss 0.001,
rollout_probs_diff 0.006. GPU unit tests on p1@4bcbbeb9 (`gpu-tests-p1-1`): **503 passed / 0 failed / 8 skipped**, torchrun
2/2 rc=0 - identical to main.

### P2 old-log-prob skip (`actor_rollout_ref.actor.skip_old_log_prob_when_on_policy`), `baseline-a-k1`, 8x H100

Headline: `timing_s/old_log_prob` 3.4-3.7 s -> **0.000 s on every ON run** (the pass is skipped), with `update_actor`
read ON vs OFF on the same tree (it absorbs the log-prob return and the no-grad entropy metric), and `pg_clipfrac` /
`ppo_kl` exactly 0 on the ON arms. NODE VARIANCE: runs on-3 and on-4 (launched 04:45-04:52) landed on slow nodes -
every phase is slower there, including `ref` (+50 %) and `gen` (+15 %), which P2 does not touch - so cross-node step or
update_actor comparisons need the same-window pair off-3 / on-5; the within-run equality columns are unaffected.
Branch shas: arms at 1e45a508 (5 commits) and 5cb89c52 (9); final HEAD 2b259b2d (10 commits, validation/test/docs only).

| run | tree | old_log_prob | ref | update_actor | train sum | step | gen | peak mem GB | pg_clipfrac / ppo_kl | entropy |
|-----|------|--------------|-----|--------------|-----------|------|-----|-------------|----------------------|---------|
| baseline-a-k1-main-1 | main@2ab35292 | 3.719 | 2.880 | 9.112 | 15.71 | 31.40 | 12.79 | 15.07 | 0.000 / 0.000 | 0.204 |
| baseline-a-k1-p2-off-1 | p2@1e45a508 OFF (5 commits) | 3.509 | 2.593 | 8.603 | 14.70 | 29.20 | 10.88 | 14.97 | 0.000 / 0.000 | 0.207 |
| baseline-a-k1-p2-on-1 | p2@1e45a508 ON | **0.000** | 3.094 | 9.262 | **12.36** | 26.39 | 10.46 | 14.88 | **0 / 0 exact** | 0.206 |
| baseline-a-k1-main-2 | main@2ab35292 | 3.644 | 2.896 | 9.408 | 15.95 | 32.16 | 13.16 | 15.05 | 0.000 / 0.000 | 0.205 |
| baseline-a-k1-p2-off-2 | p2@1e45a508 OFF | 3.431 | 2.645 | 8.323 | 14.40 | 29.05 | 10.85 | 14.91 | 0.000 / 0.000 | 0.205 |
| baseline-a-k1-p2-on-3 | p2@5cb89c52 ON (9 commits; branch HEAD later 2b259b2d, validation-only) | **0.000** | 4.271 (slow host: +50 % on this flag-independent phase, gen +20 %) | 11.672 | 15.94 | 34.97 | 14.60 | 14.92 | **0 / 0 exact** | 0.207 |
| baseline-a-k1-p2-on-4 | p2@5cb89c52 ON | **0.000** | 4.383 (slow host again: ref/gen are flag-independent) | 11.992 | 16.37 | 35.37 | 14.62 | 14.89 | **0 / 0 exact** | 0.205 |
| baseline-a-k1-p2-off-3 | p2@2b259b2d OFF (10 commits, final HEAD); same-window pair with on-5, both on slow nodes (wall 801 / 711 s vs ~500 s) | 4.702 | 3.418 | 9.623 | 17.74 | 39.11 | 14.83 | 15.07 | 0.000 / 0.000 | 0.207 |
| baseline-a-k1-p2-on-5 | p2@2b259b2d ON (final HEAD) | **0.000** | 4.213 | 11.390 | **15.60** | 33.78 | 13.75 | 15.00 | **0 / 0 exact** | 0.203 |
| same-window ON vs OFF (on-5 vs off-3) | | -4.70 s | +0.8 | +1.77 (+18 %) | **-2.14 s (-12.1 %)** | -5.3 s (-13.6 %) | | | | |
| fast-node pairs ON vs OFF (on-1/on-2 vs off-1/off-2, 1e45a508) | | -3.47 s | +0.67 | +1.26 (+15 %) | **-1.53 s (-10.5 %)** | -1.3 s | | | | |
| baseline-a-k1-p2-on-2 | p2@1e45a508 ON | **0.000** | 3.490 | 10.180 | **13.67** | 29.20 | 12.77 | 15.07 | **0 / 0 exact** | 0.206 |

### P3 self-healing (`trainer.v1.sampler.inflight_ttl_s=60`, `rollout.agent.relaunch_dead_workers=True`), `separate-async-2gpu`, 2x H100

| run | tree | kill after step 2 | outcome | steps / 6 | kill -> next step | recovery metrics |
|-----|------|-------------------|---------|-----------|-------------------|------------------|
| sep2-chaos-main-1 / -2 / -3 | main@2ab35292 | yes (main-1/-2 landed late: after step 4) | **crashed x3** (rc=1, `ActorDiedError` in `AgentLoopManagerTQ.generate_sequences`, ~7 s after the kill) | 5 / 4 / 4 | 2.4-4.2 s (in-flight step only) | - |
| sep2-chaos-p3-off-1 | p3@5e6f6b74 OFF | yes | **crashed** (same traceback) | 4 | 2.4 s | - |
| sep2-chaos-p3-on-1 | p3@5e6f6b74 ON | yes | **completed** rc=0 | 6 | 2.8 s | step 5: 13.9 s (vs 4.2), `rollout/dead_workers_relaunched=1`, `prompts_redispatched=4`, `inflight_reissued=0` |
| sep2-chaos-p3-on-2 | p3@5e6f6b74 ON | yes | **completed** rc=0 | 6 | 3.3 s | step 5: 24.5 s (vs 7.4 on this host), relaunched=1, redispatched=4 |
| sep2-chaos-p3b-on-1 | **p3b@e43e784c** ON (TTL 60 + max_inflight_reissues 3 + relaunch; contains p3a@7f43e43e) | yes | **completed** rc=0 | 6 | 2.8 s | step 5: 13.4 s (vs 3.9), relaunched=1, redispatched=4, inflight_reissued=0 |
| sep2-chaos-p3b-on-2 | p3b@e43e784c ON | yes | **completed** rc=0 | 6 | 2.9 s | step 5: 14.3 s (vs 4.3), relaunched=1, redispatched=4 |
| **sep2-chaos-p3b-on-final** | **p3b@f6846af0** ON (3 commits, FINAL: adds the actor-error dispatch-retry path - no same-chunk re-dispatch, orphaned uids go through the fresh-uid re-issue) | yes | **completed** rc=0, wall 273 s | 6 | 2.6 s | step 5: 6.1 s (`dead_workers_relaunched=1`, `prompts_orphaned_by_dead_workers=4`, `prompts_orphaned_at_dispatch=4`), step 6: 10.8 s (`inflight_reissued=8`, `inflight_dead_worker_reissued=8`); steady state 3.9-4.2 s |
| sep2-chaos-p3b-off-1 | p3b@e43e784c OFF | yes | **crashed** rc=1 (same `ActorDiedError` path as main) | 4 | 2.7 s | - |
| sep2-chaos-p3a-ttl-1 | **p3a@7f43e43e** TTL only (`inflight_ttl_s=60`, no relaunch knob on that tree) | yes | **crashed** rc=1 | 4 | 2.8 s | the next dispatch hits the dead handle ~7 s after the kill, before any 60 s TTL can re-issue: the TTL alone cannot save this failure mode (it covers prompts stuck on a live-but-silent worker) |
| sep2-main-1 / -2 | main@2ab35292 | no | completed | 6 | - | train sum 3.09 / 3.26 s, step 3.78 / 3.98 s |
| sep2-p3-on-1 / -2 | p3@5e6f6b74 ON | no | completed | 6 | - | train sum 3.41 / 3.11 s, step 4.13 / 3.79 s (no regression within spread) |
| sep2-p3b-on-1 | p3b@e43e784c ON | no | completed | 6 | - | train sum 3.54 s, step 4.17 s (main 3.78-3.98, p3 4.13 / 3.79: inside the 2-GPU spread) |

Score: main 0/3 survive the kill, knobs OFF 0/2 (p3, p3b), TTL only 0/1, TTL + relaunch 4/4 (p3 x2, p3b x2).

### GPU unit tests per branch (`gpu-tests`, main reference 503 passed / 0 failed / 8 skipped, torchrun 2/2)

| run | tree | pytest | torchrun | new failures |
|-----|------|--------|----------|--------------|
| gpu-tests-p1-1 | p1@4bcbbeb9 | 503 / 0 / 8, 26 min 41 s | 2/2 rc=0 | none |
| gpu-tests-p2-1 | p2@1e45a508 (5 commits) | 503 / 0 / 8, 25 min 23 s (rc=0) | 2/2 rc=0 | none |
| gpu-tests-p2-2 | p2@5cb89c52 (9 commits; final HEAD 2b259b2d is validation/docs only) | 503 / 0 / 8, 25 min 18 s (rc=0) | 2/2 rc=0 | none |
| gpu-tests-p1-2 | p1@b32c777e (9 commits; final HEAD a11ca00b is one docs/test commit later) | 503 / 0 / 8, 33 min 6 s (rc=0) | 2/2 rc=0 | none |
| gpu-tests-p4b-1 | p4b@97a43c92 (7 commits, final; contains p4a@f4651857; combined p4 912f79fb = same tree) | 503 / 0 / 8, 26 min 37 s (rc=0) | 2/2 rc=0 | none |
| gpu-tests-p3b-1 | p3b@e43e784c (2 commits; contains p3a@7f43e43e) | 503 / 0 / 8, 33 min 20 s (rc=0) | 2/2 rc=0 | none |
| gpu-tests-p3b-2 | p3b@f6846af0 (3 commits, final; combined p3 f6366f48 = same tree) | 503 / 0 / 8, 26 min 57 s (rc=0) | 2/2 rc=0 | none |
| gpu-tests-p0-1 | p0@6684917e (4 commits) | 503 / 0 / 8, 26 min 32 s (rc=0) | 2/2 rc=0 | none |
| gpu-tests-p3-1 | p3@5e6f6b74 (10 commits, pre-split) | 503 / 0 / 8, 33 min 18 s (rc=0) | 2/2 rc=0 | none |
| gpu-tests-p4-1 | p4@f9d52c4c (9 commits, pre-split; final p4 = 912f79fb = p4b tree) | 503 / 0 / 8, 32 min 42 s (rc=0) | 2/2 rc=0 | none |

### GPU unit tests re-run with the models linked (results/gpu-tests-main-2/, app <modal-app>, 8xH100, 2026-10-29 02:40Z)

Same invocation plus `/root/models/<org>/<name>` links to the hf-cache snapshots (meta `models_linked`):
**511 selected: 503 passed, 0 failed, 8 skipped, 24 min 59 s** (rc=0; the junit file counts 529 entries
because of per-test reruns/parametrisation bookkeeping, pytest's own line is authoritative), plus
torchrun `test_special_linear_cross_entropy_tp` rc=0 and `test_special_megatron_kl_loss_tp` rc=0.
This is the reference for branch test runs: a branch must show 0 failed here, or its failures must be explained.

## Status log, review round 2 (UTC, 2026-10-29)

- 13:05 Review round 2 starts (research/review-bundle/REVIEW.md: P4 needs an ENABLED end-to-end GPU run, P1 a frozen-batch
  identity test). Harness additions: `gsm8k-tool` preset (P4), `build-tool-dataset`, `--ship-extras` (bench/gsm8k_tool/ ->
  `<src>/bench_extras/`: the Gsm8kTool verl removed in 5a506cc5, its tool config, the custom reward function),
  `{run_dir}` placeholder in overrides, rollout/validation dump aggregation in summary.json, turn-level tables in summarize.py.
- 13:16 build-gsm8k-tool-1 (<modal-app>, CPU, main@2ab35292): `examples/data_preprocess/gsm8k_tool_agent_loop.py`
  -> /hf-cache/data/gsm8k-tool.
- 13:17 detached worktree wt/bench-p4b-97a43c92 (p4b reviewed head) created for the smoke; authors' wt/* never shipped.

## `gsm8k-tool` preset (P4 enabled-feature run): how it is built

- Data: `build-tool-dataset` runs verl's own `examples/data_preprocess/gsm8k_tool_agent_loop.py` in the download container
  (`results/build-gsm8k-tool-1/build.json`: train 7473 / test 1319 rows, columns `data_source, agent_name, prompt, ability,
  reward_model, extra_info`, `agent_name=tool_agent`, `reward_model.ground_truth == extra_info.tools_kwargs.calc_gsm8k_reward.
  create_kwargs.ground_truth`, parquet sha256 `29ffdf74...` / `fa5a7e78...`) -> `/hf-cache/data/gsm8k-tool`.
- Tool: main@2ab35292 has no GSM8K tool any more (5a506cc5, #6302 removed `verl/tools/gsm8k_tool.py` and the
  `examples/sglang_multiturn` tool configs: "reference patterns now live in docs/git history"). `bench/gsm8k_tool/gsm8k_tool.py`
  is that class from `git show 5a506cc5^:verl/tools/gsm8k_tool.py` with the imports moved to the flat `verl.tools` layout and
  two config knobs (`step_reward: verl|correctness`, `done_when_correct`); the reference step reward is kept as the default:
  0.0 for a submission that scores higher than the previous one, else -0.05. `gsm8k_tool_config.yaml` is the reference tool
  config pointing at `bench_extras.gsm8k_tool.Gsm8kTool`.
- Reward: `bench/gsm8k_tool/gsm8k_tool_reward.py` is the P4 author's `packages/p4/bench/gsm8k_tool_reward.py` (sha256
  9fd54018...; verl's strict `#### <answer>` GSM8K rule + diagnostics `turn_count`, `turn_rewards_sum`, `episode_done`,
  `spans_ok`, `metadata_present`, `num_turns` read from `extra_info`). smoke-gsm8k-tool-p4b-on-1 ran the bench placeholder
  (`.placeholder_gsm8k_tool_reward.py`, same score rule, extra `tool_calls` / `n_spans` keys) before the author's file landed.
- Shipping: `--ship-extras gsm8k_tool` (the preset's default) copies `bench/gsm8k_tool/*` into `<src>/bench_extras/`; the
  files are untracked, so the manifest ships them and `_exact_tree()` keeps them (`results/<run>/extras.json` has their
  sha256s), `src_git` therefore reads `HEAD@<sha>-dirty` for a tree that is exactly `<sha>` + `bench_extras/`. Ray puts the
  driver's cwd (`/workspace/verl`) on every worker's `sys.path`, which is what makes `bench_extras.gsm8k_tool` importable in
  the agent-loop workers. Refused for anything but a `wt/bench-*` worktree (never a author's worktree).
- Diagnostics path: under V1, `trainer.rollout_data_dir` dumps `input/output/gts/score/uid` only (`_log_rollout_data`
  passes no reward extra info), so the custom reward's diagnostics are visible through validation only
  (`val-aux/openai/gsm8k/<key>/mean@1` in metrics.jsonl and `val/<step>.jsonl` rows): the runs set `trainer.test_freq` for
  that. The dense turn rewards themselves are visible on the training side as `critic/rewards/min = -0.05` (the tool's
  penalty landed in `rm_scores`; the outcome-only writer cannot produce a negative minimum) and in `num_turns/*`.

## Review round 2: P4 enabled smoke (`gsm8k-tool`, p4b@97a43c92 + bench_extras, 2x H100, 2 steps, bs 16, n 4, mini 16)

Three runs, all rc=0 (`meta.json`: `src_head` 97a43c92, `tree.digest_match=True`, `extras.json` with the shipped files):

| run | arm | wall | num_turns mean / max (step 1, 2) | critic/rewards min | advantages min / max | pg_loss | grad_norm | val acc@1 (1319) | val metadata_present / spans_ok / turn_count / turn_rewards_sum |
|-----|-----|------|----------------------------------|--------------------|----------------------|---------|-----------|------------------|------------------------------------------------------------------|
| smoke-gsm8k-tool-p4b-on-1 (<modal-app>) | ENABLED (`turn_level_rewards=True`, `turn_rloo`), bench placeholder reward, no validation | 329.8 s | 2.75 / 4, 2.94 / 6 | **-0.05** (step 1) | -1.0 / 1.0 | 0.030, -0.006 | 0.944, 0.933 | - | - |
| smoke-gsm8k-tool-p4b-on-2 (<modal-app>) | ENABLED, author's reward file, `test_freq=2` | 311.2 s | 2.94 / 4, 3.25 / 6 | 0.0, **-0.10** (step 2: two penalties on one trajectory) | -1.05 / 1.05 | 0.037, 0.014 | 0.910, 0.810 | **0.6088** | **1.0 / 1.0 / 1.453 / -0.0010** (episode_done 0.0: the tool never sets `done`) |
| smoke-gsm8k-tool-p4b-off-1 (<modal-app>) | CONTROL (`grpo`, flag off), same reward file, `test_freq=2` | 486.2 s | 2.72 / 6, 3.13 / 6 | 0.0, 0.0 | -1.5 / 1.5 (GRPO std-normalised) | 0.100, 0.019 | 1.463, 1.272 | 0.6187 | **0.0 / 0.0 / 0.0 / 0.0** |

What the stages prove (the review's list): generation + tool execution = `num_turns/mean` 2.7-3.3 with max 6 (every trajectory
with `num_turns >= 4` made at least one `calc_gsm8k_reward` call; validation `turn_count` = 1.45 generated assistant turns
per trajectory); the reward callback saw the P4 metadata = validation `metadata_present` 1.0 and `spans_ok` 1.0 on 1319
trajectories under the flag and exactly 0.0 with the flag off; the dense row reached the trainer through TransferQueue =
`critic/rewards/min` -0.05 / -0.10 (the tool's non-improvement penalty on a turn's last generated token; the outcome-only
writer cannot produce a negative minimum, CONTROL shows 0.0); `turn_rloo` = advantages in [-1.05, 1.05] with mean -0.014
(the LOO return-to-go, unnormalised; GRPO's are +-1.5) and `timing_s/adv` 0.06 s; actor update = `actor/pg_loss` and
`actor/grad_norm` 0.81-0.94 on both steps. Shutdown traceback = the known DataLoader atexit noise (rc=0). Clip ratio
0.09-0.23 at `max_response_length=1024` (Qwen3 thinking is on; identical in both arms). No `RuntimeWarning` from the
dense writer and no tokenization-sanity warning in either train.log.

Knobs settled during the smoke: `trainer.v1.trainer_mode=sync` (V1 forces TransferQueue on), `max_assistant_turns=3`,
`format=hermes`, `tokenization_sanity_check_mode` left `strict`, thinking left on, `trainer.test_freq` set per run because
V1's training dumps carry no reward extra info (diagnostics are validation metrics), the tool/config/reward shipped as
`bench_extras/` (see the preset section). Nothing in verl or in the branch was patched.
- 13:18 smoke-gsm8k-tool-p4b-on-1 (<modal-app>, H100:2): rc=0, 2 steps, num_turns 2.75/2.94, critic/rewards/min -0.05; diagnostics not in the training dumps (V1 dumps uid only) -> validation added.
- 13:27 smoke-gsm8k-tool-p4b-on-2 (<modal-app>) + smoke-gsm8k-tool-p4b-off-1 (<modal-app>), test_freq=2, author's reward file: both rc=0; ENABLED val metadata_present/spans_ok 1.0, acc 0.609; CONTROL 0.0/0.0, acc 0.619. SMOKE PASSES.
- 13:40 lead: P1 final d7d820aa -> wt/bench-p1: gpu-identity-p1-1 (<modal-app>, H100:8) + gpu-tests-p1-r2 (<modal-app>); gpu-tests-p0-r2 / p2-r2 launched on e0a7eb22 / ba60cf8e then CANCELLED in the queue (<modal-app>, <modal-app>) on the lead's correction; both worktrees removed.
- 13:52 H100:8 queue slow ('waiting to be scheduled on a GPU_H100 worker' for 15 min on the identity run).
- 13:45 lead: p3b 0d28f34f -> wt/bench-p3b: chaos-p3b-r2-1 (<modal-app>, H100:2) COMPLETED rc=0 6/6, kill of
  agent_loop_worker_0 0.1 s after the trigger, steps 3-5 4.2 / 4.4 / 4.1 s, step 5 `rollout/dead_workers_relaunched=1`,
  `prompts_orphaned_by_dead_workers=4`, `prompts_orphaned_at_dispatch=4`, step 6 14.0 s with `training/inflight_reissued=4`,
  `inflight_dead_worker_reissued=4`, `inflight_reissue_exhausted=0`; wall 293.7 s. chaos-p3b-r2-2 <modal-app> launched 13:55.
  gpu-tests-p3b-r2 (<modal-app>), gpu-tests-p1-r2 (<modal-app>) and gpu-tests-p0-r2 (<modal-app>,
  p0 final 9c99205c) withdrawn from the saturated H100:8 queue at 13:50 so the P4 arms go first; relaunched once those run.
- 13:46 gpu-identity-p1-2 (<modal-app>, H100:1, p1@d7d820aa): `tests/workers/test_response_only_lm_head_on_gpu.py`
  **6 failed / 0 passed in 235.8 s** (first GPU run of that test; NOT patched, log pulled to results/gpu-identity-p1-2/pytest.log):
  every variant trips the gradient check `relative Frobenius difference < 1e-2`, e.g. fsdp2/lora
  `grad/...layers.0.mlp.down_proj.lora_B.default.weight` 1.676e-02. The H100:8 copy gpu-identity-p1-1 (<modal-app>) never
  got a node and was withdrawn.
- 13:56 P4 author-final p4b 64f8f604 (contains p4a a600e810) -> wt/bench-p4b; ENABLED-RUN.md read: tool config now ships
  `step_reward: correctness` to BOTH arms (the flag-off arm ignores step rewards), `algorithm.gamma=1.0` explicit on ENABLED,
  validation at steps 0/3/6 (`val_before_train=True test_freq=3`). p4-enabled-r2-1 <modal-app>, p4-control-r2-1
  <modal-app> (H100:8, 6 steps). 14:00 curves p4-curve-enabled-s1 / p4-curve-control-s1 (30 steps, test_freq 5,
  data.seed=1 in both arms; rollout.seed 42 default) launched; p2 final 7f0c53f5 -> wt/bench-p2 (suite queued after the P4 arms).
- 14:05 lead: p1 gets one more commit (GPU-test tolerances redesigned) -> gpu-tests-p1-r2 stays withdrawn; the d7d820aa
  identity deltas are the H100:1 run `gpu-identity-p1-2` (the name the lead reserved for the re-run; the re-run on the new
  sha will be `gpu-identity-p1-3`). Measured on d7d820aa: forward log_probs / entropy bitwise identical in all six variants,
  sum_pi_squared max_rel 3.8e-6; gradients: first key over the 1e-2 relative-Frobenius tolerance = layer 0
  (fsdp plain/gc `_flat_param` 1.694e-02; fsdp2 plain/gc `input_layernorm.weight` 2.201e-02; lora `down_proj.lora_B` 1.676e-02).
- 14:08 H100:8 pool saturated (P4 arms waiting since 13:57); stock suites gpu-tests-{p0,p2,p3b,p4b}-r2 queued on H200:8 instead
  (p0 9c99205c, p2 7f0c53f5, p3b 0d28f34f, p4b 64f8f604) so they never compete with the P4 arms for H100 nodes.
- 14:12 chaos-p3b-r2-2 (<modal-app>) COMPLETED rc=0 6/6, kill +255.2 s, step 6 12.8 s with inflight_reissued=8; p3b@0d28f34f 2/2.
- 14:16 lead: p3b moves again -> chaos-p3b-r2-1/-2 labelled '0d28f34f, superseded' (both had completed), gpu-tests-p3b-r2 (<modal-app>) withdrawn. p4-control-r2-1 scheduled on H100:8 at ~14:14; p4-enabled-r2-1 still waiting -> H200:8 twin launched.
- 14:22 P4 arms scheduled: p4-control-r2-1 (H100:8) step 0 validation acc 0.6027 (metadata_present 0.0), steps 1-2 done;
  p4-enabled-r2-1 (H100:8) started 14:06; its H200:8 twin p4-enabled-r2-1h (<modal-app>) stopped at uv sync.
  Curves p4-curve-enabled-s1 / -control-s1 both started on H100:8 ~14:07. gpu-tests-p0-r2 running on H200:8; p2-r2 / p4b-r2 queued.
- 14:24 lead: P1 final 407082ec -> wt/bench-p1b: identity re-run as `gpu-identity-p1-3` (H100:1; the name
  gpu-identity-p1-2 is the d7d820aa run on the volume, whose logs are append-mode) + gpu-tests-p1-r2 (H200:8). d7d820aa identity
  result labelled "d7d820aa, superseded by 407082ec" (its 6 reds were at the layer-0 gradient rule, not only the LoRA/AdamW cases).
- 14:30 p4-control-r2-1 (<modal-app>) rc=0 wall 496.8 s: val acc 0.603 / 0.630 / 0.671 at steps 0/3/6, metadata_present 0.0,
  advantages +-2.475, step mean(2..6) 17.17 s. p4-enabled-r2-1 (<modal-app>) rc=0 wall 489.9 s: rows_with_turn_spans
  512 every step, returns/mean ~1.0, score/mean 1.11-1.34 (rewards/max 3.0), val acc 0.608 / 0.626 / 0.702, metadata_present 1.0,
  turn_count 1.47 -> 1.73, step mean(2..6) 17.14 s, timing_s/adv 0.32 vs 0.32. Both labelled "64f8f604, superseded if the sha moves".
- 14:32 gpu-identity-p1-3 (<modal-app>, p1 FINAL 407082ec, H100:1): 6 failed / 0 passed / 0 skipped in 246.9 s, the same
  layer-0 gradient values as on d7d820aa (1.694e-02 / 2.201e-02 / 1.676e-02 vs the 1e-2 relative-Frobenius bound); forward
  log_probs / entropy bitwise identical. Log pulled, nothing patched. gpu-tests-p1-r2 (<modal-app>, H200:8) running.
- 14:33 lead: p4b gets one more commit -> gpu-tests-p4b-r2 (<modal-app>) withdrawn; curves to run on the final sha.
- CLOCK NOTE: the status-log times above from "14:22" to "14:33" are ~15 min too late (the wall-clock was read from a
  wrong estimate); the launch registry carries the exact `date -u` of every launch/stop. The 6-step arms exited at 14:08:36Z
  (control) and 14:12:05Z (enabled); gpu-identity-p1-3 finished 14:12:50Z.
- 14:21 the two 64f8f604 curves were stopped at ~step 5 (enabled) / ~step 10 (control) of 30 (23 s/step, both healthy:
  "Training Progress 4/30" at 14:16:39Z and "9/30" at 14:16:48Z) because the lead's rule is curves on the final p4b
  sha only; the stop notes first written in the launch registry ("stalled") were wrong and were corrected; partial pulls kept.

## Review round 2: results so far (2026-10-29, image verlai/verl:uv.cu130.dev5; every run's tree in its meta.json, app ids in the launch registry)

### P4 enabled end-to-end run (`gsm8k-tool` preset, 8x H100, 6 steps, 64 x 8, mini 64, `step_reward: correctness`, validation at 0/3/6)

Tree p4b@64f8f604 (+ `bench_extras/`), labelled "64f8f604, superseded if the sha moves" (p4b gets one more small commit;
the pair is repeated on the final sha as `p4-enabled-r2-final` / `p4-control-r2-final`, and the 30-step curves + the stock
suite run on the final sha only).

| stage | metric | p4-enabled-r2-1 (<modal-app>) | p4-control-r2-1 (<modal-app>) |
|---|---|---|---|
| multi-turn generation | `num_turns/mean` steps 1..6 (max 6) | 3.00 -> 3.43 | 2.96 -> 3.18 |
| tool executed, per-turn rewards | val `turn_rewards_sum`, `turn_count` at steps 0/3/6 | 0.45 / 0.52 / 0.70; 1.47 / 1.55 / 1.73 | 0 / 0 / 0; 0 / 0 / 0 |
| reward callback got the metadata | val `metadata_present`, `spans_ok` | 1.0, 1.0 (x3) | 0.0, 0.0 (x3) |
| dense rm_scores row | `critic/score/mean` (rewards/max) | 1.11-1.34 (3.0) | 0.64-0.70 (1.0) |
| TQ -> span lift -> turn_rloo | `adv_estimator/rows_with_turn_spans` | 512 on every step | absent |
| turn_rloo ran | `critic/returns/mean` | 0.91-1.09 | -0.04..-0.02 |
| actor update | `critic/advantages` min/max, `actor/grad_norm` | -2.1 / 2.7, 0.43-0.62 | -2.5 / 2.5, 0.50-0.66 |
| outcome scorer | `val-core/openai/gsm8k/acc/mean@1` at 0/3/6 | 0.608 / 0.626 / 0.702 | 0.603 / 0.630 / 0.671 |
| cost | `timing_s/adv`, `timing_s/step` mean(2..6) | 0.32 s, 17.14 s | 0.32 s, 17.17 s |

Smoke on 97a43c92 (2x H100, 2 steps): same picture (bench/README.md "P4 enabled smoke"). Partial 64f8f604 curves (stopped
at ~step 5 / ~step 10): `results/p4-curve-{enabled,control}-s1/metrics.jsonl`.

### P3b chaos re-run (p3b@0d28f34f, "superseded": p3b moves again) - packages/p3b/bench/RESULTS.md
chaos-p3b-r2-1 / -2 (<modal-app> / <modal-app>): both completed rc=0 6/6, kill->next step 2.5 / 2.6 s,
recovery step 14.0 / 12.8 s, `dead_workers_relaunched=1`, orphaned 4+4, `inflight_reissued` 4 / 8, exhausted 0. Knobs-ON 7/7 cumulative.

### P1 GPU identity (tests/workers/test_response_only_lm_head_on_gpu.py, 1x H100) - packages/p1/bench/RESULTS.md
d7d820aa (`gpu-identity-p1-2`, superseded) and FINAL 407082ec (`gpu-identity-p1-3`): 6 failed / 0 passed / 0 skipped each;
forward log_probs / entropy bitwise identical in all six variants, sum_pi_squared max_abs 1.08e-7; the gradient check fails at
its first key (layer 0) with identical values on both trees: fsdp 1.694e-02, fsdp2 2.201e-02, lora 1.676e-02 vs 1e-2. Not patched.

### Stock GPU unit-test selection (regression coverage: excludes `*_on_cpu.py` and `tests/experimental`) - H200:8
| run | tree | pytest | torchrun | app |
|---|---|---|---|---|
| gpu-tests-p0-r2 | p0@9c99205c (final) | **503 / 0 / 8**, 30 min 22 s (rc=0) | 2/2 rc=0 (39.5 s, 34.0 s) | <modal-app> |
| gpu-tests-p2-r2 | p2@7f0c53f5 (final) | **503 / 0 / 8**, 30 min 26 s (rc=0) | 2/2 rc=0 (38.8 s, 33.4 s) | <modal-app> |
| gpu-tests-p1-r2 | p1@407082ec ("407082ec; the only later change is the GPU test file itself") | **503 passed / 6 failed / 8 skipped**, 35 min 10 s (rc=1): the 6 failures are exactly `tests/workers/test_response_only_lm_head_on_gpu.py` (the new identity test is a `*_on_gpu.py` file under tests/workers, so the stock selection collects it); every stock test passes | 2/2 rc=0 (38.8 s, 33.7 s) | <modal-app> |
| gpu-tests-p3b-r2 | p3b@3f3d2c24 (final) | (running) | | <modal-app> |
| gpu-tests-p4b-r2 | p4b@988a10bc (final) | (queued) | | <modal-app> |
- 14:26 lead: p3b FINAL 3f3d2c24 (contains p3a 6445f21b) -> wt/bench-p3b-final: chaos-p3b-final-1 / -2 (H100:2 each, knobs ON) and gpu-tests-p3b-r2 (H200:8) launched; wt/bench-p3b (0d28f34f) removed. Final-sha list: p0 9c99205c, p1 407082ec, p2 7f0c53f5, p3b 3f3d2c24; p4b pending.
- 14:3x p3b FINAL chaos: chaos-p3b-final-1 (<modal-app>) COMPLETED rc=0 6/6 (kill +250.4 s, kill->next 2.6 s, step 5 6.0 s
  relaunched=1 orphaned 4+4, step 6 10.0 s inflight_reissued=8 dead-worker 8 exhausted 0, wall 274.9 s).
- 14:34 p4b FINAL 988a10bc (contains p4a d8c4b925) -> wt/bench-p4b-final: p4-enabled-r2-final <modal-app>, p4-control-r2-final
  <modal-app>, p4-curve-enabled-s1 <modal-app>, p4-curve-control-s1 <modal-app> (H100:8; the
  64f8f604 partial curves are parked locally as results/p4-curve-*-s1-64f8f604-partial and their volume dirs removed so the final runs own
  the names), gpu-tests-p4b-r2 <modal-app> (H200:8). ENABLED-RUN.md arms unchanged for 988a10bc.
- 14:35 gpu-tests-p0-r2 (<modal-app>, p0@9c99205c, H200:8): pytest 503 passed / 0 failed / 8 skipped in 30:22 (rc=0).
- 14:40 chaos-p3b-final-2 (<modal-app>) COMPLETED rc=0 6/6 (kill +260.2 s, step 5 14.8 s: relaunched=1, orphaned 12+4, inflight_reissued=7); no ttl_suspended key emitted on 3f3d2c24. p3b final 2/2.
- 14:53 gpu-tests-p2-r2 (7f0c53f5) 503/0/8 + torchrun 2/2; gpu-tests-p1-r2 (407082ec) 503 passed / 6 failed (= the identity test file) / 8 skipped + torchrun 2/2. p1 FINAL 363470c1 -> wt/bench-p1c, gpu-identity-p1-4 (<modal-app>, H100:1). p4-enabled-r2-final exited rc=0 (val acc step 6 0.694).
- 14:57 p4-enabled-r2-final (<modal-app>, 988a10bc, H100:8) rc=0 wall 482 s: rows_with_turn_spans 512 x6, returns/mean 0.95-1.13, score 1.16-1.34, val acc 0.599 / 0.633 / 0.694 (0/3/6), metadata_present/spans_ok 1.0, turn_count 1.45 -> 1.73, clip 0.15-0.22, step mean(2..6) 17.4 s. p4-control-r2-final still queued on H100:8; curve-enabled at step 7; curve-control queued; p3b-r2 suite late, p4b-r2 early.

### FINAL shas (lead's list: p0 9c99205c, p1 363470c1, p2 7f0c53f5, p3b 3f3d2c24, p4b 988a10bc)

| run | what | result |
|---|---|---|
| p4-enabled-r2-final / p4-control-r2-final (<modal-app> / <modal-app>) | P4 6-step pair on 988a10bc, 8x H100 | both rc=0; ENABLED `rows_with_turn_spans` 512 x6, returns/mean 0.95-1.13, score 1.16-1.34, val metadata_present/spans_ok 1.0, acc 0.599 / 0.633 / 0.694 (0/3/6), step 17.4 s; CONTROL metadata 0.0, returns ~0, acc 0.608 / 0.627 / 0.684, step 17.8 s (packages/p4/bench/ENABLED-RUN.md) |
| p4-curve-enabled-s1 / p4-curve-control-s1 (<modal-app> / <modal-app>) | 30-step A/B, data.seed=1, val every 5 | acc 0.588 -> 0.826 vs 0.599 -> 0.832 (same curve within noise); ENABLED learns more tool calls (val turn_count 1.44 -> 2.60, num_turns 5.2 vs 3.8) at 20.6 vs 16.1 s/step |
| p4-curve-enabled-s2 / p4-curve-control-s2 (<modal-app> / <modal-app>) | same, data.seed=2 | acc 0.597 -> 0.810 vs 0.597 -> 0.824; turn_count 1.45 -> 2.70; 17.6 vs 15.7 s/step. **Pooled 2 seeds at steps 0/5/10/15/20/25/30: ENABLED 0.593 0.684 0.732 0.763 0.790 0.820 0.818 vs CONTROL 0.598 0.663 0.733 0.774 0.796 0.818 0.828** (every difference within +-2 points; packages/p4/bench/ENABLED-RUN.md) |
| chaos-p3b-final-1 / -2 (<modal-app> / <modal-app>) | p3b 3f3d2c24 chaos, knobs ON | completed 2/2 (kill->next 2.6 s; recovery steps 6.0 + 10.0 s and 14.8 s; relaunched 1, orphaned 4+4 / 12+4, inflight_reissued 8 / 7, exhausted 0; no ttl_suspended key emitted) |
| gpu-identity-p1-4 (<modal-app>) | p1 363470c1 identity, H100:1 | 2 passed (lora x2) / 4 failed (plain, gc x fsdp, fsdp2): the fp32 arm's optimizer-update rule at 1e-4 trips by 1.00-1.29e-04 on 8-11 layers; forward quantities bitwise identical; bf16 grad deltas 1.7e-2 (layer 0) -> 2.2e-3 (layer 23) vs fp32 1.0-1.8e-5; `fp32 vs bf16 flag-off log-probs: mean=2.903e-02 max=4.309e-01` (packages/p1/bench/RESULTS.md) |
| gpu-identity-p1-5 (<modal-app>) | p1 FINAL cf539906 identity, H100:1 | **6 passed / 0 failed / 0 skipped, 196.6 s**: bf16 max D_on/D_off ratio 1.01 (fsdp), 1.22 (fsdp2, one layer-15 tensor), 1.10 (lora); fp32 on-vs-off max 1.8e-05 / 5.7e-05 / 3.8e-05; SGD residual bf16 0.213 / 0.292->0.245 / 0.000 ulp (packages/p1/bench/RESULTS.md) |
| gpu-tests-p1-r2-final (<modal-app>) | stock suite cf539906, H100:8 (the H200:8 copy <modal-app> never got a node) | console **509 passed / 0 failed / 8 skipped**, 27 min 19 s (rc=0) = 503 stock + 6 identity cases; junit 535 / 527 / 0 / 8; torchrun 2/2 rc=0 (32.6 s, 28.1 s) |
| gpu-tests-p0-r2 / p2-r2 | stock suites 9c99205c / 7f0c53f5, H200:8 | 503 / 0 / 8 + torchrun 2/2 each |
| gpu-tests-p1-r2 (407082ec) | stock suite | 503 passed / 6 failed (= the identity test file, collected by the stock glob) / 8 skipped + torchrun 2/2; "407082ec; the only later change is the GPU test file itself" |
| gpu-tests-p3b-r2-2 (<modal-app>) | stock suite 3f3d2c24, H200:8 | **503 / 0 / 8**, 30 min 36 s (rc=0) + torchrun 2/2 rc=0 (40.1 s, 34.4 s). The first attempt (<modal-app>, run-name gpu-tests-p3b-r2) was stopped ~25 min in by my own follow/stop chain (`;` instead of a guard) and has no summary; its volume dir is left as is |
| gpu-tests-p4b-r2 (<modal-app>) | stock suite 988a10bc, H200:8 | **503 / 0 / 8**, 30 min 56 s (rc=0) + torchrun 2/2 rc=0 (39.6 s, 34.7 s); `src_git` reads `-dirty` only because the shipped tree carries the untracked `bench_extras/` of the train launches |
- 15:05 final-sha results written (P4 pair + curves, p1-4 identity, p0/p2/p1 suites); s2 curves launched; p3b suite relaunched as gpu-tests-p3b-r2-2 after my premature stop of <modal-app>.
- 15:32 gpu-tests-p4b-r2 (988a10bc) 503/0/8 + torchrun 2/2. p4-curve-enabled-s2 rc=0 (wall 977 s); control-s2 near the end; gpu-tests-p3b-r2-2 in its last directories.
- 15:36 curves s2 done: ENABLED 0.810 vs CONTROL 0.824 at step 30 (pooled 2 seeds 0.818 vs 0.828: same curve); gpu-tests-p3b-r2-2 pytest 503/0/8 (30:35), torchrun pending.
- 15:40 gpu-tests-p3b-r2-2 (3f3d2c24) 503/0/8 + torchrun 2/2. ROUND 2 COMPLETE: every final sha measured (p0 9c99205c, p1 363470c1, p2
  7f0c53f5, p3b 3f3d2c24, p4b 988a10bc); every app stopped by id; detached worktrees removed; results under bench/results/
  (570 MB). Open items for the authors: P1 identity test 4/6 red on 363470c1 (optimizer-update tolerances), P1 stock suite
  collects that test file (6 red in gpu-tests-p1-r2 on 407082ec).
- 15:47 lead: P1 FINAL cf539906 -> wt/bench-p1d: gpu-identity-p1-5 (H100:1) + gpu-tests-p1-r2-final (H200:8) launched.
- 15:52 gpu-identity-p1-5 (cf539906, H100:1): 6 passed in 196.6 s. gpu-tests-p1-r2-final moved from H200:8 (unscheduled 12 min) to H100:8.
- 16:24 gpu-tests-p1-r2-final (cf539906, H100:8): console 509 passed / 8 skipped, junit 535/527/0/8, torchrun 2/2. ALL ROUND-2 RUNS DONE; final P1 identity 6/6 passed on cf539906.

## Review round 3 (2026-10-29, the AI follow-up review: research/review-bundle/follow-up-review/REVIEW.md "P4 benchmark")

Finding: round 2's P4 learning experiment paid +1 for EVERY correct submission (`step_reward: correctness`,
`done_when_correct: false`), so the enabled arms learned to resubmit the same correct answer (923 / 1085 of 1319 validation
episodes repeated an identical answer, 901 / 1049 earned >= 2 step points; controls 2-3 / 0), and the comparison changed reward
and estimator at once. `bench/analyze_runs.py` is the reviewer's audit script (only change: run names / validation step on
the command line); re-run on the round-2 curves it reproduces the audited counts.

Redesign (bench/gsm8k_tool/): `step_reward: first_correct` (+1.0 only for the first correct submission of an episode, 0.0 for
every other call) and `done_when_correct: true` (a correct submission ends the episode through `ToolResponse.done`; the loop
appends the tool response as the final observation and stops) shipped in ONE yaml to every arm, so the arms share the
environment and differ only in reward density and estimator; the round-2 yaml is frozen as
`gsm8k_tool_config_v1_correctness.yaml`. Outcome scorer for all arms: the P4 author's v2 reward function (correctness of
the last submitted answer parsed from the text; until it lands the round-2 scorer runs and reads 0 on `done` episodes).

Arms on the final p4b sha (30 steps, `test_freq=5`, `val_before_train=True`, `data.seed` 1 and 2, 8x H100; run names
`p4-v2-{A,B,C}-s{1,2}`): A = outcome-only GRPO control (flag off, `adv_estimator=grpo`); B = dense first-correct reward +
`turn_level_rewards=True` + `adv_estimator=reinforce_plus_plus` (stock estimator consuming dense rewards); C = the same dense
reward + `adv_estimator=turn_rloo algorithm.gamma=1.0`. Per run: val acc at 0/5/.../30, mean step time excluding step 1,
val `turn_count`, and the reviewer's repeated-submission counts (parsed tool-call answers > distinct answers = repeated).

### Status log, round 3 (UTC)
- 17:29 tool mode first_correct + done_when_correct=true written and CPU-checked; bench/analyze_runs.py copied; wt/bench-p4b-988a created for the smoke.
- 17:36 smokes on 988a10bc (2x H100, 2 steps, bs 16 x n 4, test_freq 2): smoke-v2-C-1 <modal-app> (turn_rloo) and smoke-v2-B-1 <modal-app> (reinforce_plus_plus) shipped the round-2 scorer; the author's v2 scorer (packages/p4/bench/gsm8k_tool_reward.py sha256 e127d432..., outcome = correctness of the last submitted tool-call answer, fallback to the strict text rule, diagnostics submissions / distinct_submissions / repeated_submission / scored_by_submission / final_text_correct) adopted into bench/gsm8k_tool/ and the pair relaunched as smoke-v2-C-2 <modal-app> / smoke-v2-B-2 <modal-app>.
- SMOKE OF THE NEW TOOL MODE PASSES (988a10bc, 2x H100, 2 steps, bs 16 x n 4): smoke-v2-C-1 (turn_rloo, rc=0, 303 s) and
  smoke-v2-B-1 (reinforce_plus_plus, rc=0, 335 s), round-2 scorer: validation (1319) `episode_done` 0.451 / 0.412 =
  `turn_rewards_sum` 0.451 / 0.412 (one point, only when solved), `spans_ok` 1.0, `metadata_present` 1.0, `turn_count` 1.02,
  reviewer's audit: parsed tool calls 626 / 577, **repeated identical answer 0 / 0**, step reward >= 2: 0 / 0; train
  `critic/rewards/max` 2.0 (outcome + step point on the same last generated token), C `rows_with_turn_spans` 64/64, B has no
  span metric (reinforce_plus_plus takes no spans), grad_norm 0.9-1.0 (C) / 1.9 (B, whitened advantages). The old strict
  scorer reads acc 0.246 / 0.297 because a `done` episode has no `#### ` text: the v2 scorer (C-2 / B-2) fixes exactly that.
- v2-SCORER SMOKE PASSES: smoke-v2-C-2 (<modal-app>, rc=0, 288 s) and smoke-v2-B-2 (<modal-app>, rc=0, 297 s)
  with the author's v2 reward function: val acc 0.674 / 0.663 (`scored_by_submission` 0.470 / 0.466, `final_text_correct`
  0.262 / 0.265 for the fallback rows), `submissions` = `distinct_submissions` 0.470 / 0.468 vs 0.466, `repeated_submission`
  0.000 / 0.0015, `episode_done` = `turn_rewards_sum` 0.445 / 0.439, `spans_ok` + `metadata_present` 1.0, `turn_count` 1.02;
  analyze_runs: repeated 0 / 2 of 1319, step reward >= 2: 0 / 0; independent manifest digest matches (tree + current extras).
  Train: C `rows_with_turn_spans` 64/64, returns/mean ~1.0, advantages [-2.0, 2.0]; B returns/mean ~1.0, whitened
  advantages [-1.2, 1.06]; `critic/rewards/max` 2.0 in both (outcome + step point on the same token).
- gpu-tests-p2-r3 (<modal-app>, p2 round-3 5532f3a4, H200:8) queued after the smoke (low priority, separate pool
  from the arms). p1 round-3 e7c0d616 -> wt/bench-p1e: gpu-identity-p1-6 (H100:2) + gpu-tests-p1-r3 (H200:8) launched.

### Round-3 arm commands (run on the final p4b sha in a detached `wt/bench-p4b-r3`; `<s>` = 1, 2)

```
cd <repo>; PY=.venv-bench/bin/python; SRC=<repo>/wt/bench-p4b-r3
COMMON="trainer.val_before_train=True trainer.test_freq=5 data.seed=<s>"
A="algorithm.adv_estimator=grpo $COMMON"                                                                         # outcome-only control, flag off
B="actor_rollout_ref.rollout.multi_turn.turn_level_rewards=True algorithm.adv_estimator=reinforce_plus_plus $COMMON"   # dense first-correct + stock estimator
C="actor_rollout_ref.rollout.multi_turn.turn_level_rewards=True algorithm.adv_estimator=turn_rloo algorithm.gamma=1.0 $COMMON"
$PY bench/verl_bench.py train --preset gsm8k-tool --src $SRC --run-name p4-v2-A-s<s> --total-training-steps 30 --extra-overrides "$A"
$PY bench/verl_bench.py train --preset gsm8k-tool --src $SRC --run-name p4-v2-B-s<s> --total-training-steps 30 --extra-overrides "$B"
$PY bench/verl_bench.py train --preset gsm8k-tool --src $SRC --run-name p4-v2-C-s<s> --total-training-steps 30 --extra-overrides "$C"
# afterwards: python3 -I bench/analyze_runs.py --tree wt/bench-p4b-r3 p4-v2-A-s1 p4-v2-B-s1 p4-v2-C-s1 p4-v2-A-s2 p4-v2-B-s2 p4-v2-C-s2
```
The shipped extras are the same for every arm (`gsm8k_tool_config.yaml`: `step_reward: first_correct`, `done_when_correct: true`;
the v2 scorer), so the arms differ only in the two overrides above.
- 17:48 lead: hold on p2 -> gpu-tests-p2-r3 (5532f3a4, queued on H200:8) labelled superseded, left to finish. gpu-identity-p1-6 (e7c0d616, H100:2): 8 passed / 2 failed in 323.6 s (details in packages/p1/bench/RESULTS.md). P4 round-3 e1052882 -> wt/bench-p4b-r3, smoke-v2-C-3 <modal-app> launched; arms follow the smoke.
- 17:53 smoke-v2-C-3 on e1052882 (<modal-app>) PASSES: rc=0, val acc 0.6755, episode_done = turn_rewards_sum 0.447, spans_ok/metadata 1.0, repeated_submission 0.0008 (1 of 1319), rows_with_turn_spans 64/64, manifest digest matches. Six arms p4-v2-{A,B,C}-s{1,2} launched on wt/bench-p4b-r3 (H100:8) + gpu-tests-p4b-r3 (H200:8). p2 final e19e7949 suite relaunched as gpu-tests-p2-r3-final (<modal-app>); gpu-tests-p1-r3 held; p3b 45690b0f worktree ready (chaos + suite at low priority).
- 18:00 p3b round-3 45690b0f: chaos-p3b-r3-1 <modal-app> (H100:2, knobs ON; TTL budget path changed, dead-worker path re-expressed) + gpu-tests-p3b-r3 <modal-app> (H200:8). p1 FINAL 1dbc0c07 -> wt/bench-p1f: gpu-identity-p1-7 (H100:2) + gpu-tests-p1-r3 (H200:8); wt/bench-p1e removed. Arms: C-s1 at step 1, A-s2 started, the rest queued on H100:8.
- 18:00 lead: p3b moves once more -> chaos-p3b-r3-1 and gpu-tests-p3b-r3 on 45690b0f labelled '45690b0f, superseded'; no new p3b runs until the final sha.
- 18:04 chaos-p3b-r3-1 (45690b0f, superseded) COMPLETED rc=0 6/6: kill +266.3 s, kill->next 2.7 s, step 6 13.8 s with inflight_reissued=4 (dead-worker 4), relaunched=1, orphaned 4+4; no TTL-budget key emitted.
- 18:07 gpu-identity-p1-7 (p1 FINAL 1dbc0c07, <modal-app>, H100:2): 10 failed / 0 passed in 255 s; log pulled, nothing patched; details in packages/p1/bench/RESULTS.md. gpu-tests-p3b-r3 (45690b0f, superseded, 0 tasks, no results dir) withdrawn from the H200 queue.
- 18:13 p4-v2-C-s1 (<modal-app>) rc=0 wall 878.8 s: val acc 0.667 0.753 0.790 0.811 0.820 0.830 0.841 (steps 0..30); turn_count 1.02 -> 1.08; episode_done = turn_rewards_sum 0.42 -> 0.84; submissions 0.44 -> 0.93 (distinct 0.91); repeated_submission 0.002 -> 0.014 (audit: 18 of 1319 at step 30, step reward >= 2: 0); rows_with_turn_spans 512 x30; critic/returns/mean 1.42; mean step 15.73 s; manifest digest matches.
- 18:17 p4-v2-A-s2 (<modal-app>) rc=0 wall 864.1 s: val acc 0.672 0.737 0.789 0.814 0.814 0.826 0.830; metadata/turn_count/episode_done 0 (flag off, as designed; episodes still end on done); submissions 0.45 -> 0.89 (distinct 0.89); repeated_submission 0.0008 -> 0.003 (audit 4 of 1319); critic/returns/mean -0.05; mean step 15.16 s.
- 18:27 p3b round-3 64212367 (contains p3a e598929b) -> wt/bench-p3b-r3b: chaos-p3b-r3-1 (H100:2, knobs ON; the 45690b0f result parked locally as chaos-p3b-r3-1-45690b0f-superseded and its volume dir removed) + gpu-tests-p3b-r3 (H200:8) launched; wt/bench-p3b-r3 removed.
- 18:33 p4-v2-B-s2 (<modal-app>) rc=0 wall 923.4 s: val acc 0.668 0.750 0.785 0.803 0.809 0.820 0.821; turn_count 1.02 -> 1.06; episode_done = turn_rewards_sum 0.43 -> 0.82; submissions 0.45 -> 0.88; repeated_submission 0 -> 0.003 (audit 4 of 1319); no span metric (reinforce_plus_plus); critic/returns/mean 1.39; mean step 16.15 s. gpu-tests-p4b-r3 (e1052882) pytest 503/0/8 in 30:45.
- 18:36 p4-v2-A-s1 (<modal-app>) rc=0 wall 911.9 s: val acc 0.664 0.725 0.792 0.807 0.816 0.823 0.835; submissions 0.44 -> 0.90; repeated 0 -> 0.0015 (audit 2); mean step 15.84 s. p4-v2-C-s2 (<modal-app>) rc=0 wall 936.9 s: val acc 0.657 0.762 0.787 0.792 0.804 0.826 0.818; turn_count 1.02 -> 1.06; episode_done = turn_rewards_sum 0.41 -> 0.82; repeated 0.0008 -> 0.003 (audit 4); rows_with_turn_spans 512 x30; returns/mean 1.40; mean step 16.52 s. gpu-tests-p4b-r3 (<modal-app>): 503/0/8 + torchrun 2/2 rc=0 (40.4 s, 34.0 s), junit 529/521/0/8.
- 18:39 p1 8c340e47 -> wt/bench-p1g: gpu-identity-p1-8 (H100:2) + gpu-tests-p1-r3-final (H200:8; the name gpu-tests-p1-r3 is the running 1dbc0c07 suite, now labelled superseded). p3b 64212367 chaos COMPLETED 6/6 (kill->next 2.7 s, step 6 14.4 s, reissued 4) but p3b moves once more: both 64212367 runs labelled superseded.

### Round-3 arms (p4b e1052882): results

| run (app) | arm | rc / wall s | val acc at steps 0 / 5 / 10 / 15 / 20 / 25 / 30 | mean step s (excl. 1) | val turn_count 0 -> 30 | val num_turns 0 -> 30 | repeated_submission (metric, step 30) | submissions / distinct (30) | episode_done = turn_rewards_sum (30) | critic/returns/mean (2..30) | rows_with_turn_spans |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p4-v2-A-s1 (<modal-app>) | A: outcome-only GRPO (flag off) | 0 / 912 | 0.6641 / 0.7248 / 0.7923 / 0.8074 / 0.8158 / 0.8234 / 0.8347 | 15.84 | - | 2.45 -> 2.96 | 0.0015 | 0.901 / 0.900 | - (flag off: not exported; episodes still end on done) | -0.050 | absent |
| p4-v2-A-s2 (<modal-app>) | A: outcome-only GRPO (flag off) | 0 / 864 | 0.6717 / 0.7369 / 0.7892 / 0.8143 / 0.8135 / 0.8256 / 0.8302 | 15.16 | - | 2.47 -> 2.95 | 0.0030 | 0.892 / 0.889 | - (flag off: not exported; episodes still end on done) | -0.046 | absent |
| p4-v2-B-s1 (<modal-app>) | B: first-correct dense + reinforce_plus_plus | 0 / 1152 | 0.6588 / 0.7544 / 0.8059 / 0.8082 / 0.8150 / 0.8165 / 0.8234 | 23.67 | 1.023 -> 1.070 | 2.47 -> 2.96 | 0.0121 | 0.901 / 0.886 | 0.8226 = 0.8226 | 1.415 | absent |
| p4-v2-B-s2 (<modal-app>) | B: first-correct dense + reinforce_plus_plus | 0 / 923 | 0.6679 / 0.7498 / 0.7847 / 0.8029 / 0.8089 / 0.8196 / 0.8211 | 16.15 | 1.019 -> 1.057 | 2.46 -> 2.93 | 0.0030 | 0.880 / 0.876 | 0.8211 = 0.8211 | 1.386 | absent |
| p4-v2-C-s1 (<modal-app>) | C: first-correct dense + turn_rloo | 0 / 879 | 0.6672 / 0.7528 / 0.7900 / 0.8112 / 0.8196 / 0.8302 / 0.8408 | 15.73 | 1.022 -> 1.084 | 2.46 -> 3.01 | 0.0136 | 0.931 / 0.914 | 0.8408 = 0.8408 | 1.416 | 512 x30 |
| p4-v2-C-s2 (<modal-app>) | C: first-correct dense + turn_rloo | 0 / 937 | 0.6573 / 0.7619 / 0.7870 / 0.7923 / 0.8044 / 0.8264 / 0.8180 | 16.52 | 1.023 -> 1.058 | 2.46 -> 2.93 | 0.0030 | 0.880 / 0.876 | 0.8180 = 0.8180 | 1.403 | 512 x30 |

Pooled over the two data seeds (mean of s1 and s2), validation accuracy on 1319 prompts:

| arm | 0 | 5 | 10 | 15 | 20 | 25 | 30 | mean step s |
|---|---|---|---|---|---|---|---|---|
| A outcome-only GRPO (flag off) | 0.6679 | 0.7309 | 0.7908 | 0.8108 | 0.8146 | 0.8245 | 0.8324 | 15.50 |
| B first-correct dense + reinforce_plus_plus | 0.6634 | 0.7521 | 0.7953 | 0.8055 | 0.8120 | 0.8180 | 0.8222 | 19.91 |
| C first-correct dense + turn_rloo | 0.6622 | 0.7574 | 0.7885 | 0.8017 | 0.8120 | 0.8283 | 0.8294 | 16.13 |
| B - A (points) | -0.5 | +2.1 | +0.5 | -0.5 | -0.3 | -0.6 | -1.0 | |
| C - A (points) | -0.6 | +2.7 | -0.2 | -0.9 | -0.3 | +0.4 | -0.3 | |
| C - B (points) | -0.1 | +0.5 | -0.7 | -0.4 | +0.0 | +1.0 | +0.7 | |

Audit and reading: packages/p4/bench/RESULTS.md "v2"; raw audit `bench/results/p4-v2-audit.json`.

### Round-3 stock suites and identity runs (regression coverage; identity runs separately)

| run | tree | result | app |
|---|---|---|---|
| gpu-tests-p4b-r3 | p4b e1052882 (final) | console **503 / 0 / 8** (30:45), junit 529 / 521 / 0 / 8, torchrun 2/2 rc=0 (40.4 s, 34.0 s), H200:8 | <modal-app> |
| gpu-tests-p2-r3-final | p2 e19e7949 (final) | console **503 / 0 / 8** (30:29), junit 529 / 521 / 0 / 8, torchrun 2/2 rc=0 (38.4 s, 35.2 s), H200:8 | <modal-app> |
| gpu-identity-p1-6 | p1 e7c0d616 (superseded) | 8 passed / 2 failed (the two new two-rank cases crash in the loss: padded 256 vs 40 tokens; predicted by the re-review), H100:2 | <modal-app> |
| gpu-identity-p1-7 | p1 1dbc0c07 (superseded) | 0 passed / 10 failed: every case at line 537 `baseline[key] - reference[key]` on a bool tensor, before any measurement | <modal-app> |
| gpu-tests-p1-r3 | p1 1dbc0c07 (superseded) | console 502 passed / 11 failed / 8 skipped (35:06): the 10 identity cases of that tree + `checkpoint_engine/test_sharded_delta::test_gather_slot_entries_sub_rounds_world1` (`EADDRINUSE` on port 29512, environmental; passes on every other suite); junit 539 / 520 / 11 / 8; torchrun 2/2 | <modal-app> |
| gpu-identity-p1-8 | p1 8c340e47 (final) | **10 passed / 0 failed / 0 skipped** in 320.5 s (H100:2): max distance ratio D_on/D_off 1.02 (fsdp) / 1.22 (fsdp2) / 1.11 (lora, lora-ckpt) / 1.01 and 1.13 (two-rank fsdp / fsdp2); two-rank bf16 forward quantities bitwise identical on 50 rows per rank, `sgd/residual_ulps` 0.217 (packages/p1/bench/RESULTS.md) | <modal-app> |
| gpu-tests-p1-r3-final | p1 8c340e47 (final) | console **513 passed / 0 failed / 8 skipped** (30:08) = 503 stock + 10 identity cases; junit 539 / 531 / 0 / 8; torchrun 2/2 rc=0 (31.6 s, 26.6 s), H200:8 | <modal-app> |
| chaos-p3b-r3-1 (45690b0f, superseded) | p3b 45690b0f | completed 6/6, kill->next 2.7 s, recovery step 13.8 s, relaunched 1, orphaned 4+4, inflight_reissued 4 | <modal-app> (local dir `chaos-p3b-r3-1-45690b0f-superseded`) |
| chaos-p3b-r3-1 (64212367, superseded) | p3b 64212367 | completed 6/6, kill->next 2.7 s, recovery step 14.4 s, relaunched 1, orphaned 4+4, inflight_reissued 4 | <modal-app> |
| gpu-tests-p3b-r3 (64212367, superseded) | p3b 64212367 | console 503 / 0 / 8 (30:36), junit 529 / 521 / 0 / 8, torchrun 2/2 rc=0 (local dir `gpu-tests-p3b-r3-64212367-superseded`) | <modal-app> |
| chaos-p3b-final3-1 | p3b 1b397dfd (final) | completed 6/6, kill->next 2.3 s, recovery steps 6.2 + 11.5 s, relaunched 1, orphaned 4+4, inflight_reissued 7, exhausted 0 (packages/p3b/bench/RESULTS.md) | <modal-app> |
| gpu-tests-p3b-r3-final | p3b 1b397dfd (final) | console **503 / 0 / 8** (30:25), junit 529 / 521 / 0 / 8, torchrun 2/2 rc=0 (39.0 s, 34.3 s), H200:8 | <modal-app> |
| gpu-tests-p3b-r3 (45690b0f) | withdrawn unrun (0 tasks) when superseded | | <modal-app> |
- 18:50 gpu-tests-p2-r3-final (e19e7949) 503/0/8 + torchrun 2/2; v2 arm tables written to packages/p4/bench/RESULTS.md and README.
- 18:51 p3b FINAL 1b397dfd (contains p3a 2bd5c040) -> wt/bench-p3b-final3: chaos-p3b-final3-1 (H100:2, knobs ON) + gpu-tests-p3b-r3-final (H200:8); the 64212367 runs stay labelled superseded. Final round-3 list: p0 9c99205c, p1 8c340e47, p2 e19e7949, p3b 1b397dfd, p4b e1052882.
- 18:56 gpu-tests-p1-r3 (1dbc0c07, superseded; <modal-app>, H200:8): console 502 passed / 11 failed / 8 skipped in 35:06 (junit 539/520/11/8), torchrun 2/2 rc=0: the 10 identity cases of that tree (bool-tensor bug, fixed in 8c340e47) plus tests/checkpoint_engine/test_sharded_delta::test_gather_slot_entries_sub_rounds_world1 with 'DistNetworkError: The server socket has failed to listen ... port: 29512 ... EADDRINUSE' (environmental port collision, not a product failure; the same test passed on every other suite today). gpu-identity-p1-8 (8c340e47): 10 passed in 320.5 s.
- 19:03 chaos-p3b-final3-1 (1b397dfd, <modal-app>) COMPLETED rc=0 6/6: kill +266.8 s, kill->next 2.3 s, step 5 6.2 s (relaunched 1, orphaned 4+4), step 6 11.5 s (inflight_reissued 7, dead-worker 7, exhausted 0).
- 19:23 gpu-tests-p1-r3-final (8c340e47) console 513/0/8, junit 539/531/0/8, torchrun 2/2. gpu-tests-p3b-r3 (64212367, superseded) 503/0/8 + torchrun 2/2. Only gpu-tests-p3b-r3-final (1b397dfd) still running.
- 19:35 gpu-tests-p3b-r3-final (1b397dfd) 503/0/8 + torchrun 2/2. ROUND 3 COMPLETE: p1 8c340e47 (identity-8 10/10 + suite 513/0/8), p2 e19e7949 (503/0/8), p3b 1b397dfd (chaos completed + 503/0/8), p4b e1052882 (six v2 arms + 503/0/8); every app stopped by id; detached worktrees removed.
