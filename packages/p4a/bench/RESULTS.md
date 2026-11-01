# P4a: GPU unit tests (run on the stacked p4b branch, which contains p4a) - bench/README.md harness

The GPU test arm for p4a is `gpu-tests-p4b-1` on `--src <repo>/wt/p4b` at
`contrib/p4b-turn-rloo@d8c16498` (5 commits ahead of main@2ab35292), which contains `contrib/p4a-turn-signals@71bd8b61`
(3 commits). Same job as every other branch: 8x H100 80GB, the `.github/workflows/gpu_unit_tests.yml` selection
(`uv run --frozen --all-packages --extra sglang --extra megatron --extra ci pytest -s <13 --ignore-glob filters> tests/`,
no `-x`) plus the two torchrun tests, models linked under `/root/models`.

| run | tree | pytest (passed / failed / skipped, wall) | torchrun linear_cross_entropy_tp | torchrun megatron_kl_loss_tp | new failures vs main |
|-----|------|------------------------------------------|----------------------------------|------------------------------|----------------------|
| gpu-tests-main-2 (reference) | main@2ab35292 | 503 / 0 / 8, 24 min 59 s | rc=0 | rc=0 | - |
| gpu-tests-p4-1 (pre-split combined branch) | contrib/p4-turn-rewards@f9d52c4c (9 commits) | **503 / 0 / 8, 32 min 42 s** (rc=0) | (see bench/README.md status) | (see status) | **none** |
| gpu-tests-p4b-1 | p4b@97a43c92 (7 commits, final; contains p4a@f4651857) | **503 / 0 / 8, 26 min 37 s** (rc=0) | rc=0 | rc=0 | **none** |

App <modal-app> (the launch registry (kept privately)); artifacts in `bench/results/gpu-tests-p4b-1/`.
