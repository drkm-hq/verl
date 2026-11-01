# P0: GPU unit tests on the branch (bench/README.md harness)

`gpu-tests-p0-1`: `--src <repo>/wt/p0` at `contrib/p0-loss-horizon@6684917e` (4 commits
ahead of main@2ab35292, clean tree; `meta.json` `src_head` / `tree.digest_match=True`), 8x H100 80GB, exactly the
`.github/workflows/gpu_unit_tests.yml` job: `uv run --frozen --all-packages --extra sglang --extra megatron --extra ci
pytest -s <the workflow's 13 --ignore-glob filters> tests/` (without CI's `-x`, so counts are complete) plus the two
torchrun tests, with `/root/models/<org>/<name>` linked to the hf-cache snapshots the tests expect.

| run | tree | pytest (passed / failed / skipped, wall) | torchrun linear_cross_entropy_tp | torchrun megatron_kl_loss_tp | new failures vs main |
|-----|------|------------------------------------------|----------------------------------|------------------------------|----------------------|
| gpu-tests-main-2 (reference) | main@2ab35292 | 503 / 0 / 8, 24 min 59 s | rc=0 | rc=0 | - |
| gpu-tests-p0-1 | p0@6684917e (4 commits) | **503 / 0 / 8, 26 min 32 s** (rc=0) | rc=0 (37 s) | rc=0 (32 s) | **none** |

Sha note: the arm ran at 6684917e (4 commits); the branch's final HEAD is 0dab0eb5 (5 commits, validation/test/docs
only), so the result stands for the final tree.

App <modal-app> (the launch registry (kept privately)). Artifacts: `bench/results/gpu-tests-p0-1/{pytest.log,pytest.xml,
summary.json,torchrun-*.log,command.txt}`; `summary.json["pytest"]["failed"]` lists any failing test ids.
