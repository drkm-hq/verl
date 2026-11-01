# P4b (stacked on P4a): GPU unit tests on the branch (bench/README.md harness)

`gpu-tests-p4b-1`: `--src <repo>/wt/p4b` at `contrib/p4b-turn-rloo@d8c16498` (5 commits
ahead of main@2ab35292, contains p4a `contrib/p4a-turn-signals@71bd8b61`; clean tree; `meta.json` `src_head` /
`tree.digest_match=True`), 8x H100 80GB, exactly the `.github/workflows/gpu_unit_tests.yml` job:
`uv run --frozen --all-packages --extra sglang --extra megatron --extra ci pytest -s <the workflow's 13 --ignore-glob
filters> tests/` (without CI's `-x`, so counts are complete) plus the two torchrun tests, with `/root/models/<org>/<name>`
linked to the hf-cache snapshots the tests expect.

| run | tree | pytest (passed / failed / skipped, wall) | torchrun linear_cross_entropy_tp | torchrun megatron_kl_loss_tp | new failures vs main |
|-----|------|------------------------------------------|----------------------------------|------------------------------|----------------------|
| gpu-tests-main-2 (reference) | main@2ab35292 | 503 / 0 / 8, 24 min 59 s | rc=0 | rc=0 | - |
| gpu-tests-p4-1 (pre-split branch) | contrib/p4-turn-rewards@f9d52c4c (9 commits) | **503 / 0 / 8, 32 min 42 s** (rc=0) | (see bench/README.md status) | (see status) | **none** |
| gpu-tests-p4b-1 | p4b@97a43c92 (7 commits, final; contains p4a@f4651857; the combined p4 = 912f79fb is the same tree) | **503 / 0 / 8, 26 min 37 s** (rc=0) | rc=0 | rc=0 | **none** |

The first gpu-tests-p4b-1 launch (<modal-app>, p4b@d8c16498) never got a GPU and was stopped in the
queue when the final sha landed; the run above is the relaunch from 97a43c92.

Apps <modal-app> (p4-1), <modal-app> (p4b-1) (the launch registry (kept privately)). Artifacts:
`bench/results/<run>/{pytest.log,pytest.xml,summary.json,torchrun-*.log,command.txt}`; `summary.json["pytest"]["failed"]`
lists any failing test ids.
