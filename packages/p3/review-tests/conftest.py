# Adversarial review tests for branch contrib/p3-v1-never-hang (reviewer-owned, not part of the worktree).
# Run with:  cd wt/p3 && PYTHONPATH=$PWD .venv-cpu/bin/python -m pytest -c ../../pytest-cpu.ini \
#            --rootdir ../../packages/p3/review-tests -q --asyncio-mode=auto ../../packages/p3/review-tests
import pytest
import ray
import transfer_queue as tq


@pytest.fixture(scope="session")
def ray_tq():
    if not ray.is_initialized():
        ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True, logging_level="ERROR")
    tq.init()
    yield
    tq.close()
    ray.shutdown()
