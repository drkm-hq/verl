# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""R3a (the submitter's R2-04, fix 0d4cd8ca): the flag-OFF callback contract of every reward-loop manager, measured against
main @ 2ab35292 line by line by running ``r3_probe.py`` in a subprocess on each tree's PYTHONPATH (main: the clone at
verl/; branch: the tree that this test imports ``verl`` from). Compared per manager: the keys and values the
callback received, whether it received the dataset's own dict object, and the dataset dict after the call."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import verl
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, os.pardir))  # the repository root

ROOT = Path(_REPO)
MAIN = ROOT / "verl"
PROBE = Path(__file__).with_name("r3_probe.py")
TREE = Path(verl.__file__).resolve().parents[1]
MANAGERS = ["NaiveRewardManager", "DAPORewardManager", "GDPORewardManager", "RateLimitedRewardManager", "RemoteRewardManager"]
DATASET = {"num_turns": 7, "tool_rewards": [123.0], "rollout_reward_scores": {"dataset": 1}, "keep": "me"}


def run(tree: Path) -> dict:
    env = {**os.environ, "PYTHONPATH": str(tree), "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    out = subprocess.run([sys.executable, str(PROBE)], env=env, capture_output=True, text=True, timeout=300, cwd=str(PROBE.parent))
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def reports():
    assert TREE != MAIN.resolve(), "run with PYTHONPATH at a branch worktree, the clone is the main reference"
    return run(MAIN), run(TREE)


@pytest.mark.parametrize("name", MANAGERS)
def test_the_callback_receives_the_same_keys_and_values_as_on_main(reports, name):
    main, branch = reports[0][name], reports[1][name]
    assert main["error"] is None and branch["error"] is None, (main["error"], branch["error"])
    assert branch["received"] == main["received"], f"{name}: main {main['received']} vs branch {branch['received']}"


def test_main_reference_values_are_what_reviewer_measured(reports):
    main = reports[0]
    for name in ("DAPORewardManager", "GDPORewardManager"):
        assert main[name]["received"]["num_turns"] == 7 and main[name]["received"]["tool_rewards"] == [123.0]
        assert main[name]["received"]["rollout_reward_scores"] == {"dataset": 1}
    assert main["GDPORewardManager"]["received"]["experiment_name"] == "r3"
    limited = main["RateLimitedRewardManager"]["received"]
    assert limited["num_turns"] == 7 and limited["tool_rewards"] == [2.0] and limited["rollout_reward_scores"] == {"dataset": 1}
    for name in ("NaiveRewardManager", "RemoteRewardManager"):
        assert main[name]["received"]["num_turns"] == 3 and main[name]["received"]["tool_rewards"] == [2.0]
        assert main[name]["received"]["rollout_reward_scores"] == {"acc": 1.0}


@pytest.mark.parametrize("name", MANAGERS)
def test_object_identity_and_dataset_mutation_per_manager(reports, name):
    main, branch = reports[0][name], reports[1][name]
    if name in ("DAPORewardManager", "GDPORewardManager"):
        assert main["is_dataset_object"] and branch["is_dataset_object"], "the dataset's own object on both trees"
        assert branch["dataset_after"] == main["dataset_after"]  # gdpo's experiment_name lands on it on both
    else:
        # main merged into the dataset dict in place; since round 2 the branch hands over a copy. Unobservable in
        # the shipped flow (the reward worker scores a pickled DataProto: agent_loop.py compute_score.remote).
        assert main["is_dataset_object"] and not branch["is_dataset_object"]
        assert main["dataset_after"] == main["received"], "main leaked the merge into the dataset dict"
        assert branch["dataset_after"] == DATASET, "the branch leaves the dataset dict pristine"
