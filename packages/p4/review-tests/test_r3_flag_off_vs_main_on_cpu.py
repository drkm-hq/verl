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
"""R3: flag-off identity against MAIN, not against the branch's own flag-off path. ``episode_dump.py`` runs
the same scripted episode (3 turns, a GLM-style trailing-token removal, done on the last call) once with
PYTHONPATH=<main clone> and once with PYTHONPATH=<branch>, and this test compares the two rows."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))  # the repository root

HERE = Path(__file__).resolve().parent
MAIN = Path(os.path.join(_REPO, "verl"))
BRANCH = Path(os.path.join(_REPO, "wt/p4"))
NEW_KEYS = {"turn_rewards", "generated_turn_spans", "episode_done"}


def _dump(tree: Path, tmp_path: Path) -> dict:
    out = tmp_path / f"{tree.name}.pt"
    env = {**os.environ, "PYTHONPATH": str(tree), "TORCH_COMPILE_DISABLE": "1", "TORCHINDUCTOR_DISABLE": "1"}
    subprocess.run([sys.executable, "-W", "ignore", str(HERE / "episode_dump.py"), str(out)], check=True, env=env)
    row = torch.load(out, weights_only=False)
    assert row["verl_file"].startswith(str(tree)), row["verl_file"]
    return row


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rows")
    return _dump(MAIN, tmp), _dump(BRANCH, tmp)


def test_as_dict_tensors_are_bit_identical_to_main(rows):
    main, branch = rows
    for key in ("prompts", "responses", "response_mask", "rm_scores"):
        assert torch.equal(main[key], branch[key]), key
        assert main[key].dtype == branch[key].dtype, key
    assert main["num_turns"] == branch["num_turns"] and main["server_calls"] == branch["server_calls"]
    assert branch["rm_scores"].nonzero().flatten().tolist() == [len(branch["response_mask"]) - 1]


def test_as_dict_keys_and_extra_fields_are_identical_to_main_with_the_flag_off(rows):
    """After fix D1 (column half) the flag-off metadata is exactly main's: no new keys at all.
    (The pre-fix version of this test asserted the three keys were always exported.)"""
    main, branch = rows
    skip = {"metrics", "verl_file", "v0_batch", "v0_non_tensor_keys", "v0_meta", "server_calls"}
    assert set(main) - skip == set(branch) - skip
    assert set(branch["extra_fields"]) == set(main["extra_fields"])
    assert not NEW_KEYS & set(branch["extra_fields"])
    for key in main["extra_fields"]:
        assert main["extra_fields"][key] == branch["extra_fields"][key], key
    assert main["server_calls"] == branch["server_calls"] == 4


def test_v0_postprocess_row_is_bit_identical_to_main_including_non_tensor_keys(rows):
    main, branch = rows
    assert set(main["v0_batch"]) == set(branch["v0_batch"])
    for key, value in main["v0_batch"].items():
        assert torch.equal(value, branch["v0_batch"][key]), key
    assert branch["v0_non_tensor_keys"] == main["v0_non_tensor_keys"]
    assert main["v0_meta"] == branch["v0_meta"]
