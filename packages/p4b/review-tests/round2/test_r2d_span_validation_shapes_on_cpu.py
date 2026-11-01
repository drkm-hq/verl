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
"""R2d (round-2 finding 3): ``_validated_turn_spans`` / ``compute_turn_rloo_advantage`` attacked with the shapes a
producer or a TransferQueue round trip can hand over: lists vs tuples, numpy / torch integers, 2-D int array rows,
a 3-D int array column, a plain-list column, a span equal to the whole row, a one-token span at the row end, mask
zeros inside a span, ragged None / [] / () rows next to span rows; the rejected shapes name the row; after
988a10bc (LOW 3 / 4) non-integral offsets and a non-sequence column are rejected by name too."""

import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import (
    _validated_turn_spans,
    compute_rloo_outcome_advantage,
    compute_turn_rloo_advantage,
)

L = 6
REWARDS = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 3.0, 0.0, 1.0, 0.0]])
MASK = torch.tensor([[1, 1, 0, 0, 1, 1], [1, 1, 1, 0, 1, 0]], dtype=torch.float32)
INDEX = np.array(["g", "g"], dtype=object)
CANON = [[(0, 2), (4, 6)], [(0, 3), (4, 5)]]


def column(rows):
    spans = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        spans[i] = row
    return spans


def run(spans):
    return compute_turn_rloo_advantage(REWARDS.clone(), MASK, INDEX, turn_spans=spans)


ACCEPTED = {
    "lists": [[[0, 2], [4, 6]], [[0, 3], [4, 5]]],
    "numpy-ints": [[(np.int64(0), np.int32(2)), (np.int64(4), np.int64(6))], [(np.int64(0), np.int64(3)), (np.int64(4), np.int64(5))]],
    "torch-ints": [[(torch.tensor(0), torch.tensor(2)), (torch.tensor(4), torch.tensor(6))], CANON[1]],
    "2d-array-rows": [np.array([[0, 2], [4, 6]]), np.array([[0, 3], [4, 5]])],
    "tuple-of-tuples": [tuple(CANON[0]), tuple(CANON[1])],
}


@pytest.mark.parametrize("name", list(ACCEPTED))
def test_equivalent_spellings_of_the_same_spans_give_the_canonical_advantages(name):
    expected, _ = run(column(CANON))
    got, _ = run(column(ACCEPTED[name]))
    torch.testing.assert_close(got, expected)


def test_3d_int_array_and_plain_list_columns_are_accepted():
    expected, _ = run(column(CANON))
    torch.testing.assert_close(run(np.array(CANON))[0], expected)  # a (bs, turns, 2) int array
    torch.testing.assert_close(run(CANON)[0], expected)  # a plain list column
    torch.testing.assert_close(run([tuple(row) for row in CANON])[0], expected)


def test_a_span_covering_the_whole_row_and_a_one_token_span_at_the_row_end_are_accepted():
    assert _validated_turn_spans(0, [(0, L)], L) == [(0, L)]
    assert _validated_turn_spans(0, [(0, 1), (L - 1, L)], L) == [(0, 1), (L - 1, L)]
    whole, _ = run(column([[(0, L)], [(0, L)]]))
    rloo, _ = compute_rloo_outcome_advantage(REWARDS.clone(), MASK, INDEX)
    torch.testing.assert_close(whole, rloo)


def test_mask_zeros_inside_a_span_get_zero_advantage_and_zero_return():
    adv, ret = run(column([[(0, L)], [(0, L)]]))
    assert torch.all(adv[MASK == 0] == 0) and torch.all(ret[MASK == 0] == 0)
    assert torch.all(adv[MASK == 1] != 0)


def test_ragged_none_empty_rows_are_one_whole_row_turn_next_to_span_rows():
    expected, _ = run(column([[(0, L)], CANON[1]]))
    for empty in (None, [], ()):
        torch.testing.assert_close(run(column([empty, CANON[1]]))[0], expected)


REJECTED = {
    "end-past-the-row": [(0, 2), (4, L + 1)],
    "overlap": [(0, 3), (2, 5)],
    "reversed-order": [(4, 6), (0, 2)],
    "empty-span": [(2, 2)],
    "start-after-end": [(3, 1)],
    "negative-start": [(-1, 2)],
    "triple": [(0, 1, 2)],
    "none-inside": [(None, 2)],
    "scalar-span": [5],
    "string-row": "ab",
    "dict-row": {0: 1},
    "non-numeric-strings": [("0", "x")],
}


@pytest.mark.parametrize("name", list(REJECTED))
def test_malformed_rows_raise_a_value_error_naming_the_row(name):
    with pytest.raises(ValueError, match=r"turn_spans\[1\]"):
        run(column([CANON[0], REJECTED[name]]))


def test_the_message_states_the_bounds_and_the_offending_span():
    with pytest.raises(ValueError, match=r"turn_spans\[1\]: span \(4, 7\) must satisfy 0 <= start < end <= 6"):
        run(column([CANON[0], [(4, 7)]]))
    with pytest.raises(ValueError, match=r"turn_spans\[1\]: span \(2, 3\) must satisfy 4 <= start"):
        run(column([CANON[0], [(0, 4), (2, 3)]]))


def test_row_count_mismatch_and_a_non_sequence_column_are_named():
    with pytest.raises(ValueError, match="3 rows for a batch of 2"):
        run(column([CANON[0], CANON[1], CANON[1]]))
    with pytest.raises(ValueError, match="1 rows for a batch of 2"):
        run(column([CANON[0]]))
    # LOW 4, closed by 988a10bc: a scalar column raises the named ValueError instead of a TypeError at len()
    with pytest.raises(ValueError, match=r"turn_spans must hold one entry per row \(2 rows\), got a int"):
        run(5)


@pytest.mark.parametrize(
    "value,name",
    [
        (0.9, "float"),
        (np.float64(2.0), "float64"),
        (torch.tensor(2.0), r"Tensor\[torch.float32\]"),
        (False, "bool"),
        (np.bool_(True), "bool"),  # numpy 2 names np.bool_ "bool"
        (torch.tensor(True), r"Tensor\[torch.bool\]"),
    ],
    ids=["float", "np-float64", "torch-float", "bool", "np-bool", "torch-bool"],
)
def test_non_integral_offsets_are_rejected_naming_the_row_and_the_type(value, name):
    """LOW 3, closed by 988a10bc: no silent int() truncation; floats and bools of every flavour are named."""
    with pytest.raises(ValueError, match=rf"turn_spans\[0\]: span .* has a {name} offset"):
        _validated_turn_spans(0, [(value, 2)], L)
    with pytest.raises(ValueError, match=r"turn_spans\[1\]"):
        run(column([CANON[0], [(0, 2), (4, value)]]))


def test_integral_offsets_of_every_flavour_are_accepted():
    assert _validated_turn_spans(0, [(np.int64(0), np.int32(2)), (torch.tensor(2), 4)], L) == [(0, 2), (2, 4)]
    assert _validated_turn_spans(0, [(np.uint8(0), torch.tensor(6, dtype=torch.int32))], L) == [(0, 6)]
