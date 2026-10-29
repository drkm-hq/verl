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
"""``actor.skip_old_log_prob_when_on_policy``: the exactness proof, through verl's real ``ppo_loss``.

With one mini-batch per sampled batch and ``ppo_epochs=1`` every micro-batch forward of the actor
update runs before the single optimizer step, so the log-probs the update pass computes are the
old log-probs. ``ppo_loss`` then uses ``log_prob.detach()`` as ``old_log_prob``
(``on_policy_skip_old_log_prob``) instead of the field a separate old-log-prob forward would
have written. This test checks, on a tiny Qwen3 in fp32:

* the update pass's response log-probs equal a separate eval/no-grad old-log-prob forward
  bitwise, although the two passes micro-batch the data differently (2/2/2/2 vs 3/3/2) and the
  update pass accumulates gradients;
* the accumulated gradient with ``old_log_prob = log_prob.detach()`` equals the gradient
  against the stored old log-probs bitwise, and ``pg_clipfrac`` / ``ppo_kl`` are exactly 0;
* negative control: after one optimizer step (a second mini-batch) the equality is gone, which
  is why the config gate only accepts the single-mini-batch case.
"""

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl.utils import tensordict_utils as tu
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss

B, P, R, V = 8, 12, 20, 512  # batch, prompt length, max response length, vocab
RESP_LENS = [20, 14, 17, 11, 20, 13, 19, 16]  # ragged responses


def _make_model():
    cfg = Qwen3Config(
        vocab_size=V,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    return Qwen3ForCausalLM(cfg).float()


def _make_data():
    """Right-padded inputs of a fixed length P + R, so every micro-batch has the same sequence shape and
    only the batch dimension varies between the two passes (as in verl, where shapes come from the
    micro-batching, not from the sample)."""
    input_ids = torch.randint(0, V, (B, P + R))
    attention_mask = torch.ones(B, P + R, dtype=torch.long)
    for i, r in enumerate(RESP_LENS):
        attention_mask[i, P + r :] = 0
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    rows = {
        "prompts": [input_ids[i, :P] for i in range(B)],
        "responses": [input_ids[i, P : P + r] for i, r in enumerate(RESP_LENS)],
        "response_mask": [torch.ones(r, dtype=torch.long) for r in RESP_LENS],
        "advantages": [torch.randn(r) for r in RESP_LENS],
    }
    return input_ids, attention_mask, position_ids, rows


def _nested(rows):
    return torch.nested.as_nested_tensor(list(rows), layout=torch.jagged)


class _Setup:
    def __init__(self, entropy_coeff: float = 0.0):
        torch.manual_seed(0)
        self.model = _make_model()
        self.input_ids, self.attention_mask, self.position_ids, self.rows = _make_data()
        self.config = ActorConfig(
            strategy="fsdp",
            rollout_n=1,
            ppo_micro_batch_size_per_gpu=2,
            clip_ratio=0.2,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            clip_ratio_c=3.0,
            entropy_coeff=entropy_coeff,
            use_kl_loss=False,
        )
        self.total_response_tokens = sum(RESP_LENS)

    def full_log_probs(self, idx: list[int]) -> torch.Tensor:
        """The engine's ``model_output["log_probs"]``: a nested (bsz, prompt_len + response_len) tensor where
        position t holds log p(token_{t+1} | tokens <= t)."""
        out = self.model(
            input_ids=self.input_ids[idx],
            attention_mask=self.attention_mask[idx],
            position_ids=self.position_ids[idx],
            use_cache=False,
        )
        rows = []
        for j, i in enumerate(idx):
            length = P + RESP_LENS[i]
            labels = torch.roll(self.input_ids[i, :length], shifts=-1)
            rows.append(logprobs_from_logits(out.logits[j, :length].float(), labels))
        return _nested(rows)

    def micro_batch_data(self, idx: list[int], old_log_probs: list[torch.Tensor] | None, on_policy: bool):
        tensors = {key: _nested(rows[i] for i in idx) for key, rows in self.rows.items()}
        if old_log_probs is not None:
            tensors["old_log_probs"] = _nested(old_log_probs[i] for i in idx)
        data = tu.get_tensordict(tensor_dict=tensors)
        # the loss is normalised by the global token count, so the micro-batch losses add up to the
        # batch loss and the accumulated gradient is the batch gradient (as in the engine)
        tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=self.total_response_tokens, global_batch_size=B)
        if on_policy:
            tu.assign_non_tensor(data, on_policy_skip_old_log_prob=True)
        return data

    def old_log_prob_pass(self, micro_sizes):
        """The old-log-prob pass as verl runs it: eval mode, no grad, its own micro-batching."""
        self.model.eval()
        rows = []
        with torch.no_grad():
            for idx in _chunks(micro_sizes):
                full = self.full_log_probs(idx)
                rows.extend(_response_rows(full, idx))
        return rows

    def update_pass(self, micro_sizes, old_log_probs, on_policy: bool):
        """One mini-batch == the whole batch, micro-batched with gradient accumulation, no optimizer step."""
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        rows, clipfracs, kls = [], [], []
        for idx in _chunks(micro_sizes):
            full = self.full_log_probs(idx)
            data = self.micro_batch_data(idx, None if on_policy else old_log_probs, on_policy)
            loss, metrics = ppo_loss(self.config, {"log_probs": full}, data)
            loss.backward()
            rows.extend(_response_rows(full.detach(), idx))
            clipfracs.append(metrics["actor/pg_clipfrac"].aggregate())
            kls.append(abs(metrics["actor/ppo_kl"].aggregate()))
        grads = [p.grad.detach().clone() for p in self.model.parameters() if p.grad is not None]
        return rows, grads, max(clipfracs), max(kls)


def _chunks(sizes):
    start, out = 0, []
    for size in sizes:
        out.append(list(range(start, start + size)))
        start += size
    assert start == B
    return out


def _response_rows(full: torch.Tensor, idx: list[int]) -> list[torch.Tensor]:
    # response token r is predicted by sequence position P - 1 + r (what no_padding_2_padding slices)
    return [row[P - 1 : P - 1 + RESP_LENS[i]] for row, i in zip(full.unbind(), idx, strict=True)]


def test_update_pass_log_probs_and_gradient_match_the_old_log_prob_pass_bitwise():
    setup = _Setup()
    old = setup.old_log_prob_pass([2, 2, 2, 2])

    # PPO as on main: the stored old log-probs, micro-batched 3/3/2 with gradient accumulation
    rows_main, grads_main, clip_main, kl_main = setup.update_pass([3, 3, 2], old, on_policy=False)
    # the skip: old_log_prob = log_prob.detach(), same micro-batching
    rows_skip, grads_skip, clip_skip, kl_skip = setup.update_pass([3, 3, 2], None, on_policy=True)

    for a, b in zip(rows_main, old, strict=True):
        assert torch.equal(a, b), "update-pass log-probs differ from the old-log-prob pass"
    for a, b in zip(rows_skip, old, strict=True):
        assert torch.equal(a, b)
    assert len(grads_main) == len(grads_skip) > 0
    for a, b in zip(grads_main, grads_skip, strict=True):
        assert torch.equal(a, b), "the on-policy gradient differs from the stored-old-log-prob gradient"
    assert clip_skip == 0.0 and kl_skip == 0.0
    assert clip_main == 0.0 and kl_main == 0.0


def test_a_second_mini_batch_is_not_on_policy():
    """After one AdamW step the next mini-batch's log-probs are no longer the old ones: only the
    single-mini-batch configuration is exact, which is what validate_config enforces."""
    setup = _Setup()
    old = setup.old_log_prob_pass([2, 2, 2, 2])
    optimizer = torch.optim.AdamW(setup.model.parameters(), lr=1e-4)

    setup.model.train()
    setup.model.zero_grad(set_to_none=True)
    first = list(range(4))
    full = setup.full_log_probs(first)
    loss, _ = ppo_loss(setup.config, {"log_probs": full}, setup.micro_batch_data(first, old, on_policy=False))
    loss.backward()
    optimizer.step()

    second = list(range(4, 8))
    with torch.no_grad():
        rows = _response_rows(setup.full_log_probs(second), second)
    max_diff = max((a - old[i]).abs().max().item() for a, i in zip(rows, second, strict=True))
    assert max_diff > 1e-5, "the second mini-batch unexpectedly still matches the sampling weights"


@pytest.mark.parametrize(
    ("on_policy", "entropy_coeff", "expect_entropy_loss"),
    [(True, 0.0, False), (True, 0.01, True), (False, 0.0, True)],
)
def test_entropy_is_a_loss_term_only_when_it_carries_a_coefficient_or_the_flag_is_off(
    on_policy, entropy_coeff, expect_entropy_loss
):
    """Under the skip with entropy_coeff=0 the entropy is computed (without grad) only for the trainer's
    actor/entropy metric: no actor/entropy_loss, nothing added to the loss. Flag off: unchanged."""
    setup = _Setup(entropy_coeff=entropy_coeff)
    old = setup.old_log_prob_pass([2, 2, 2, 2])
    idx = list(range(3))
    setup.model.train()
    full = setup.full_log_probs(idx)
    with torch.no_grad():
        entropy = _nested(torch.rand(P + RESP_LENS[i]) for i in idx)
    data = setup.micro_batch_data(idx, None if on_policy else old, on_policy)

    loss, metrics = ppo_loss(setup.config, {"log_probs": full, "entropy": entropy}, data)
    pg_loss, _ = ppo_loss(setup.config, {"log_probs": full}, data)

    assert ("actor/entropy_loss" in metrics) is expect_entropy_loss
    if entropy_coeff == 0.0:
        assert torch.equal(loss, pg_loss)
    else:
        assert not torch.equal(loss, pg_loss)
