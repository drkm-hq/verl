"""V1 trainer path: seq-mean-token-sum-norm's horizon is the *micro-batch* max response length.

Through the real engine loss function verl/workers/utils/losses.py::ppo_loss with nested (jagged) inputs,
exactly as the V1 trainer hands them over (verl/trainer/ppo/v1/trainer_base.py:1668-1669 and :1798 write
old_log_probs / advantages / response_mask back as nested tensors; nothing sets `max_response_len`, which only
verl/workers/utils/padding.py:49-51 (left_right_2_no_padding, the legacy trainer) assigns).

  losses.py:59   log_prob = no_padding_2_padding(model_output["log_probs"], data)
  padding.py:113 max_response_len = get_non_tensor_data(data, "max_response_len", default=-1)
  padding.py:118 if max_response_len < 0: max_response_len = response_lens.max()      <- micro-batch max
  losses.py:103  data.select(...).to_padded_tensor()                                     <- micro-batch max
  core_algos.py:1189-1193  if loss_agg_mode == "seq-mean-token-sum-norm": loss /= loss_mask.shape[-1]

So each micro-batch is normalised by its own padded width, and the accumulated gradient depends on how the
mini-batch was split (rearrange_micro_batches / dynamic bsz), unless actor.loss_scale_factor is set.
"""

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss

torch.manual_seed(0)
prompt_lens = [4, 3, 5, 2, 6, 3, 4, 5]
resp_lens = [12, 1, 3, 7, 12, 2, 9, 5]
B = len(resp_lens)
prompts = [torch.randint(10, 100, (p,)) for p in prompt_lens]
responses = [torch.randint(10, 100, (r,)) for r in resp_lens]
input_ids = [torch.cat([p, r]) for p, r in zip(prompts, responses, strict=True)]
resp_mask = [torch.ones(r, dtype=torch.int64) for r in resp_lens]
resp_mask[0][4:7] = 0  # a tool-response span inside the first response (multi-turn)
old_lp = [-1.5 + 0.5 * torch.randn(r) for r in resp_lens]
adv = [torch.randn(r) for r in resp_lens]
seq_lens = torch.tensor([p + r for p, r in zip(prompt_lens, resp_lens, strict=True)])
offsets = torch.cat([torch.zeros(1, dtype=torch.long), seq_lens.cumsum(0)])
N_valid = int(sum(int(m.sum()) for m in resp_mask))


def nest(lst):
    return torch.nested.as_nested_tensor(lst, layout=torch.jagged)


def make_td():
    td = TensorDict(
        {
            "prompts": nest(prompts),
            "responses": nest(responses),
            "input_ids": nest(input_ids),
            "response_mask": nest(resp_mask),
            "loss_mask": nest(resp_mask),
            "old_log_probs": nest(old_lp),
            "advantages": nest(adv),
        },
        batch_size=[B],
    )
    # what FSDPEngine.forward_backward_batch assigns (transformer_impl.py:734-739) + the trainer (v1 :1856)
    tu.assign_non_tensor(td, dp_size=1, batch_num_tokens=N_valid, global_batch_size=B)
    return td


def make_cfg(agg_mode, loss_scale_factor=None):
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=-1,
        loss_agg_mode=agg_mode,
        loss_scale_factor=loss_scale_factor,
        policy_loss=PolicyLossConfig(loss_mode="vanilla"),
    )


base_vals = -1.2 + 0.4 * torch.randn(int(offsets[-1]))


def logprobs_for(leaf, rows):
    pieces = [leaf[offsets[i] : offsets[i + 1]] for i in rows]
    lens = torch.tensor([int(seq_lens[i]) for i in rows])
    off = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)])
    return torch.nested.nested_tensor_from_jagged(torch.cat(pieces), off)


def grad_for(agg_mode, splits, loss_scale_factor=None):
    leaf = base_vals.clone().requires_grad_(True)
    cfg = make_cfg(agg_mode, loss_scale_factor)
    td = make_td()
    total = 0.0
    widths = []
    for rows in splits:
        mb = tu.index_select_tensor_dict(td, torch.tensor(rows))
        model_output = {"log_probs": logprobs_for(leaf, rows)}
        loss, _ = ppo_loss(cfg, model_output, mb)
        loss.backward()
        total += loss.item()
        widths.append(max(resp_lens[i] for i in rows))
    return leaf.grad.clone(), total, widths


SPLITS = [[0], [1, 2], [3, 4, 5, 6, 7]]
SPLITS2 = [[0, 1, 2, 3], [4, 5, 6, 7]]
WHOLE = [list(range(B))]

print(f"B={B} resp_lens={resp_lens} valid_response_tokens={N_valid}")
for agg_mode, lsf in [
    ("token-mean", None),
    ("seq-mean-token-mean", None),
    ("seq-mean-token-sum", None),
    ("seq-mean-token-sum-norm", None),
    ("seq-mean-token-sum-norm", 12),
    ("seq-mean-token-sum-norm", 4096),
]:
    g_whole, l_whole, w_whole = grad_for(agg_mode, WHOLE, lsf)
    scale = g_whole.abs().max().item()
    for name, splits in [("split[1,2,5]", SPLITS), ("split[4,4]", SPLITS2)]:
        g_split, l_split, widths = grad_for(agg_mode, splits, lsf)
        rel = (g_split - g_whole).abs().max().item() / scale
        flag = "OK " if rel < 1e-5 else "BAD"
        print(
            f"{flag} {agg_mode:24s} loss_scale_factor={str(lsf):5s} {name:12s} "
            f"micro-batch widths={widths} whole width={w_whole} "
            f"loss split/whole={l_split / l_whole:6.3f} max rel grad err={rel:.2e}"
        )
