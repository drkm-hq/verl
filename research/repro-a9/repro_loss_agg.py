"""Micro-batch invariance of the gradient for every loss_agg_mode x policy loss.

verl's engine (verl/workers/engine/fsdp/transformer_impl.py:753-771) calls loss.backward() once per
micro-batch with NO rescaling; every loss is expected to normalise itself with the *global* batch
info (dp_size, batch_num_tokens, global_batch_size) that losses.py:77-80 copies into
config.global_batch_info.  So for a mini-batch split into micro-batches of unequal token counts the
sum of the per-micro-batch gradients must equal the gradient of the loss on the whole mini-batch.

We check d(loss)/d(log_prob) for
  (a) one micro-batch (whole)
  (b) three unequal micro-batches [1, 2, 5] rows, same global info, gradients accumulated
  (c) dp=2 simulation: two ranks with 4 rows each, dp_size=2, gradient averaged (FSDP mean-reduce)
and two column-width conventions:
  fixed  : every micro-batch keeps the mini-batch's padded width T (legacy trainer, max_response_len set)
  mbwidth: every micro-batch is trimmed to its own max response length (V1 trainer: response_mask is a
           nested tensor and losses.py:103 pads it to the micro-batch max; no_padding_2_padding has no
           max_response_len and uses response_lens.max(), padding.py:113-119)
"""

import sys

import torch

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.workers.config import ActorConfig, PolicyLossConfig

torch.manual_seed(0)
B, T = 8, 12
lengths = torch.tensor([12, 0, 3, 7, 12, 1, 9, 5])  # row 1 fully masked (zero response tokens)
mask = (torch.arange(T)[None, :] < lengths[:, None]).float()
base_logp = -1.5 + 0.5 * torch.randn(B, T)
old_logp = base_logp + 0.1 * torch.randn(B, T)
ref_logp = base_logp + 0.2 * torch.randn(B, T)
adv = torch.randn(B, T)
adv[2] = 0.7  # a row with uniform positive advantage
is_w = (1.0 + 0.3 * torch.randn(B, T)).clamp(min=0.2, max=2.0) * mask
returns = torch.randn(B, T)
old_values = torch.randn(B, T)

AGG_MODES = ["token-mean", "token-sum", "seq-mean-token-sum", "seq-mean-token-sum-norm", "seq-mean-token-mean"]
LOSS_MODES = ["vanilla", "gspo", "sapo", "cispo", "gpg", "dro", "dppo_tv", "dppo_kl", "geo_mean", "clip_cov", "kl_cov"]
SPLITS = [[0], [1, 2], [3, 4, 5, 6, 7]]
DP_SPLITS = [[0, 1, 2, 3], [4, 5, 6, 7]]


def make_cfg(loss_mode, agg_mode, loss_scale_factor=None):
    pl = PolicyLossConfig(loss_mode=loss_mode, dro_beta=0.1 if loss_mode == "dro" else None)
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=-1,
        loss_agg_mode=agg_mode,
        loss_scale_factor=loss_scale_factor,
        policy_loss=pl,
    )


def global_info(dp_size=1):
    return dict(
        dp_size=dp_size,
        batch_num_tokens=int(mask.sum().item()),
        global_batch_size=B,
        loss_scale_factor=None,
    )


def run_policy(loss_mode, agg_mode, rows, width, dp_size, leaf, use_is=False):
    """Loss of policy loss `loss_mode` on `rows`, columns trimmed to `width`; returns loss (graph on leaf)."""
    torch.manual_seed(1234)  # clip_cov uses randperm
    cfg = make_cfg(loss_mode, agg_mode)
    cfg.global_batch_info.update(global_info(dp_size))
    fn = get_policy_loss_fn(loss_mode)
    idx = torch.tensor(rows)
    lp = leaf[idx][:, :width]
    loss, _ = fn(
        old_log_prob=old_logp[idx][:, :width],
        log_prob=lp,
        advantages=adv[idx][:, :width],
        response_mask=mask[idx][:, :width],
        loss_agg_mode=agg_mode,
        config=cfg,
        rollout_is_weights=(is_w[idx][:, :width] if use_is else None),
    )
    return loss


def run_term(term, agg_mode, rows, width, dp_size, leaf):
    idx = torch.tensor(rows)
    lp = leaf[idx][:, :width]
    m = mask[idx][:, :width]
    info = global_info(dp_size)
    if term == "entropy":  # entropy enters the loss through agg_loss(entropy, mask, mode, **info), losses.py:164-168
        return agg_loss(lp, m, agg_mode, **info)
    if term == "kl":  # losses.py:175-180
        kld = kl_penalty(lp, ref_logp[idx][:, :width], "low_var_kl")
        return agg_loss(kld, m, agg_mode, **info)
    if term == "value":  # losses.py:226-237
        vf, _ = compute_value_loss(
            vpreds=lp,
            returns=returns[idx][:, :width],
            values=old_values[idx][:, :width],
            response_mask=m,
            cliprange_value=0.5,
            loss_agg_mode=agg_mode,
            dp_size=dp_size,
            batch_num_tokens=info["batch_num_tokens"],
            global_batch_size=info["global_batch_size"],
        )
        return vf
    raise ValueError(term)


def grad_of(fn_loss, rows_list, width_mode, dp_size):
    leaf = base_logp.clone().requires_grad_(True)
    total = 0.0
    for rows in rows_list:
        width = T if width_mode == "fixed" else max(int(lengths[rows].max().item()), 1)
        loss = fn_loss(rows, width, dp_size, leaf)
        loss.backward()
        total += loss.item()
    g = leaf.grad.clone()
    if dp_size > 1:  # FSDP/DDP averages gradients across ranks
        g = g / dp_size
    return g, total


def compare(name, fn_loss):
    rows_all = [list(range(B))]
    g_whole, l_whole = grad_of(fn_loss, rows_all, "fixed", 1)
    scale = g_whole.abs().max().item() + 1e-12
    out = {}
    for width_mode in ["fixed", "mbwidth"]:
        g_split, l_split = grad_of(fn_loss, SPLITS, width_mode, 1)
        out[width_mode] = ((g_split - g_whole).abs().max().item() / scale, l_split / (l_whole if abs(l_whole) > 1e-12 else 1.0))
    g_dp, _ = grad_of(fn_loss, DP_SPLITS, "fixed", 2)
    out["dp2"] = ((g_dp - g_whole).abs().max().item() / scale, None)
    return out


def fmt(v):
    rel, ratio = v
    flag = "OK " if rel < 1e-4 else "BAD"
    s = f"{flag} rel_grad_err={rel:9.2e}"
    if ratio is not None:
        s += f" loss_ratio(split/whole)={ratio:7.3f}"
    return s


print(f"torch {torch.__version__}; B={B} T={T} lengths={lengths.tolist()} valid_tokens={int(mask.sum())}")
print("=" * 110)
print("POLICY LOSSES: gradient wrt log_prob, micro-batch split [1,2,5] rows vs whole; dp=2 simulation")
bad = []
for agg_mode in AGG_MODES:
    for loss_mode in LOSS_MODES:
        try:
            res = compare(f"{loss_mode}/{agg_mode}", lambda rows, w, dp, leaf: run_policy(loss_mode, agg_mode, rows, w, dp, leaf))
        except Exception as e:  # noqa: BLE001
            print(f"{loss_mode:9s} {agg_mode:24s} ERROR {type(e).__name__}: {str(e)[:80]}")
            continue
        line = f"{loss_mode:9s} {agg_mode:24s} fixed: {fmt(res['fixed'])} | mbwidth: {fmt(res['mbwidth'])} | dp2: {fmt(res['dp2'])}"
        print(line)
        for k, v in res.items():
            if v[0] >= 1e-4:
                bad.append((loss_mode, agg_mode, k, v[0]))
print("=" * 110)
print("TERMS (entropy via agg_loss, low_var_kl via agg_loss, critic value loss)")
for term in ["entropy", "kl", "value"]:
    for agg_mode in AGG_MODES:
        res = compare(f"{term}/{agg_mode}", lambda rows, w, dp, leaf: run_term(term, agg_mode, rows, w, dp, leaf))
        print(f"{term:9s} {agg_mode:24s} fixed: {fmt(res['fixed'])} | mbwidth: {fmt(res['mbwidth'])} | dp2: {fmt(res['dp2'])}")
        for k, v in res.items():
            if v[0] >= 1e-4:
                bad.append((term, agg_mode, k, v[0]))
print("=" * 110)
print("vanilla + rollout_is_weights (TIS path, losses.py:109 -> core_algos.py:1364-1365)")
for agg_mode in AGG_MODES:
    res = compare("vanilla+is", lambda rows, w, dp, leaf: run_policy("vanilla", agg_mode, rows, w, dp, leaf, use_is=True))
    print(f"{'vanilla+is':9s} {agg_mode:24s} fixed: {fmt(res['fixed'])} | mbwidth: {fmt(res['mbwidth'])} | dp2: {fmt(res['dp2'])}")
print("=" * 110)
print("NOT micro-batch invariant (rel grad err >= 1e-4):")
for b in bad:
    print("  ", b)
sys.exit(0)
