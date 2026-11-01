"""clip_cov: is the token selection (and hence the gradient) invariant to the micro-batch split? (core_algos.py:1808-1832)"""
import torch
from verl.trainer.ppo.core_algos import get_policy_loss_fn
from verl.workers.config import ActorConfig, PolicyLossConfig

B, T = 64, 32
fn = get_policy_loss_fn("clip_cov")
worst = 0.0
for seed in range(6):
    torch.manual_seed(seed)
    lengths = torch.randint(1, T + 1, (B,))
    mask = (torch.arange(T)[None, :] < lengths[:, None]).float()
    base = -1.5 + 0.5 * torch.randn(B, T)
    old = base + 0.3 * torch.randn(B, T)  # off-policy enough that clip_by_origin is not everything
    adv = torch.randn(B, T)
    def run(splits, seed_sel):
        leaf = base.clone().requires_grad_(True)
        pl = PolicyLossConfig(loss_mode="clip_cov", clip_cov_ratio=0.05, clip_cov_lb=-1e9, clip_cov_ub=1e9)
        cfg = ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=-1, loss_agg_mode="token-mean", policy_loss=pl)
        cfg.global_batch_info.update(dict(dp_size=1, batch_num_tokens=int(mask.sum()), global_batch_size=B, loss_scale_factor=None))
        tot = 0.0
        for rows in splits:
            torch.manual_seed(seed_sel)
            idx = torch.tensor(rows)
            loss, m = fn(old_log_prob=old[idx], log_prob=leaf[idx], advantages=adv[idx], response_mask=mask[idx], loss_agg_mode="token-mean", config=cfg)
            loss.backward(); tot += loss.item()
        return leaf.grad.clone(), tot
    g1, l1 = run([list(range(B))], 7)
    g2, l2 = run([list(range(0, 8)), list(range(8, 24)), list(range(24, B))], 7)
    rel = (g1 - g2).abs().max().item() / g1.abs().max().item()
    worst = max(worst, rel)
    print(f"seed={seed}: loss whole={l1:+.5f} split={l2:+.5f} max rel grad err={rel:.3e}")
print("worst rel grad err:", worst)
