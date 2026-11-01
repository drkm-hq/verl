"""KL estimator checks (core_algos.py:2188-2251, ray_trainer.py:78-117)."""

import torch

from verl.trainer.ppo.core_algos import kl_penalty, kl_penalty_forward

torch.manual_seed(0)
print("=" * 100)
print("A. k3 cancellation in float32 (core_algos.py:2240-2245 computes exp(x) - x - 1)")
for x0 in [-1e-4, 1e-4, -1e-5, 1e-3]:
    n = 32768
    logp = torch.zeros(n)
    ref = torch.full((n,), x0)  # kl = ref - logp = x0
    k3 = kl_penalty_forward(logp, ref, "low_var_kl")
    x = torch.full((n,), x0)
    expm1_form = torch.expm1(x) - x
    f64 = (torch.expm1(x.double()) - x.double())
    print(
        f"  x={x0:+.0e}: per-token k3={k3[0].item():+.5e} (expm1 form {expm1_form[0].item():+.5e}, f64 {f64[0].item():+.5e});"
        f" sum over {n} tokens: current={k3.sum().item():+.5e} expm1={expm1_form.sum().item():+.5e} f64={f64.sum().item():+.5e}"
    )

print("=" * 100)
print("B. low_var_kl clamps: inner clamp of the log-ratio to [-20,20] and outer clamp of kld to [-10,10]")
print("   -> gradient wrt logprob is exactly 0 outside the window (dead zone)")
print("   d = ref_logprob - logprob;  kld = exp(d) - d - 1")
for d in [-25.0, -15.0, -12.0, -10.0, -5.0, -1.0, 0.0, 1.0, 2.0, 2.5, 3.0, 5.0, 15.0]:
    logp = torch.tensor([-3.0], requires_grad=True)
    ref = torch.tensor([-3.0 + d])
    kld = kl_penalty(logp, ref, "low_var_kl")
    kld.sum().backward()
    exact = (torch.exp(torch.tensor(d)) - d - 1).item()
    print(f"  d={d:+6.1f}: kld={kld.item():8.3f} (unclamped {exact:12.3f})  d(kld)/d(logprob)={logp.grad.item():+10.4f}")

print("=" * 100)
print("C. 'k3+' straight-through: value == k3, gradient == k2 gradient (core_algos.py:2200-2213)")
logp = torch.randn(1000, requires_grad=True)
ref = torch.randn(1000)
v_plus = kl_penalty(logp, ref, "k3+")
(g_plus,) = torch.autograd.grad(v_plus.sum(), logp)
v_k3 = kl_penalty(logp, ref, "k3")
logp2 = logp.detach().clone().requires_grad_(True)
v_k2 = kl_penalty(logp2, ref, "k2")
(g_k2,) = torch.autograd.grad(v_k2.sum(), logp2)
print(f"  max|k3+ - k3| = {(v_plus - v_k3).abs().max().item():.2e};  max|grad(k3+) - grad(k2)| = {(g_plus - g_k2).abs().max().item():.2e}")

print("=" * 100)
print("D. Exact expected gradients on a 5-way categorical (on-policy sampling): which estimator's gradient is the KL gradient?")
theta = torch.tensor([0.5, -0.3, 1.2, 0.0, -1.0], requires_grad=True)
q = torch.softmax(torch.tensor([0.0, 0.4, -0.2, 0.9, -0.5]), 0)
logpi = torch.log_softmax(theta, 0)
pi = logpi.exp()
true_kl = (pi * (logpi - q.log())).sum()
(g_true,) = torch.autograd.grad(true_kl, theta)
for name in ["k1", "k2", "k3", "k3+"]:
    theta2 = theta.detach().clone().requires_grad_(True)
    logpi2 = torch.log_softmax(theta2, 0)
    # E_{a~pi}[ grad estimator(a) ] with the sampling distribution held fixed (what a sampled-token loss does)
    est = kl_penalty(logpi2, q.log(), name)
    expected = (pi.detach() * est).sum()
    (g,) = torch.autograd.grad(expected, theta2)
    print(f"  {name:3s}: max|E[grad] - grad KL| = {(g - g_true).abs().max().item():.2e}   (E[value]={float((pi.detach() * est.detach()).sum()):.4f}, KL={true_kl.item():.4f})")

print("=" * 100)
print("E. apply_kl_penalty sign (ray_trainer.py:100-106): token_level_rewards = scores - beta * kld, kld = k1 = logp - ref")
logp = torch.tensor([[-1.0, -2.0]])
ref = torch.tensor([[-2.0, -1.0]])
print(f"  kld(k1) = {kl_penalty(logp, ref, 'kl').tolist()}  -> positive where the policy is MORE likely than ref (penalised), negative otherwise (rewarded)")
