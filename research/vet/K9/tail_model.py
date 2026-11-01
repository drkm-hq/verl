"""K9 vet: expected gen-time saving from over-sample-and-abort in a single decode wave.
Model (stated assumptions): per-sequence response length ~ lognormal(mu, sigma) truncated at CAP;
sigma fitted so that mean = MEAN and P(len >= CAP) = CLIP (BASELINE-A run metrics); a GRPO group
completes when its slowest of n=8 siblings completes; with all B*n sequences resident in one decode wave
(8 replicas x max_num_seqs 256 = 2048 slots >= 1024), wall time is ~ affine in the longest resident length:
gen = T0 + k * max_len, with T0 (wake/prefill/dispatch) and k calibrated from the run's gen time.
Over-sampling by r dispatches ceil(B(1+r)) groups and stops when B groups are terminal, i.e. at the
B/(B(1+r)) = 1/(1+r) quantile of the group-max length distribution (the aborted groups are the longest).
Run: python3 -I research/vet/K9/tail_model.py [MEAN CLIP CAP]
"""
import math, random, sys

MEAN = float(sys.argv[1]) if len(sys.argv) > 1 else 1230.0   # response_length/mean (clipped mean)
CLIP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.19    # response_length/clip_ratio 0.15-0.21 in the run
CAP = float(sys.argv[3]) if len(sys.argv) > 3 else 2048.0
B, n = 128, 8
GEN, T0 = 12.77, 1.5   # BASELINE-A mean gen s; assumed fixed overhead (wake + prefill + dispatch + sleep)

def phi_inv(p):  # Acklam approximation
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00]
    if p < 0.02425:
        q=math.sqrt(-2*math.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1-0.02425:
        q=math.sqrt(-2*math.log(1-p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q=p-0.5; r=q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def fit_trunc(mean_trunc, clip, cap):
    """(mu, sigma) of the untruncated lognormal such that P(X>=cap)=clip and E[min(X,cap)]=mean_trunc
    (the run reports the *clipped* mean, so the untruncated mean is higher)."""
    z = phi_inv(1 - clip)
    lo, hi = 0.02, 3.0
    for _ in range(80):
        sig = (lo + hi) / 2
        mu_ = math.log(cap) - z * sig
        et = math.exp(mu_ + sig * sig / 2) * Phi((math.log(cap) - mu_ - sig * sig) / sig) + cap * clip
        if et > mean_trunc:
            lo = sig
        else:
            hi = sig
    sig = (lo + hi) / 2
    return math.log(cap) - z * sig, sig

mu, sigma = fit_trunc(MEAN, CLIP, CAP)
print(f"fit: mean={MEAN:.0f} clip={CLIP} cap={CAP:.0f} -> lognormal sigma={sigma:.3f} mu={mu:.3f}")

random.seed(0)
CLIP_CUR = CLIP
def run(sig, mu_):
    global CLIP_CUR
    CLIP_CUR = 1 - Phi((math.log(CAP) - mu_) / sig)
    trials = 400
    sav = {r: [] for r in (0.1, 0.25, 0.5)}
    base_max = []
    for _ in range(trials):
        gm = []
        for _g in range(int(B * 1.5)):
            gm.append(max(min(CAP, math.exp(random.gauss(mu_, sig))) for _ in range(n)))
        full = max(gm[:B]); base_max.append(full)
        for r in sav:
            m = int(math.ceil(B * (1 + r)))
            stop = sorted(gm[:m])[B - 1]      # length at which the B-th group finishes
            sav[r].append(stop)
    k = (GEN - T0) / (sum(base_max) / trials)
    print(f"  baseline: E[max resident len]={sum(base_max)/trials:.0f}  k={k*1e3:.2f} ms/token  gen={GEN} s")
    for r in sav:
        s = sum(sav[r]) / trials
        g = T0 + k * s
        print(f"  r={r:<4}: P(group has a capped seq)={1-(1-CLIP_CUR)**n:.0%}  stop at E[len]={s:.0f}  gen~{g:.2f} s  saving {GEN-g:.2f} s = {(GEN-g)/GEN:.0%} of gen, {(GEN-g)/31.15:.1%} of step; "
              f"wasted/recycled tokens ~ {r/(1+r):.0%} of dispatched groups")
print("BASELINE-A fit (GSM8K, Qwen3-1.7B, cap 2048; k calibrated from the run, % of step uses 31.15 s):"); run(sigma, mu)

def fit_and_run(mean, clip, cap, label):
    global CAP
    CAP = cap
    mu2, sig2 = fit_trunc(mean, clip, cap)
    print(f"{label}: mean={mean:.0f} clip={clip} cap={cap:.0f} -> sigma={sig2:.3f}  (only '% of gen' is meaningful here)")
    run(sig2, mu2)

fit_and_run(3500.0, 0.01, 16384.0, "long-tail reference A (DAPO-math-like, 1% clipped at 16k)")
fit_and_run(3500.0, 0.003, 16384.0, "long-tail reference B (0.3% clipped at 16k)")
fit_and_run(3500.0, 0.05, 8192.0, "long-tail reference C (5% clipped at 8k: cap-bound, abort cannot help)")
