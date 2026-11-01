"""K6 vet: re-derive the V1 driver<->TransferQueue bytes per step from the field lists in
verl/trainer/ppo/v1/trainer_base.py @ 2ab35292 (sites: _compute_old_log_prob :1656-1665,
_compute_ref_log_prob :1713-1718, _compute_values :1734-1740, _compute_advantage :1746-1747,
_compute_metrics :1876-1894) at the BASELINE-A shape measured in bench/results/baseline-a-main-1.
Time model: research/03 section 3.8 measured 25-40 us fixed cost per row per field on GET
(unbind + msgpack frame + as_nested_tensor) and ~1 GB/s payload; PUT ~2x the per-row cost.
Pure arithmetic, no verl import.  Run: python3 -I research/vet/K6/bytes_baseline_a.py [N P L]
"""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024      # B=128 prompts x n=8
P = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0    # mean valid prompt tokens (see run metrics)
L = float(sys.argv[3]) if len(sys.argv) > 3 else 1225.0  # mean valid response tokens (see run metrics)
S = P + L
F32, I64 = 4, 8
FIX_GET, FIX_PUT = 30e-6, 60e-6   # s per row per field (03 s3.8 mid-range)
BW = 1.0e9                        # B/s payload (03 s3.8: 0.45-1.3 GB/s)

# per-row bytes of each field as written by agent_loop_tq.py:187-232 / tqbridge outputs
fields = {
    "log_probs": F32 * S, "entropy": F32 * S,            # full sequence (worker output, prompt+response)
    "values": F32 * S,                                    # critic output (full seq) - absent in GRPO
    "response_mask": I64 * L, "responses": I64 * L, "prompts": I64 * P,
    "rollout_log_probs": F32 * L, "rm_scores": F32 * L,
    "old_log_probs": F32 * L, "ref_log_prob": F32 * L, "advantages": F32 * L, "returns": F32 * L,
    "values_resp": F32 * L, "token_level_rewards": F32 * L, "uid": 64, "num_turns": 16,
}
sites = [  # (name, GET fields, PUT fields) - GRPO default: no critic, calculate_log_probs=True
    ("old_log_prob", ["entropy", "log_probs", "response_mask", "responses", "rollout_log_probs"], ["old_log_probs", "entropy"]),
    ("ref", ["log_probs", "response_mask"], ["ref_log_prob"]),
    ("adv", ["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob"], ["advantages", "returns"]),
    ("metrics", ["prompts", "responses", "response_mask", "advantages", "returns", "rm_scores", "num_turns"], []),
]
def cost(get, put):
    gb = sum(fields[f] for f in get) * N
    pb = sum(fields[f] for f in put) * N
    t = len(get) * N * FIX_GET + gb / BW + len(put) * N * FIX_PUT + pb / BW
    return gb, pb, t

tot_g = tot_p = tot_t = 0.0
print(f"N={N} P={P:.0f} L={L:.0f}  (S={S:.0f})")
for name, g, p in sites:
    gb, pb, t = cost(g, p)
    tot_g += gb; tot_p += pb; tot_t += t
    print(f"{name:13s} GET {gb/1e6:6.1f} MB ({len(g)} fields)  PUT {pb/1e6:5.1f} MB ({len(p)} fields)  ~{t:5.2f} s")
print(f"{'TOTAL':13s} GET {tot_g/1e6:6.1f} MB            PUT {tot_p/1e6:5.1f} MB            ~{tot_t:5.2f} s/step")

# K6 (a): worker returns response-sliced old_log_probs/ref_log_prob -> old_log_prob & ref GET+PUT disappear
# except the driver still needs responses+rollout_log_probs+old_log_probs for calculate_debug_metrics and entropy
# metric (moved to worker). (b): metrics GET drops prompts/responses (lengths from tags) and is merged with adv GET.
after = [
    ("old_log_prob*", [], []),
    ("ref*", [], []),
    ("adv+metrics*", ["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob", "num_turns"], ["advantages", "returns"]),
]
a_t = sum(cost(g, p)[2] for _, g, p in after)
a_g = sum(cost(g, p)[0] for _, g, p in after)
print(f"after K6(a)+(b): GET {a_g/1e6:.1f} MB  ~{a_t:.2f} s/step  -> saving ~{tot_t - a_t:.2f} s/step")
full_seq_bytes = N * (fields["log_probs"] + fields["entropy"] + fields["log_probs"])  # old: 2 full-seq, ref: 1
print(f"share of GET bytes that are full-sequence log_probs/entropy: {full_seq_bytes / tot_g:.0%}; "
      f"prompt-position waste of those: {P / S:.0%}")
step = 31.15
print(f"vs BASELINE-A step {step} s (gen 12.77, old_log_prob 3.49, ref 3.43, adv 0.61, update_actor 8.71): "
      f"TQ chain {tot_t / step:.1%} of step, K6 saving {(tot_t - a_t) / step:.1%}; poll_interval mean idle 1.0 s = {1.0 / step:.1%}")
