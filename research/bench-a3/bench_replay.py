"""V1 replay buffer polling cost without a live TransferQueue: simulate the dict `tq.kv_list()` returns and time
`ReplayBuffer._sync_metadata_from_transfer_queue` (replay_buffer.py:201-237) + `_materialize_batch` (394-406)
+ the key-parsing loops of compute_advantage_for_multi_trajectories (v1/utils.py:171-191) and _compute_metrics.

In async mode TQ holds every in-flight / finished trajectory: with train_batch_size B prompts, rollout.n, and
max_inflight_gen_batches g, TQ holds ~ g*B prompt tags + g*B*n trajectory tags.
"""

import sys
import time
import uuid
from collections import defaultdict

sys.path.insert(0, os.path.join(_REPO, "research/bench-a3"))
from bench_common import machine_banner, timeit  # noqa: E402
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))  # the repository root

machine_banner()


def make_kv_list(num_prompts, n, finished_frac=0.6):
    items = {}
    for p in range(num_prompts):
        uid = str(uuid.uuid4())
        status = "finished" if p < finished_frac * num_prompts else "running"
        items[uid] = {"is_prompt": True, "status": status, "global_steps": p // 128}
        if status == "finished":
            for s in range(n):
                items[f"{uid}_{s}_0"] = {
                    "status": "success",
                    "prompt_len": 1000,
                    "response_len": 3000,
                    "seq_len": 4000,
                    "global_steps": p // 128,
                    "min_global_steps": p // 128,
                    "max_global_steps": p // 128,
                }
    return {"train": items}


def sync_metadata(data):
    """Verbatim logic of ReplayBuffer._sync_metadata_from_transfer_queue."""
    partitions = defaultdict(dict)
    pending, running, finished, failure = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    prompt_global_steps = defaultdict(dict)
    for partition_id, items in data.items():
        partition = partitions[partition_id]
        for key, tag in items.items():
            if tag.get("is_prompt", False):
                prompt_global_steps[partition_id][key] = tag["global_steps"]
                match tag["status"]:
                    case "pending":
                        pending[partition_id].add(key)
                    case "running":
                        running[partition_id].add(key)
                    case "finished":
                        finished[partition_id].add(key)
                    case "failure":
                        failure[partition_id].add(key)
            else:
                if key not in partition:
                    partition[key] = {}
                partition[key].update(tag)
    return partitions, finished


for num_prompts, n in [(512, 8), (2048, 8), (8192, 16)]:
    data = make_kv_list(num_prompts, n)
    nkeys = len(data["train"])
    print(f"\n## TQ holds {num_prompts} prompts x n={n}: {nkeys} keys")
    timeit(lambda: sync_metadata(data), label="_sync_metadata_from_transfer_queue (python rebuild per poll)")
    partitions, finished = sync_metadata(data)
    partition_snapshot = dict(partitions["train"])
    selected = set(sorted(finished["train"])[:512])

    def materialize():
        keys, tags = [], []
        for key, tag in partition_snapshot.items():
            uid = key.split("_")[0]
            if uid in selected:
                keys.append(key)
                tags.append(tag)
        return keys, tags

    timeit(materialize, label="_materialize_batch (split every key in TQ)")
    keys, tags = materialize()

    def parse_keys():
        final_sessions = {}
        row_session_keys = []
        for i, key in enumerate(keys):
            fields = key.rsplit("_", 2)
            uid, session_id, index = fields[0], fields[1], int(fields[2])
            session_key = f"{uid}_{session_id}"
            row_session_keys.append(session_key)
            if session_key not in final_sessions or final_sessions[session_key][0] < index:
                final_sessions[session_key] = (index, i)
        return final_sessions

    timeit(parse_keys, label=f"compute_advantage_for_multi_trajectories key loop over {len(keys)} rows")
    # the pickle size of the kv_list payload the controller ships to the driver on every poll
    import pickle

    blob = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"kv_list payload pickled: {len(blob) / 1e6:.1f} MB for {nkeys} keys")
