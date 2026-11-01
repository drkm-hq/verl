#!/usr/bin/env python3
"""Chaos: ray.kill one actor of a given class in the running (local) Ray cluster.

Run inside the trainer's venv from a side process:
    uv run --frozen --all-packages --extra vllm --extra fsdp python3 /root/chaos_kill.py --class-name AgentLoopWorkerTQ \
        [--wait-for-file /tmp/x.trigger --ready-file /tmp/x.ready]

Finds the actor through ray.util.list_named_actors(all_namespaces=True) joined with
ray._private.state.actors() (class name, pid), gets the handle by name+namespace and
ray.kill()s it (no_restart=True). With --wait-for-file the process connects to Ray, resolves
the victim's handle (the GCS listing takes ~8 s), touches --ready-file and only then waits for
the trigger file, so the kill lands within ~0.1 s of the harness's decision. Prints one
`CHAOS_KILL {...}` JSON line.
"""

import argparse
import json
import logging
import os
import time


def find_actors(ray, class_name: str) -> tuple[list[dict], list[str]]:
    named = {(x["name"], x["namespace"]) for x in ray.util.list_named_actors(all_namespaces=True)}
    by_name = {}
    for actor_id, info in ray._private.state.actors().items():
        if info.get("State") not in ("ALIVE",):
            continue
        by_name[info.get("Name") or actor_id] = {"actor_id": actor_id, "class_name": info.get("ActorClassName"),
                                                  "pid": info.get("Pid"), "state": info.get("State")}
    matches = [{"name": name, "namespace": ns, **by_name[name]} for name, ns in sorted(named)
               if name in by_name and by_name[name]["class_name"] == class_name]
    return matches, sorted({v["class_name"] for v in by_name.values()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-name", default="AgentLoopWorkerTQ")
    ap.add_argument("--index", type=int, default=0, help="which matching actor (sorted by name) to kill")
    ap.add_argument("--address", default="auto")
    ap.add_argument("--wait-for-file", default=None, help="arm now (connect + resolve the handle), fire when this file appears")
    ap.add_argument("--ready-file", default=None, help="touched once the handle is resolved")
    ap.add_argument("--connect-retry-s", type=float, default=900)
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()
    import ray

    t_connect = time.time()
    while True:
        try:
            ray.init(address=a.address, ignore_reinit_error=True, log_to_driver=False, logging_level=logging.ERROR)
            break
        except Exception:  # noqa: BLE001  (no cluster yet)
            if not a.wait_for_file or time.time() - t_connect > a.connect_retry_s:
                raise
            time.sleep(2)
    out = {"class_name": a.class_name, "connect_s": round(time.time() - t_connect, 2)}
    t_list = time.time()
    matches, classes = find_actors(ray, a.class_name)
    out.update({"n_alive": len(matches), "all_alive_classes": classes, "list_s": round(time.time() - t_list, 2)})
    if a.list_only or not matches:
        out["killed"] = None
        print("CHAOS_KILL " + json.dumps(out), flush=True)
        return 0 if matches else 2
    victim = matches[min(a.index, len(matches) - 1)]
    handle = ray.get_actor(victim["name"], namespace=victim["namespace"])
    out["armed_unix"] = time.time()
    if a.ready_file:
        open(a.ready_file, "w").close()
    if a.wait_for_file:
        print("CHAOS_READY " + json.dumps(out), flush=True)
        while not os.path.exists(a.wait_for_file):
            time.sleep(0.02)
        out["trigger_unix"] = time.time()
    t0 = time.time()
    ray.kill(handle, no_restart=True)
    out.update({"killed": victim, "kill_unix": t0, "kill_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                "kill_call_s": round(time.time() - t0, 3), "survivors": [m["name"] for m in matches if m is not victim]})
    if "trigger_unix" in out:
        out["trigger_to_kill_s"] = round(t0 - out["trigger_unix"], 3)
    # confirm the process is gone (ray.kill is async at the GCS; the pid vanishes within a second or two)
    pid = victim.get("pid")
    gone = None
    for _ in range(50):
        try:
            os.kill(int(pid), 0)
            time.sleep(0.1)
        except (ProcessLookupError, TypeError, ValueError):
            gone = round(time.time() - t0, 2)
            break
        except PermissionError:
            break
    out["pid_gone_after_s"] = gone
    print("CHAOS_KILL " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
