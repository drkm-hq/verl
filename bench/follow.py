#!/usr/bin/env python3
"""Follow a Modal app's logs for a bounded time, printing only new lines.

    python3 bench/follow.py <app-id> [--seconds 230] [--every 15] [--tail 400] [--grep PATTERN] [--until PATTERN]

`modal app logs` returns immediately while an app is still building its image and streams
only once containers run, so this polls it every --every seconds for up to --seconds and prints
lines not seen in the previous poll. Exits early when --until matches a line.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

MODAL = Path(__file__).resolve().parent.parent / ".venv-bench" / "bin" / "modal"


def fetch(app_id: str, tail: int) -> list[str]:
    env = {**os.environ}
    p = subprocess.run(["perl", "-e", "alarm shift; exec @ARGV", "40", str(MODAL), "app", "logs", app_id, "--tail", str(tail)],
                       env=env, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    return [ln.rstrip() for ln in out.replace("\r", "\n").splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("app_id")
    ap.add_argument("--seconds", type=int, default=230)
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--tail", type=int, default=400)
    ap.add_argument("--grep", default=None, help="only print lines matching this regex")
    ap.add_argument("--until", default=None, help="stop as soon as a line matches this regex")
    ap.add_argument("--max-lines", type=int, default=60, help="print at most this many new lines per poll")
    a = ap.parse_args()
    seen: list[str] = []
    t_end = time.time() + a.seconds
    first = True
    while True:
        lines = fetch(a.app_id, a.tail)
        # new lines = suffix of `lines` after the longest overlap with the previous poll
        new = lines
        if seen:
            for k in range(min(len(seen), len(lines)), 0, -1):
                if seen[-k:] == lines[:k]:
                    new = lines[k:]
                    break
            else:
                new = [ln for ln in lines if ln not in set(seen)]
        if first:
            new = lines[-a.max_lines:]
            first = False
        shown = [ln for ln in new if not a.grep or re.search(a.grep, ln)]
        for ln in shown[-a.max_lines:]:
            print(ln, flush=True)
        seen = lines
        if a.until and any(re.search(a.until, ln) for ln in lines):
            print(f"[follow] matched --until {a.until!r}", flush=True)
            return 0
        if time.time() + a.every > t_end:
            print(f"[follow] {a.seconds}s elapsed; last line: {lines[-1] if lines else '(none)'}", flush=True)
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
