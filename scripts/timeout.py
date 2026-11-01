#!/usr/bin/env python3
"""coreutils-style `timeout` for macOS (no GNU coreutils here).

Usage: timeout.py SECONDS CMD [ARG...]
Runs CMD in its own process group; on expiry sends SIGTERM to the whole group,
then SIGKILL 10 s later, and exits 124 (like GNU timeout). Otherwise exits with
CMD's exit status. Killing the group matters for pytest runs that start Ray:
a bare alarm/SIGALRM would leave raylet/gcs/worker children running.
"""
import os
import signal
import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 3:
        sys.stderr.write(__doc__)
        return 2
    seconds = float(sys.argv[1])
    proc = subprocess.Popen(sys.argv[2:], start_new_session=True)
    try:
        return proc.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"timeout.py: {seconds:g}s elapsed, terminating process group {proc.pid}\n")
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        return 124


if __name__ == "__main__":
    sys.exit(main())
