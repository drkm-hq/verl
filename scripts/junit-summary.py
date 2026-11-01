#!/usr/bin/env python3
"""Summarise pytest junit XML files: counts per file plus failing/erroring test ids.

Usage: junit-summary.py research/junit/*.xml
"""
import sys
import xml.etree.ElementTree as ET


def main(paths):
    for path in paths:
        root = ET.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root)
        tests = fails = errs = skips = 0
        bad = []
        for s in suites:
            tests += int(s.get("tests", 0)); fails += int(s.get("failures", 0))
            errs += int(s.get("errors", 0)); skips += int(s.get("skipped", 0))
            for tc in s.iter("testcase"):
                for kind in ("failure", "error"):
                    el = tc.find(kind)
                    if el is not None:
                        msg = (el.get("message") or el.text or "").strip().splitlines()
                        bad.append((kind, f"{tc.get('classname')}::{tc.get('name')}", msg[0][:160] if msg else ""))
        print(f"{path}: tests={tests} passed={tests - fails - errs - skips} failed={fails} errors={errs} skipped={skips}")
        for kind, tid, msg in bad:
            print(f"  {kind.upper():7} {tid}\n          {msg}")


if __name__ == "__main__":
    main(sys.argv[1:])
