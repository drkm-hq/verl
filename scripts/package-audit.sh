#!/usr/bin/env bash
# Audit every contribution package: branch state, commit count, cleanliness, required files,
# whether diff.patch matches the branch, and whether the package's branch.txt names the current HEAD.
cd "$(dirname "$0")/.." || exit 1
req=(PR-TITLE.txt PR-BODY.md branch.txt diff.patch tests.log CHECKLIST.md NOTES.md REVIEW.md)
printf "%-5s %-32s %-9s %-4s %-5s %-6s %-7s %s\n" pkg branch head n dirty patch brtxt missing
for p in p0 p1 p2 p3 p3a p3b p4 p4a p4b; do
  wt="wt/$p"; pk="packages/$p"
  [ -d "$wt" ] || { printf "%-5s (no worktree)\n" "$p"; continue; }
  br=$(git -C "$wt" branch --show-current); head=$(git -C "$wt" rev-parse --short HEAD)
  n=$(git -C "$wt" log --oneline main..HEAD | wc -l | tr -d ' '); dirty=$(git -C "$wt" status --porcelain | wc -l | tr -d ' ')
  miss=""; for f in "${req[@]}"; do [ -f "$pk/$f" ] || miss="$miss $f"; done
  case "$p" in p3b) base=contrib/p3a-inflight-ttl;; p4b) base=contrib/p4a-turn-signals;; *) base=main;; esac
  if [ -f "$pk/diff.patch" ]; then
    a=$(git -C "$wt" diff "$base"...HEAD | shasum | cut -c1-8); b=$(shasum < "$pk/diff.patch" | cut -c1-8)
    [ "$a" = "$b" ] && patch=fresh || patch=STALE
  else patch=none; fi
  if [ -f "$pk/branch.txt" ] && grep -q "$(git -C "$wt" rev-parse HEAD | cut -c1-8)" "$pk/branch.txt"; then brtxt=ok; else brtxt=STALE; fi
  printf "%-5s %-32s %-9s %-4s %-5s %-6s %-7s %s\n" "$p" "$br" "$head" "$n" "$dirty" "$patch" "$brtxt" "${miss:-none}"
done
echo "main clone dirty lines: $(git -C verl status --porcelain | wc -l | tr -d ' ')  branch: $(git -C verl branch --show-current)"
echo "remotes with any contrib branch pushed: $(git -C verl branch -r | grep -c contrib || true)"
echo "ray processes: $(pgrep -fl 'raylet|gcs_server' | wc -l | tr -d ' ')"
