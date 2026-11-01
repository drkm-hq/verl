import json, re, sys
import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))  # the repository root
# usage: ghsearch.py <prs|issues|merged> <regex> [max]
kind = sys.argv[1]; pat = re.compile(sys.argv[2], re.I); mx = int(sys.argv[3]) if len(sys.argv) > 3 else 25
f = {'prs': 'prs_open.json', 'issues': 'issues_open.json', 'merged': 'prs_merged_recent.json'}[kind]
d = json.load(open(os.path.join(_REPO, "gh/") + f))
hits = []
for it in d:
    t = it.get('title') or ''; b = it.get('body') or ''
    if pat.search(t) or pat.search(b):
        hits.append(it)
print(f"{len(hits)} hits for /{sys.argv[2]}/ in {kind}")
for it in hits[:mx]:
    t = it.get('title') or ''; b = (it.get('body') or '').replace('\n', ' ')
    m = pat.search(t) or pat.search(b)
    ctx = ''
    if m and not pat.search(t):
        s = max(0, m.start() - 60); ctx = ' ... ' + b[s:m.end() + 80]
    extra = ''
    if 'reactionGroups' in it:
        n = sum(r.get('users', {}).get('totalCount', 0) for r in it.get('reactionGroups') or [])
        c = len(it.get('comments') or [])
        extra = f' [react={n} comments={c}]'
    if 'isDraft' in it and it['isDraft']:
        extra += ' [DRAFT]'
    print(f"#{it['number']} {it.get('createdAt','')[:10]} {t[:110]}{extra}{ctx[:160]}")
