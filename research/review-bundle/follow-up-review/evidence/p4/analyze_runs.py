"""Read-only audit of the saved final P4 runs. No model execution or network calls."""
import hashlib
import json
import pathlib
import re
import statistics

ROOT = pathlib.Path('<repo>')
RESULTS = ROOT / 'bench/results'
RUNS = ['p4-enabled-r2-final', 'p4-control-r2-final',
        'p4-curve-enabled-s1', 'p4-curve-control-s1',
        'p4-curve-enabled-s2', 'p4-curve-control-s2']


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


audit = {}
for run in RUNS:
    folder = RESULTS / run
    meta = json.loads((folder / 'meta.json').read_text())
    metrics = read_jsonl(folder / 'metrics.jsonl')
    train = [row for row in metrics if 'timing_s/step' in row['data']]
    record = {
        'src_head': meta['cfg']['src_head'],
        'digest': meta['tree']['digest'],
        'digest_match_reported': meta['tree']['digest_match'],
        'train_steps': len(train),
        'span_rows': sorted({row['data'].get('adv_estimator/rows_with_turn_spans', -1) for row in train}),
        'mean_step_seconds_excluding_first': statistics.mean(row['data']['timing_s/step'] for row in train[1:]),
    }
    last_validation = folder / 'val/30.jsonl'
    if last_validation.exists():
        rows = read_jsonl(last_validation)
        repeated, calls_total, rewards_at_least_two = 0, 0, 0
        for row in rows:
            answers = []
            for raw in re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', row['output'], flags=re.S):
                try:
                    call = json.loads(raw)
                    answers.append(str(call.get('arguments', {}).get('answer')))
                except (ValueError, AttributeError):
                    pass
            calls_total += len(answers)
            repeated += len(answers) > len(set(answers))
            rewards_at_least_two += row.get('turn_rewards_sum', 0) >= 2
        record.update(validation_rows=len(rows),
                      accuracy=statistics.mean(row['acc'] for row in rows),
                      parsed_tool_calls=calls_total,
                      trajectories_with_repeated_identical_answer=repeated,
                      trajectories_with_step_reward_at_least_two=rewards_at_least_two)
    audit[run] = record

# Independently reproduce the remote manifest digest from the current reviewed
# product tree plus the three explicitly shipped benchmark-only extras.
folder = RESULTS / RUNS[0]
digest = hashlib.sha256()
for rel in sorted((folder / 'manifest.txt').read_text().splitlines()):
    path = ROOT / 'wt/p4b' / rel
    if rel.startswith('bench_extras/'):
        path = ROOT / 'bench/gsm8k_tool' / rel.split('/', 1)[1]
    digest.update(f'{rel}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n'.encode())
audit['independent_manifest_digest'] = digest.hexdigest()
audit['independent_manifest_matches'] = digest.hexdigest() == (folder / 'digest.txt').read_text().strip()
print(json.dumps(audit, indent=2))
