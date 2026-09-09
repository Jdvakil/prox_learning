"""Read-only snapshot invoked by the root agent for manual run reviews."""
import datetime
import hashlib
import json
import pathlib
import time

C = pathlib.Path(__file__).resolve().parents[1]
ROOT = C.parents[1]


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def last(path):
    if not path.exists():
        return None
    rows = path.read_text().splitlines()
    return json.loads(rows[-1]) if rows else None


status = read(C / 'run_status.json')
progress = read(C / 'progress.json')
contract = read(C / 'contract.json')
workers = []
for launch in C.glob('attempts/**/launch.json'):
    info = read(launch)
    receipt = read(launch.parent / 'exit_receipt.json')
    if receipt is not None:
        continue
    proc = pathlib.Path('/proc') / str(info['pid'])
    item = {'id': info['id'], 'pid': info['pid'], 'alive': proc.exists()}
    if proc.exists():
        fields = (proc / 'stat').read_text().split()
        item.update(start_ticks_match=fields[21] == info['proc_start_ticks'],
                    state=fields[2], cpu_ticks=int(fields[13]) + int(fields[14]))
    workers.append(item)
checks = {}
for name, expected in contract['code_hashes'].items():
    checks[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
eval_path = ROOT / 'EVAL.md'
checks['original_EVAL.md'] = hashlib.sha256(eval_path.read_bytes()).hexdigest() == contract['input_hashes'][str(eval_path)]
snapshot = {
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'reviewer': 'root_agent_manual_invocation',
    'status': status,
    'progress': progress,
    'progress_age_s': round(time.time() - (C / 'progress.json').stat().st_mtime, 1),
    'workers_without_exit_receipts': workers,
    'training': read(C / 'checkpoint/progress.json'),
    'checkpoint': read(C / 'checkpoint/checkpoint_progress.json'),
    'latest_training_resources': last(C / 'training_resources.jsonl'),
    'valid_jobs': len((C / 'valid_ledger.jsonl').read_text().splitlines()) if (C / 'valid_ledger.jsonl').exists() else 0,
    'invalid_jobs': len((C / 'invalid_ledger.jsonl').read_text().splitlines()) if (C / 'invalid_ledger.jsonl').exists() else 0,
    'all_reviewed_code_and_original_report_hashes_match': all(checks.values()),
    'failed_hash_checks': [key for key, value in checks.items() if not value],
}
with (C / 'root_review/manual_checks.jsonl').open('a') as stream:
    stream.write(json.dumps(snapshot, sort_keys=True) + '\n')
print(json.dumps(snapshot, indent=2))
