"""Read-only health review, invoked manually by the root agent each hour."""
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'diagnostics_output/pact_place_v1010c_readout_s3_v1'
def read(path):return json.loads(path.read_text()) if path.exists() else None
def last(path):
    rows=path.read_text().splitlines() if path.exists() else []
    return json.loads(rows[-1]) if rows else None
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    seeds={}
    for seed in (3104,3105):
        c=D/f'seed{seed}';contract=read(c/'contract.json');workers=[]
        for launch in c.glob('attempts/**/launch.json'):
            if (launch.parent/'exit_receipt.json').exists():continue
            info=read(launch);proc=Path('/proc')/str(info['pid'])
            record={'id':info['id'],'pid':info['pid'],'alive':proc.exists()}
            try:
                fields=(proc/'stat').read_text().split()
                record.update(start_ticks_match=fields[21]==info['proc_start_ticks'],
                    state=fields[2],cpu_ticks=int(fields[13])+int(fields[14]))
            except FileNotFoundError:pass
            workers.append(record)
        checked={} if contract is None else {p:sha(ROOT/p)==expected for p,expected in contract['code_hashes'].items()}
        seeds[str(seed)]={'status':read(c/'run_status.json'),'progress':read(c/'progress.json'),
            'training':read(c/'checkpoint/progress.json'),'checkpoint':read(c/'checkpoint/checkpoint_progress.json'),
            'latest_training_resources':last(c/'training_resources.jsonl'),'workers':workers,
            'valid_exits':len((c/'valid_ledger.jsonl').read_text().splitlines()) if (c/'valid_ledger.jsonl').exists() else 0,
            'invalid_exits':len((c/'invalid_ledger.jsonl').read_text().splitlines()) if (c/'invalid_ledger.jsonl').exists() else 0,
            'code_hashes_passed':all(checked.values()),'failed_hashes':[p for p,v in checked.items() if not v]}
    cg=Path('/sys/fs/cgroup')
    resources={name:int((cg/name).read_text()) for name in ('memory.current','memory.max','pids.current','pids.max')}
    resources.update(disk_free_bytes=shutil.disk_usage(D).free,
        memory_events=dict(line.split() for line in (cg/'memory.events').read_text().splitlines()))
    import pynvml
    pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByIndex(0)
    memory=pynvml.nvmlDeviceGetMemoryInfo(handle,version=pynvml.nvmlMemory_v2)
    resources.update(vram_used_bytes=memory.used,vram_total_bytes=memory.total)
    snapshot={'utc':datetime.now(timezone.utc).isoformat(),'reviewer':'root_agent_manual_invocation',
        'pipeline_status':read(D/'run_status.json'),'pipeline_progress':read(D/'progress.json'),
        'seeds':seeds,'resources':resources,'original_eval_sha256':sha(ROOT/'EVAL.md')}
    with (D/'manual_checks.jsonl').open('a') as f:f.write(json.dumps(snapshot,sort_keys=True)+'\n')
    print(json.dumps(snapshot,indent=2))


if __name__=='__main__':main()
