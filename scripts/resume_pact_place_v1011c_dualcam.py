"""Explicit recovery of PACT's recorded thread-creation failure, not a fresh run."""
from pact_place_v1011c_dualcam import *
import subprocess
import time
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def main():
    directory = TRAIN/'pact_seed3103'
    archive = WORK/'recovery_01'
    archive.mkdir(exist_ok=False)
    failure = read(WORK/'models/pact_seed3103.json')
    assert failure['returncode']==1 and not failure['verified']
    assert "RuntimeError: can't start new thread" in failure['output_tail']
    bundle_path = directory/'resume_bundle.ckpt'
    bundle = torch.load(bundle_path,map_location='cpu',weights_only=False)
    epoch,step = int(bundle['epoch']),int(bundle['global_step'])
    assert (epoch,step)==(1400,14010)
    lines = (directory/'epoch_log.jsonl').read_text().splitlines(keepends=True)
    rows = [json.loads(x) for x in lines]
    assert [r['epoch'] for r in rows] == list(range(len(rows)))
    assert rows[-1]['global_step']==14230
    for name in ('resume_bundle.ckpt','run_manifest.json','dataset_stats.pkl'):
        os.link(directory/name,archive/name)
    os.replace(directory/'epoch_log.jsonl',archive/'epoch_log_before_resume.jsonl')
    with (directory/'epoch_log.jsonl').open('x') as stream:
        stream.writelines(lines[:epoch+1])
    os.replace(WORK/'models/pact_seed3103.json',archive/'initial_training_failure.json')
    os.replace(WORK/'pilot_failure.json',archive/'initial_completion_supervisor_failure.json')
    freeze(archive/'recovery_plan.json',{**empty_authorization(),'failure_returncode':1,
        'failure_output':"RuntimeError: can't start new thread",'resume_epoch':epoch,'resume_global_step':step,
        'previous_complete_log_updates':14230,'complete_updates_replayed':14230-step,
        'partial_failed_epoch_updates_not_committed':True,'segment_targets':[20000,30000],
        'resume_bundle_sha256':sha256_file(archive/'resume_bundle.ckpt'),
        'optimizer_and_rng_restored':True,'dataset_split_normalization_and_4_loader_workers_unchanged':True,
        'extra_explicit_torch_intra_and_interop_caps':1,
        'cause_scope':'Thread creation failed. Current cgroup has headroom and a nonzero historical pids-limit counter; the exact thread owner at failure was not captured. Renew processes at safe checkpoint boundaries and record live thread counts.',
        'cgroup_pids_current':int(Path('/sys/fs/cgroup/pids.current').read_text()),
        'cgroup_pids_events':Path('/sys/fs/cgroup/pids.events').read_text()})
    del bundle
    command = list(read(WORK/'training_preflight.json')['commands']['pact'])
    command[1] = str(ROOT/'scripts/resume_pact_place_v1011c_dualcam_worker.py')
    phases = []
    for target in (20000,30000):
        current = command+['--resume','--max_steps',str(target)]
        log = WORK/'logs'/f'pact_resume_to_{target}.log'
        started = time.time()
        with log.open('x') as out:
            child = subprocess.Popen(current,cwd=ROOT/'submodules/act',env=environment(),
                stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            freeze(archive/f'launch_to_{target}.json',{'pid':child.pid,'command':current,'started_unix':started})
            code = child.wait()
        receipt = {**empty_authorization(),'returncode':code,'target_global_step':target,
            'elapsed_seconds':time.time()-started,'command':current,'log':str(log.relative_to(ROOT))}
        if code:
            receipt['output_tail'] = log.read_text()[-8000:]
        freeze(archive/f'exit_to_{target}.json',receipt)
        print(json.dumps(receipt),flush=True)
        assert code==0, receipt
        end = torch.load(bundle_path,map_location='cpu',weights_only=False)
        assert end['global_step']==target and end['epoch']==target//10-1
        del end
        if target==20000:
            os.link(bundle_path,archive/'resume_bundle_at_20000.ckpt')
        phases.append(receipt)
    epochs = [json.loads(x) for x in (directory/'epoch_log.jsonl').read_text().splitlines() if x.strip()]
    assert [r['epoch'] for r in epochs]==list(range(3000))
    assert [r['global_step'] for r in epochs]==list(range(10,30001,10))
    manifest = read(directory/'run_manifest.json')
    assert manifest['policy_config']['camera_names']==CAMERAS and manifest['policy_config']['num_queries']==100
    milestones = {}
    for target in MILESTONES:
        receipt = read(directory/f'policy_step_{target}.json')
        assert sha256_file(directory/f'policy_step_{target}.ckpt')==receipt['checkpoint_sha256']
        view = TRAIN/f'pact_seed3103_step{target}'
        view.mkdir(exist_ok=False)
        for name,source in {'policy_best.ckpt':f'policy_step_{target}.ckpt','dataset_stats.pkl':'dataset_stats.pkl',
                            'run_manifest.json':f'policy_step_{target}_run_manifest.json'}.items():
            os.link(directory/source,view/name)
        assert sha256_file(view/'run_manifest.json')==receipt['run_manifest_sha256']
        freeze(view/'checkpoint_step.json',receipt)
        milestones[str(target)] = receipt
    result = {**empty_authorization(),'arm':'pact','seed':3103,'returncode':0,'verified':True,
        'epochs_recorded':3000,'updates_recorded':30000,'milestones':milestones,
        'dataset_stats_sha256':sha256_file(directory/'dataset_stats.pkl'),
        'initial_attempt_returncode':1,'recovery_phases':phases,
        'initial_failure_receipt':str((archive/'initial_training_failure.json').relative_to(ROOT)),
        'elapsed_seconds':failure['elapsed_seconds']+sum(r['elapsed_seconds'] for r in phases)}
    freeze(WORK/'models/pact_seed3103.json',result)
    act = read(WORK/'models/act_seed3103.json')
    assert act['dataset_stats_sha256']==result['dataset_stats_sha256']
    freeze(WORK/'training_completion.json',{**empty_authorization(),'complete':True,'models':{'act':act,'pact':result}})
    print('PACT RECOVERED: retained state reaches 30000 updates, original failure preserved.',flush=True)


if __name__=='__main__':
    main()
