"""Parent-observed bounded stage execution; no separate monitoring task."""
from __future__ import annotations
import argparse
import fcntl
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pact_v1010b_contract import *


class StopExecution(RuntimeError):
    def __init__(self,reason,status='INCOMPLETE_INFRASTRUCTURE_OR_BUDGET'):
        super().__init__(reason);self.status=status


def process_start(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,IndexError):return None


def resources():
    cg=Path('/sys/fs/cgroup')
    values={k:int((cg/k).read_text()) for k in ('memory.current','memory.max','pids.current','pids.max')}
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    used,total,util=map(float,gpu.stdout.strip().splitlines()[0].split(','))
    events=dict(line.split() for line in (cg/'memory.events').read_text().splitlines())
    stat=dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
    return {'utc':now(),'ram_fraction':values['memory.current']/values['memory.max'],
        'pid_fraction':values['pids.current']/values['pids.max'],'vram_fraction':used/total,
        'vram_used_mib':used,'vram_total_mib':total,'gpu_utilization':util,'cgroup':values,
        'oom_kill':int(events['oom_kill']),'memory_file_bytes':int(stat['file']),
        'memory_anon_bytes':int(stat['anon']),'disk_free_bytes':shutil.disk_usage(B).free}


def release_cache(reason):
    before=resources()['cgroup']['memory.current'];count=total=0
    candidates=[]
    for root in (W/'raw',W/'converted',W/'evaluation',B/'rollouts'):
        for suffix in ('*.h5','*.hdf5'):
            for path in root.rglob(suffix):
                if root==W/'converted' or (path.parent/'exit_receipt.json').exists():candidates.append(path)
    for folder in (W/'checkpoints').glob('*'):
        if not (folder/'progress.json').exists() or read(folder/'progress.json')['updates']!=60000:continue
        for name in ('resume_bundle.ckpt','policy_best.ckpt','policy_last.ckpt','policy_update_30000.ckpt'):
            if (folder/name).is_file():candidates.append(folder/name)
    for path in candidates:
        if path.is_symlink():continue
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            size=os.fstat(fd).st_size;os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED);count+=1;total+=size
        finally:os.close(fd)
    append(B/'resources/cache_releases.jsonl',{'utc':now(),'reason':reason,'files':count,'file_bytes':total,
        'memory_before':before,'memory_after':int(Path('/sys/fs/cgroup/memory.current').read_text()),
        'operation':'POSIX_FADV_DONTNEED; historical/new file bytes unchanged'})


def storage_gate(stage):
    contract=read(B/'contract.json');valid=lines(B/'valid_ledger.jsonl')
    actual=[sum(p.stat().st_size for p in Path(r['directory']).iterdir() if p.is_file()) for r in valid if r.get('kind')=='rollout']
    # Instrumented clips are separately budgeted; standard size is measured from mandatory standard files.
    sizes=[]
    for row in valid:
        if row.get('kind')!='rollout':continue
        directory=Path(row['directory'])
        sizes.append(sum((directory/n).stat().st_size for n in ('trajectory.h5','telemetry.h5','initial_observation.h5','actions.npz','result.json')))
    per=max(34*MIB,max(sizes,default=0)*1.03) if len(sizes)<12 else max(float(sum(sizes)/len(sizes))*1.05,max(sizes)*1.02)
    completed=len(sizes);remaining=max(0,1056-completed)
    completed_models=len(list((B/'checkpoints').glob('*/completed.json')))
    model_remaining=max(0,12-completed_models)*(5*GIB/12)
    instrument_remaining=max(0,12-sum(r.get('stage')=='A1' for r in valid))*(GIB/12)
    projected=remaining*per+model_remaining+instrument_remaining
    free=shutil.disk_usage(B).free
    report={'utc':now(),'stage':stage,'free_bytes':free,'remaining_rollouts':remaining,
        'standard_per_rollout_bytes':per,'observed_n':len(sizes),'observed_mean_bytes':sum(sizes)/len(sizes) if sizes else None,
        'observed_max_bytes':max(sizes,default=None),'remaining_model_bytes':model_remaining,
        'remaining_instrument_bytes':instrument_remaining,'remaining_required_artifacts':projected,
        'reserve_bytes':10*GIB,'passed':free-projected>=10*GIB,
        'temporary_peak_training_note':'Serial training: at most2 atomic generation files, each<1.5GiB, before final rollout accumulation.'}
    append(B/'resources/storage_gates.jsonl',report)
    if not report['passed']:raise StopExecution(f'disk reserve gate for{stage}: needs{(projected+10*GIB-free)/GIB:.3f} additional GiB')
    return report


def resolve_job(job):
    job=dict(job);job['schedule_job_sha256']=job.pop('job_sha256')
    job['schedule_contract_sha256']=job['contract_sha256'];job['contract_sha256']=read(B/'contract.json')['contract_sha256']
    if job['checkpoint_sha256'] is None:
        completed=read(Path(job['checkpoint_path']).parent/'completed.json')
        assert completed['completed_updates']==63000 and completed['strict_reload']
        job['checkpoint_sha256']=completed['checkpoint_sha256']
    job['job_sha256']=digest(job)
    return job


def worker_command(job):
    pairs={'arm':job['arm'],'training-seed':job['training_seed'],'scene-id':job['scene_id'],
        'scene-block':job['scene_block'],'checkpoint-path':job['checkpoint_path'],
        'checkpoint-sha256':job['checkpoint_sha256'],'manifest':job['manifest'],'variant':job['variant'],
        'instrumentation':job['instrumentation'],'output-dir':job['output_dir'],
        'job-id':job['job_id'],'job-sha256':job['job_sha256']}
    return [PYTHON,str(CODE/'pact_v1010b_eval_worker.py'),*[str(v) for k,val in pairs.items() for v in ('--'+k,val)]]


def validate_completion(job,directory,receipt):
    assert receipt['returncode']==0 and receipt['observed_by_parent'] and receipt['job_sha256']==job['job_sha256']
    result=read(directory/'result.json')
    assert result['status']=='complete' and result['rollout_id']==job['job_id']
    for key in ('checkpoint_sha256','training_seed','scene_id','scene_block','variant','physical_row_digest'):
        assert result[key]==job[key],key
    assert result['schedule_row_sha256']==job['job_sha256']
    if job['selected_seed'] is not None:assert result['seed']==job['selected_seed'] and result['sampling_retry_index']==job['selected_retry']
    from pact_v1010b_metrics import recompute
    metrics=recompute(directory)
    assert len(read(directory/'provenance.json'))>0
    return metrics


class Pool:
    def __init__(self,stage,workers=12):
        self.stage=stage;self.limit=workers;self.active={};self.next_sample=0.;self.pressure=0
        self.paused=None;self.oom=resources()['oom_kill'];self.durations=[]
        self.safe_stop=datetime.fromisoformat(read(B/'contract.json')['safe_stop_utc']).timestamp()
        self.deadline=datetime.fromisoformat(read(B/'contract.json')['deadline_utc']).timestamp()
        release_cache('before '+stage+' pool')

    def sample(self):
        if time.monotonic()<self.next_sample:return
        self.next_sample=time.monotonic()+60
        value=resources()
        if value['ram_fraction']>=.73:
            release_cache('minute sample at >=73% RAM');value=resources()
        value.update(stage=self.stage,active=len(self.active),limit=self.limit)
        append(B/'resources/samples.jsonl',value)
        if value['oom_kill']>self.oom:self.paused='new cgroup OOM kill observed';self.limit=min(self.limit,10)
        self.oom=value['oom_kill']
        high=value['vram_fraction']>.85 or value['ram_fraction']>.80 or value['pid_fraction']>.75
        self.pressure=self.pressure+1 if high else 0
        if self.pressure>=3:
            if self.limit>10:self.limit=10;self.pressure=0
            else:self.paused='persistent resource pressure at10 or lower workers'
            append(B/'resources/capacity_changes.jsonl',{'utc':now(),'stage':self.stage,'limit':self.limit,'reason':'three consecutive resource threshold samples'})
        atomic(B/'progress.json',{'utc':now(),'stage':self.stage,'active_jobs':[v['job']['job_id'] for v in self.active.values()],
            'worker_limit':self.limit,'paused':self.paused,'valid_rollouts':sum(r.get('kind')=='rollout' for r in lines(B/'valid_ledger.jsonl'))})
        print(json.dumps({'utc':now(),'stage':self.stage,'active':len(self.active),'limit':self.limit,'paused':self.paused}),flush=True)
        if value['disk_free_bytes']<10*GIB:self.paused='disk reserve crossed during execution'

    def launch(self,job,command,kind):
        directory=inside(job['output_dir']);directory.mkdir(parents=True,exist_ok=True)
        assert not (directory/'job.json').exists(), 'preserve partial attempt; explicit diagnosed recovery required'
        freeze(directory/'job.json',job)
        log=(directory/'worker.log').open('x')
        proc=subprocess.Popen(command,cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        start={'pid':proc.pid,'process_start_ticks':process_start(proc.pid),'command':command,'utc':now(),
            'job_sha256':job['job_sha256'],'parent_pid':os.getpid(),'parent_start_ticks':process_start(os.getpid())}
        self.active[proc.pid]={'process':proc,'job':job,'log':log,'started':time.monotonic(),'launch':start,'kind':kind}
        freeze(directory/'launch.json',start)

    def poll(self):
        for pid,item in list(self.active.items()):
            code=item['process'].poll()
            if code is None:continue
            code=item['process'].wait();item['log'].close();del self.active[pid]
            job=item['job'];directory=Path(job['output_dir']);elapsed=time.monotonic()-item['started']
            receipt={'schema':SCHEMA,'returncode':code,'observed_by_parent':True,'parent_pid':os.getpid(),
                'job_sha256':job['job_sha256'],'job_id':job['job_id'],'pid':pid,
                'process_start_ticks':item['launch']['process_start_ticks'],'utc':now(),'elapsed_s':elapsed}
            freeze(directory/'exit_receipt.json',receipt)
            try:
                assert code==0,f'worker exit{code}; see {directory}/worker.log'
                if item['kind']=='rollout':
                    metrics=validate_completion(job,directory,receipt);freeze(directory/'recomputed.json',metrics)
                    result_path=directory/'result.json'
                else:
                    result=read(directory/'completed.json');assert result['completed_updates']==63000
                    result_path=directory/'completed.json'
                record={'kind':item['kind'],'stage':job.get('stage',self.stage),'job_id':job['job_id'],'job_sha256':job['job_sha256'],
                    'directory':str(directory),'returncode':code,'elapsed_s':elapsed,'result_sha256':sha(result_path),
                    'exit_receipt_sha256':sha(directory/'exit_receipt.json'),'valid_completion':True}
                assert job['job_id'] not in {r['job_id'] for r in lines(B/'valid_ledger.jsonl')}
                append(B/'valid_ledger.jsonl',record);self.durations.append(elapsed)
                if item['kind']=='training':
                    temporary=directory/'resume_bundle.ckpt'
                    freeze(directory/'temporary_bundle_removal.json',{'sha256':sha(temporary),'bytes':temporary.stat().st_size,
                        'parent_observed_zero_exit':True,'completed_checkpoint_sha256':read(result_path)['checkpoint_sha256'],
                        'explicitly_new_run_temporary':True,'utc':now()})
                    temporary.unlink()
            except Exception as exc:
                append(B/'invalid_ledger.jsonl',{'utc':now(),'stage':self.stage,'job_id':job['job_id'],'directory':str(directory),
                    'returncode':code,'error':repr(exc),'exit_receipt_sha256':sha(directory/'exit_receipt.json'),'scientific_outcome_imputed':False})
                self.paused=f'invalid completion: {job["job_id"]}: {exc}'
            print(json.dumps({'job':job['job_id'],'returncode':code,'elapsed_s':elapsed,'pause':self.paused}),flush=True)

    def _execute(self,jobs,kind='rollout'):
        completed={r['job_id']:r for r in lines(B/'valid_ledger.jsonl')}
        pending=[j for j in jobs if j['job_id'] not in completed]
        while pending or self.active:
            self.poll();self.sample()
            estimate=(max(self.durations)*1.20 if self.durations else (1237*1.2 if kind=='rollout' else 900))
            if time.time()+estimate>self.safe_stop:self.paused='projected next completion crosses47h launch boundary'
            if self.paused:
                if not self.active:raise StopExecution(self.paused)
            else:
                while pending and len(self.active)<self.limit:
                    job=pending.pop(0)
                    command=worker_command(job) if kind=='rollout' else job['command']
                    self.launch(job,command,kind)
            if self.active:
                if time.time()>self.deadline:
                    raise StopExecution('hard48h deadline reached; drain escalates verified owned groups TERM then KILL')
                time.sleep(2)
        return jobs

    def execute(self,jobs,kind='rollout'):
        try:
            return self._execute(jobs,kind)
        except BaseException:
            self.paused='parent exception/interruption: drain all owned children before returning'
            # Healthy children finish without scientifically changing their rollout.
            # The hard deadline bounds the drain; training responds to SIGTERM at an epoch boundary.
            escalation=None
            while self.active:
                try:self.poll()
                except Exception as exc:
                    append(B/'errors/drain.jsonl',{'utc':now(),'error':repr(exc)})
                if time.time()>=self.deadline:
                    for item in self.active.values():
                        if process_start(item['process'].pid)==item['launch']['process_start_ticks']:
                            os.killpg(item['process'].pid,signal.SIGKILL if escalation and time.monotonic()-escalation>30 else signal.SIGTERM)
                    if escalation is None:escalation=time.monotonic()
                if self.active:time.sleep(2)
            raise



def validate_ledger():
    records=lines(B/'valid_ledger.jsonl');assert len({r['job_id'] for r in records})==len(records)
    for row in records:
        directory=Path(row['directory']);receipt=read(directory/'exit_receipt.json');job=read(directory/'job.json')
        assert sha(directory/'exit_receipt.json')==row['exit_receipt_sha256'] and receipt['observed_by_parent'] and receipt['returncode']==0
        assert receipt['job_sha256']==row['job_sha256']==job['job_sha256']
        result=directory/('result.json' if row['kind']=='rollout' else 'completed.json')
        assert sha(result)==row['result_sha256']
    return records


def historical_metric(job):
    from pact_v1010b_metrics import recompute
    directory=ROOT/job['historical_directory'];receipt=read(directory/'exit_receipt.json');record=read(directory/'result.json')
    assert receipt['returncode']==0 and record['checkpoint_sha256']==job['checkpoint_sha256']
    assert record['seed']==job['selected_seed'] and record['sampling_retry_index']==job['selected_retry']
    assert record['stats_sha256']==sha(W/'dataset_stats.pkl')
    row=recompute(directory);row.update(physical_row_digest=job['physical_row_digest'],scene_block=job['scene_block'])
    row['measurement_source']='retained_original';return row


def stage_rows(stage):
    from pact_v1010b_metrics import recompute
    jobs=read(B/f'schedules/{stage}.json');records={r['job_id']:r for r in validate_ledger()};rows=[]
    previous={}
    if stage=='B':previous={(r['scene_id'],r['training_seed']):r for r in read(B/'metrics/A2.json')['rows']}
    for job in jobs:
        if job['reuse']:
            row=historical_metric(job) if stage=='A2' else dict(previous[(job['scene_id'],job['training_seed'])])
        else:
            assert job['job_id'] in records,f'missing required completion {job["job_id"]}'
            row=recompute(records[job['job_id']]['directory'])
        rows.append(row)
    assert len(rows)==len(jobs)
    freeze(B/f'metrics/{stage}.json',{'schema':SCHEMA,'stage':stage,'expected':len(jobs),'rows':rows})
    return rows


def pair_stage(stage,rows):
    from pact_v1010b_pairing import audit_initial_group
    groups=collections.defaultdict(list)
    for row in rows:groups[row['scene_id']].append(row)
    for scene,group in groups.items():audit_initial_group(group,B/f'pairings/{"A" if stage.startswith("A") else stage}/{stage}_{scene}.json')


def replay_compare(rows,complete=True):
    import numpy as np
    evidence=read(A/'evidence.json');original={(r['episode_id'],r['arm']):r for r in evidence['rollouts']}
    comparisons=[]
    from pact_v1010b_pairing import audit_initial_group
    for row in rows:
        old=original[row['scene_id'],row['arm']];directory=ROOT/old['directory']
        with np.load(directory/'actions.npz') as a,np.load(Path(row['directory'])/'actions.npz') as b:
            differences={k:{'exact':bool(np.array_equal(a[k],b[k])), 'max_abs':float(np.max(np.abs(a[k]-b[k]))),
                'first_different_step':int(np.flatnonzero(np.any(a[k]!=b[k],axis=1))[0]) if np.any(a[k]!=b[k]) else None} for k in a.files}
        historical={'scene_id':row['scene_id'],'physical_row_digest':row['physical_row_digest'],'task_seed':old['task_seed'],
            'directory':str(directory),'provenance':{'arm':row['arm'],'training_seed':old['seed'],'source':'original'}}
        pairing=audit_initial_group([historical,row],B/f'pairings/A/replay_{row["arm"]}_{row["scene_id"]}.json')
        comparisons.append({'scene_id':row['scene_id'],'arm':row['arm'],'actions':differences,
            'outcome_exact':row['task_success']==old['success'],'failure_stage_exact':row['failure_stage']==old['failure_stage'],
            'initial_pairing_passed':pairing['passed'],'original_directory':str(directory),'replay_directory':row['directory']})
    eligible=all(r['outcome_exact'] and r['failure_stage_exact'] and all(x['exact'] for x in r['actions'].values()) for r in comparisons)
    doc={'schema':SCHEMA,'count':len(rows),'comparisons':comparisons,'comparison_eligible':eligible,
        'no_demonstrated_defect':eligible,'resolution':'Exact replay actions and outcomes.' if eligible else 'Action/outcome mismatch requires explicit diagnosis; no causal eligibility inferred.'}
    freeze(B/('replay_comparison.json' if complete else 'smoke_replay_comparison.json'),doc)
    if not eligible:raise StopExecution('frozen replay action/outcome difference remains unexplained; intervention blocked','STOPPED_AT_GATE_A')
    return doc


def diagnostic():
    verify_contract();validate_ledger();storage_gate('A')
    jobs=[resolve_job(j) for j in read(B/'schedules/A1.json')]
    pool=Pool('A1_smoke',1)
    for job in jobs[:2]:pool.execute([job])
    from pact_v1010b_metrics import recompute,evaluate_gate
    smoke=[recompute(j['output_dir']) for j in jobs[:2]];replay_compare(smoke,False)
    Pool('A1_remaining',12).execute(jobs[2:])
    rows=stage_rows('A1');pair_stage('A1',rows);replay_compare(rows)
    jobs=read(B/'schedules/A2.json');new=[resolve_job(j) for j in jobs if not j['reuse']]
    # Six crossed-checkpoint starts are fully observed and audited before filling the matrix.
    Pool('A2_first6',6).execute(new[:6])
    partial=[recompute(j['output_dir']) for j in new[:6]]
    scenes={r['scene_id'] for r in partial}
    partial += [historical_metric(j) for j in jobs if j['reuse'] and j['scene_id'] in scenes]
    pair_stage('A2_first6',partial)
    Pool('A2_remaining',12).execute(new[6:])
    rows=stage_rows('A2');pair_stage('A2',rows)
    gate=evaluate_gate('A');print(json.dumps({'stage_gate':'A','passed':gate['passed'],'failed_checks':gate['failed_checks']}),flush=True)
    if not gate['passed']:raise StopExecution('Stage A conditions failed: '+', '.join(gate['failed_checks']),'STOPPED_AT_GATE_A')
    return gate


def training():
    verify_contract();require_stage('training');storage_gate('training');jobs=[]
    for variant in VARIANTS[1:]:
        for seed in SEEDS:
            for arm in ('ACT','PACT'):
                directory=B/f'checkpoints/{arm.lower()}_seed{seed}_{variant}'
                command=[PYTHON,str(CODE/'pact_v1010b_train.py'),'--arm',arm,'--training-seed',str(seed),'--variant',variant,
                    '--max-updates','63000','--output-dir',str(directory)]
                job={'job_id':f'train_{arm}_{seed}_{variant}','output_dir':str(directory),'command':command,
                    'arm':arm,'training_seed':seed,'variant':variant,'contract_sha256':read(B/'contract.json')['contract_sha256']}
                job['job_sha256']=digest(job);jobs.append(job)
    freeze(B/'schedules/training.json',jobs);Pool('training',1).execute(jobs,'training')
    checks={}
    for variant in VARIANTS[1:]:
        for seed in SEEDS:
            a=read(B/f'checkpoints/act_seed{seed}_{variant}/sample_stream_hash.json')
            p=read(B/f'checkpoints/pact_seed{seed}_{variant}/sample_stream_hash.json')
            checks[f'paired_starts_{seed}_{variant}']=a==p
    assert all(checks.values())
    freeze(B/'gates/training.json',{'schema':SCHEMA,'stage':'training','passed':True,'branches':12,'new_updates':36000,'checks':checks})


def validation_diagnostic():
    # Prior-inference closure error on the fixed40 validation demos; fixed final models only.
    import numpy as np
    import h5py
    import pickle
    import torch
    from policy import ACTPolicy
    from pact_v1010b_metrics import fk_model
    pin_torch();fk,_=fk_model();stats=pickle.loads((W/'dataset_stats.pkl').read_bytes())
    demos=[r for r in read(A/'evidence.json')['demos'] if r['split']=='validation'];assert len(demos)==40
    samples=[]
    for d in demos:
        with h5py.File(d['converted_path']) as h:
            t=d['first_close_command_step'];samples.append({'id':d['episode_id'],'q':h['observations/qpos'][t],
                'image':h['observations/images/wrist_camera'][t],'prox':h['observations/proximity_embeddings'][t],'target':h['action'][t]})
    output={}
    for variant in VARIANTS[1:]:
        for seed in SEEDS:
            directory=B/f'checkpoints/pact_seed{seed}_{variant}';manifest=read(directory/'run_manifest.json')
            sys.argv=[sys.argv[0],'--ckpt_dir',str(directory),'--policy_class','ACT','--task_name','obstacle_baseline','--seed',str(seed),'--num_epochs','1']
            policy=ACTPolicy(manifest['policy_config']);policy.load_state_dict(torch.load(directory/'policy_update_63000.ckpt',map_location='cpu',weights_only=True),strict=True);policy.cuda().eval()
            errors=[]
            for start in range(0,40,8):
                batch=samples[start:start+8]
                q=torch.tensor((np.stack([s['q'] for s in batch])-stats['qpos_mean'])/stats['qpos_std'],dtype=torch.float32,device='cuda')
                image=torch.tensor(np.stack([s['image'] for s in batch]).transpose(0,3,1,2)[:,None],dtype=torch.float32,device='cuda')/255
                prox=torch.tensor(np.stack([s['prox'] for s in batch]),dtype=torch.float32,device='cuda')
                with torch.inference_mode():prediction=policy(q,image,proximity_positions=prox).cpu().numpy()[:,0]
                prediction=prediction*stats['action_std']+stats['action_mean']
                errors.extend({'episode_id':s['id'],'error_m':float(np.linalg.norm(fk(p[:7])-fk(s['target'][:7])))} for s,p in zip(batch,prediction))
            output[f'PACT_{seed}_{variant}']={'n':40,'median_m':float(np.median([r['error_m'] for r in errors])),'rows':errors,'checkpoint_sha256':sha(directory/'policy_update_63000.ckpt')}
            del policy;torch.cuda.empty_cache()
    freeze(B/'validation_close_errors.json',output)


def evaluation(stage):
    verify_contract();require_stage(stage);storage_gate(stage)
    jobs=[resolve_job(j) for j in read(B/f'schedules/{stage}.json') if not j['reuse']]
    if stage=='B':
        first_scene=read(B/'scene_manifests/B.json')['scenes'][0]['scene_id']
        smoke=[j for j in jobs if j['scene_id']==first_scene and j['variant']!='frozen60000'];assert len(smoke)==12
        Pool('B_smoke',12).execute(smoke)
        from pact_v1010b_metrics import recompute
        pair_stage('B_smoke',[recompute(j['output_dir']) for j in smoke])
        jobs=[j for j in jobs if j not in smoke]
    Pool(stage,12).execute(jobs)
    rows=stage_rows(stage);pair_stage(stage,rows)
    if stage=='C':validation_diagnostic()
    from pact_v1010b_metrics import evaluate_gate
    gate=evaluate_gate(stage);print(json.dumps({'stage_gate':stage,'passed':gate['passed'],'failed_checks':gate['failed_checks']}),flush=True)
    if not gate['passed']:raise StopExecution(f'Stage {stage} failed: '+', '.join(gate['failed_checks']),
        'FINAL_TARGET_MISSED' if stage=='D' else 'STOPPED_AT_GATE_'+stage)
    return gate


def verify():
    contract=verify_contract(full=True);records=validate_ledger()
    from pact_v1010b_metrics import recompute
    for r in records:
        if r['kind']=='rollout':recompute(r['directory'])
    from pact_v1010b_pairing import audit_initial_group
    from pact_v1010b_metrics import recompute
    for stage in ('A1','A2','B','C','D'):
        path=B/f'metrics/{stage}.json'
        if not path.exists():continue
        rows=read(path)['rows'];groups=collections.defaultdict(list)
        for row in rows:
            raw=recompute(row['directory']);raw['physical_row_digest']=row['physical_row_digest'];groups[row['scene_id']].append(raw)
        for scene,group in groups.items():audit_initial_group(group,B/f'final_pairings/{stage}_{scene}.json')
    doc={'schema':SCHEMA,'utc':now(),'valid_rollouts':sum(r['kind']=='rollout' for r in records),
        'valid_training_branches':sum(r['kind']=='training' for r in records),'raw_recomputation_passed':True,
        'historical_protected_hashes_unchanged':True,'user_eval_preserved':True,'records_with_observed_zero_exit':len(records)}
    atomic(B/'final_verification.json',doc);return doc


def close(status,reason):
    records=validate_ledger();counts={s:sum(r['kind']=='rollout' and read(Path(r['directory'])/'job.json')['stage']==s for r in records) for s in ('A1','A2','B','C','D')}
    launches=list(B.rglob('launch.json'))
    observed=all((p.parent/'exit_receipt.json').exists() and read(p.parent/'exit_receipt.json').get('observed_by_parent',False) for p in launches)
    models=[read(p) for p in (B/'checkpoints').glob('*/completed.json')]
    doc={'schema':SCHEMA,'status':status,'reason':reason,'utc':now(),'parent_pid':os.getpid(),
        'completed_new_rollouts':sum(counts.values()),'new_rollouts_by_stage':counts,
        'expected_main_path_new_rollouts':1056,'expected_new_counts':{'A1':12,'A2':48,'B':180,'C':216,'D':600},
        'completed_training_branches':len(models),'new_training_updates':sum(m['new_updates'] for m in models),
        'final_completed':counts['D']==600,'baseline_retained':status!='FINAL_ACCEPTED',
        'rollback_models':read(B/'contract.json')['models'],'all_recorded_exits_observed':observed,'launches':len(launches),
        'separate_monitoring_enabled':False,'historical_authorizations_unchanged':True}
    text=['# V10.10b bounded grasp-acquisition experiment','',f'Status: **{status}**.','',reason,'',
        f'Completed new rollouts: {sum(counts.values())}/1056 maximum main path. Final comparison: {counts["D"]}/600.',
        f'Completed continuation branches: {len(models)}/12; new optimizer updates: {doc["new_training_updates"]}/36000.','',
        '| Stage | New rollouts completed | Main-path expected |','|---|---:|---:|']
    text += [f'| {s} | {counts[s]} | {doc["expected_new_counts"][s]} |' for s in counts]
    text += ['', 'The unchanged update60000 ACT/PACT baselines remain runnable. This execution does not alter historical qualification or authorization flags.',
        'Scene blocks and training seeds are separate identities. Curated replays are selected for mechanism inspection; crossed historical scenes are exposed diagnostics.',
        'A prerequisite stop completes the bounded experiment without demonstrating a successful fix. Missing/infrastructure endpoints are not imputed or removed from scientific denominators.',
        'All writes are confined to the new scripts/tests and this experiment directory. The pre-existing root EVAL.md is preserved.']
    for path in sorted((B/'gates').glob('*.json')):
        gate=read(path);text += ['',f'Gate {gate["stage"]}: {"PASS" if gate["passed"] else "FAIL"}. Failed checks: '+', '.join(gate.get('failed_checks',[]))]
        if gate.get('details'):text += ['', '```json',json.dumps(gate['details'],indent=2),'```']
    for name in ('smoke_replay_comparison.json','replay_comparison.json'):
        if (B/name).exists():text += ['',name+':','```json',json.dumps(read(B/name),indent=2),'```']
    (B/'EVAL.md').write_text('\n'.join(text)+'\n');atomic(B/'parent_closure.json',doc)
    return doc


def prepare(args):
    assert Path(args.output).resolve()==B and args.workers==12 and args.hours==48
    if (B/'contract.json').exists():return verify_contract()
    doc=build_contract(now());storage_gate('prepare')
    runtime={'python':sys.version,'platform':platform.platform(),'resources':resources()}
    for library in ('numpy','scipy','h5py','mujoco','torch','cv2','imageio'):
        module=__import__(library);runtime[library]=str(getattr(module,'__version__','unknown'))
    freeze(B/'runtime.json',runtime);print(json.dumps({'prepared':True,'contract_sha256':doc['contract_sha256']}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','diagnostic','train','mechanism','development','final','verify','supervise'))
    p.add_argument('--output',type=Path,default=B);p.add_argument('--workers',type=int,default=12);p.add_argument('--hours',type=int,default=48)
    args=p.parse_args();assert args.output.resolve()==B
    B.mkdir(parents=True,exist_ok=True)
    lock=(B/'parent.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.stage=='prepare':prepare(args);return
    verify_contract();validate_ledger()
    if args.stage=='verify':print(json.dumps(verify()));return
    try:
        if args.stage=='supervise':
            diagnostic();training();evaluation('B');evaluation('C');evaluation('D');verify();close('FINAL_ACCEPTED','All frozen gates passed.')
        else:{'diagnostic':diagnostic,'train':training,'mechanism':lambda:evaluation('B'),'development':lambda:evaluation('C'),'final':lambda:evaluation('D')}[args.stage]()
    except StopExecution as exc:
        verify();print(json.dumps(close(exc.status,str(exc))),flush=True)
    except BaseException as exc:
        atomic(B/f'errors/parent_{time.time_ns()}.json',{'utc':now(),'error':repr(exc),'traceback':traceback.format_exc()})
        close('INCOMPLETE_INFRASTRUCTURE_OR_BUDGET',repr(exc));raise

if __name__=='__main__':main()
