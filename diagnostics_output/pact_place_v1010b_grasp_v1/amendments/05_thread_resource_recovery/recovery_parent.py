"""One diagnosed resume and conservative capacity reduction, with unchanged frozen harness."""
import sys,fcntl,traceback,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import pact_v1010b_run as run
from pact_v1010b_contract import *
am=B/'amendments/05_thread_resource_recovery'
lock=(B/'parent.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
manifest=read(am/'recovery_manifest.json');amend=read(am/'amendment.json')
assert sha(__file__)==amend['launcher_sha256'] and sha(am/'recovery_manifest.json')==amend['recovery_manifest_sha256']
assert sha(am/'preflight.json')==amend['preflight_sha256']
assert read(B/'contract.json')['contract_sha256']==manifest['contract_sha256']==amend['contract_sha256']
run.verify_contract();run.validate_ledger();run.require_stage('training')
assert sha(manifest['resume_path'])==manifest['resume_sha256']
assert sum(r['kind']=='training' for r in lines(B/'valid_ledger.jsonl'))==11
assert manifest['per_job_retry_ordinal']==1 and manifest['retry_ordinal_global']==2
original_pool=run.Pool
# Operational backoff only: retain the same12 smoke jobs and every frozen matrix row.
def capped_pool(stage,workers=12):return original_pool(stage,min(workers,10))
run.Pool=capped_pool
append(B/'resources/capacity_changes.jsonl',{'utc':now(),'stage':'recovery02_onward','limit':10,
 'reason':'Diagnosed DataLoader thread-creation failure; serial training remains1 and unchanged4 loader workers; subsequent evaluation pools capped10.',
 'recovery_manifest_sha256':sha(am/'recovery_manifest.json')})
job=dict(manifest['prior_job']);job.pop('job_sha256');job.update(attempt=1,
 recovery_manifest_sha256=sha(am/'recovery_manifest.json'),prior_exit_receipt_sha256=manifest['prior_exit_receipt_sha256'])
job['job_sha256']=digest(job);freeze(am/'retry_job.json',job)
atomic(B/'run_status.json',{'utc':now(),'status':'RUNNING_RECOVERY02','parent_pid':os.getpid(),
 'contract_sha256':manifest['contract_sha256'],'progress_path':str(B/'progress.json'),
 'superseded_closure_path':str(am/'prior/parent_closure.json'),'recovery_manifest':str(am/'recovery_manifest.json'),
 'eval_worker_cap':10,'completed_training_branches_at_start':11,'completed_new_rollouts_at_start':60})
old=read(B/'parent_closure.json');old.update(status='SUPERSEDED_BY_ACTIVE_RECOVERY02',superseded_status=old['status'],
 superseded_closure_path=str(am/'prior/parent_closure.json'),active_run_status_path=str(B/'run_status.json'))
atomic(B/'parent_closure.json',old)
report=B/'EVAL.md';report.write_text('> This earlier closure is superseded by the active, bounded infrastructure retry. See [run_status.json](run_status.json).\n\n'+report.read_text())
try:
 run.Pool('training_retry02',1).execute([job],'training')
 run.training()
 for stage in ('B','C','D'):run.evaluation(stage)
 run.verify();run.close('FINAL_ACCEPTED','All frozen gates passed.')
except run.StopExecution as exc:
 run.verify();run.close(exc.status,str(exc))
except BaseException as exc:
 atomic(B/f'errors/recovery_parent_{time.time_ns()}.json',{'utc':now(),'error':repr(exc),'traceback':traceback.format_exc()})
 run.close('INCOMPLETE_INFRASTRUCTURE_OR_BUDGET',repr(exc));raise
finally:
 closure=read(B/'parent_closure.json')
 closure.update(operational_recovery_manifest_sha256=sha(am/'recovery_manifest.json'),
  final_eval_worker_cap=10,infrastructure_retries_used=2,uncommitted_optimizer_updates_replayed=240,
  note_update_count='new_training_updates counts committed60001..63000 transitions;240 additional uncommitted transitions were replayed after the one thread-failure retry.')
 atomic(B/'parent_closure.json',closure)
 atomic(B/'run_status.json',{'utc':now(),'status':closure['status'],'active_parent':False,
  'parent_pid':os.getpid(),'closure_path':str(B/'parent_closure.json'),
  'completed_new_rollouts':closure['completed_new_rollouts'],'completed_training_branches':closure['completed_training_branches'],
  'new_training_updates':closure['new_training_updates'],'uncommitted_optimizer_updates_replayed':240})
 print(json.dumps(closure),flush=True)
