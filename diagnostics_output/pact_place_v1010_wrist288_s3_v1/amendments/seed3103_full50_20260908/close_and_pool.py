"""Verify completed extensions and write a separate, full 150-pair summary."""
import sys
from pathlib import Path
ROOT=Path('/root/prox_learning_pact_remediation');sys.path.insert(0,str(ROOT/'scripts'))
import pact_wrist280_seed3103_full50 as extension
from pact_wrist288_common import WORK,read,lines,sha,digest,now,empty_authorization,atomic
from pact_wrist288_analysis import aggregate,paired_counts

EXT=extension.EXT
contract=extension.verify_extension()
allrows=[];sources={};pairaudits=[]
for seed,folder in ((3103,'seed3103_full50_20260908'),(3104,'seed3104_full50_20260907'),(3105,'seed3105_full50_20260907')):
    ext=WORK/'amendments'/folder
    ledger=WORK/f'evaluation/final_{seed}_full50_h100_ledger.jsonl'
    schedule=read(WORK/f'evaluation/final_{seed}_full50_h100_schedule.json')
    records=lines(ledger);complete=read(WORK/f'evaluation/final_{seed}_full50_h100_complete.json')
    result=read(ext/'analysis.json');manifest=read(WORK/f'manifests/final_{seed}.json')
    assert len(records)==len(schedule)==complete['count']==100
    assert len({r['rollout_id']for r in records})==100
    assert {r['episode_id']for r in records}=={r['episode_id']for r in manifest['rows']}
    assert len({r['episode_id']for r in records})==50
    assert complete['ledger_sha256']==sha(ledger)
    assert result['rows']==complete['rows']
    lookup={j['schedule']['rollout_id']:j for j in schedule}
    for r in records:
        assert r['valid_completion'] and r['returncode']==0 and r['validation_error'] is None
        assert r['checkpoint_seed']==seed and r['averaging_history']==100
        job=lookup[r['rollout_id']];d=Path(r['directory']);receipt=read(d/'exit_receipt.json')
        assert read(d/'job.json')==job
        assert receipt['job_sha256']==digest(job) and receipt['returncode']==0
        assert receipt['exit_evidence']=='subprocess.Popen.poll / waitpid'
        assert sha(d/'exit_receipt.json')==r['exit_receipt_sha256']
        assert sha(d/'result.json')==r['result_sha256']
    audits=sorted((WORK/'evaluation/pair_audits').glob(f'final_{seed}_full50_h100_*.json'))
    assert len(audits)==50
    seen=set()
    for path in audits:
        audit=read(path);assert audit['passed'] and not audit['violations']
        assert audit['rgb']['passed'] and audit['rgb']['max_abs_delta']<=2 and audit['rgb']['changed_fraction']<=.001
        assert any(x.startswith('physics/')for x in audit['exact_non_rgb'])
        assert path.with_suffix('.npz').is_file()
        seen.add(audit['episode_id']);pairaudits.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
    assert seen=={r['episode_id']for r in records}
    assert all(not v for k,v in result.items()if k.startswith('authorizes_'))
    allrows.extend(result['rows'])
    sources[seed]={'report':str((ext/'EVAL.md').relative_to(ROOT)),'report_sha256':sha(ext/'EVAL.md'),
                  'analysis_sha256':sha(ext/'analysis.json'),'ledger_sha256':sha(ledger),
                  'contract_sha256':read(ext/'contract.json')['sha256'],'passed_pair_audits':50}
assert len(allrows)==len({(r['seed'],r['episode_id'],r['arm'])for r in allrows})==300
assert len({r['episode_id']for r in allrows})==150
monitor=lines(WORK/'monitoring/hourly_run_check.jsonl')[-1]
assert monitor['final'] and not monitor['verification_issues'] and not monitor['new_worker_errors']
assert not monitor['owned_orphans'] and not monitor['expected_running_workers']
assert not (WORK/'PAUSE.json').exists()
assert not Path('/proc/1915411').exists() and not Path('/proc/1915412').exists()
shutdown=read(WORK/'SAFE_SHUTDOWN.json');assert shutdown['verdict']=='SEED 3103 FULL 50-PAIR EVALUATION COMPLETE'
assert all(not v for k,v in shutdown.items()if k.startswith('authorizes_'))
per_seed={str(s):{arm:aggregate([r for r in allrows if r['seed']==s and r['arm']==arm])for arm in ('ACT','PACT')}for s in (3103,3104,3105)}
pooled={arm:aggregate([r for r in allrows if r['arm']==arm])for arm in ('ACT','PACT')}
for s in per_seed:
    assert per_seed[s]==read(WORK/sources[int(s)]['report'].split('pact_place_v1010_wrist288_s3_v1/',1)[1].replace('EVAL.md','analysis.json'))['groups']['full_50']
bench={'PACT_at_least_76_of_150':pooled['PACT']['task_success']>=76,
       'PACT_at_least_15_more_successes':pooled['PACT']['task_success']-pooled['ACT']['task_success']>=15}
pair=paired_counts(allrows)
analysis={**empty_authorization(),'utc':now(),'verdict':'THREE_SEED_FULL150_VERIFIED_COMPLETE',
          'per_seed':per_seed,'pooled':pooled,'paired':pair,'benchmark':bench,'sources':sources,
          'passed_pair_audits':150,'pair_audits':pairaudits,'rows':allrows}
atomic(EXT/'three_seed_full150.json',analysis)
text=['# Full three-seed evaluation: 50 pairs per seed','',
      f'Completed and verified at {now()}: 150 physical instances, 300 ACT/PACT rollouts, all 150 initial-state pairing audits passed. All models used 60,000-update checkpoints and history 100.','',
      '| Seed | ACT task success | PACT task success | PACT−ACT | ACT collision-free success | PACT collision-free success |',
      '|---|---:|---:|---:|---:|---:|']
for s,arms in [*per_seed.items(),('Pooled',pooled)]:
    a,p=arms['ACT'],arms['PACT'];n=a['n']
    text.append(f'| {s} | {a["task_success"]}/{n} ({100*a["task_success"]/n:.1f}%) | {p["task_success"]}/{n} ({100*p["task_success"]/n:.1f}%) | {100*(p["task_success"]-a["task_success"])/n:+.1f} pp | {a["collision_free_task_success"]}/{n} ({100*a["collision_free_task_success"]/n:.1f}%) | {p["collision_free_task_success"]}/{n} ({100*p["collision_free_task_success"]/n:.1f}%) |')
text+=['','The equal-size three-seed mean equals the pooled rate. The original target was not met: PACT needed at least 76/150 successes and at least 15 more successes than ACT. The observed counts are 74/150 and 10 more, respectively. These are descriptive outcomes; no confidence intervals or significance tests are included.','',
       '| Arm | Collision-free rollouts | Hazard-contact rollouts | Hazard-free rollouts | Hazard frames | Mean hazard frames/rollout |','|---|---:|---:|---:|---:|---:|']
for arm,a in pooled.items():
    text.append(f'| {arm} | {a["collision_free"]}/150 | {a["hazard_contact_episode"]}/150 | {150-a["hazard_contact_episode"]}/150 | {a["hazard_frames"]} | {a["mean_hazard_frames"]:.2f} |')
text+=['',f'Paired successes: ACT-only {pair["ACT_only"]}; PACT-only {pair["PACT_only"]}.','',
       '| Arm | Target touch | Contact-only hold | Lift ≥1 cm | Tray support |','|---|---:|---:|---:|---:|']
for arm,a in pooled.items():text.append(f'| {arm} | {a["target_touch"]}/150 | {a["target_held"]}/150 | {a["target_lifted_1cm"]}/150 | {a["placement_support"]}/150 |')
text+=['','| Arm | Slot | Contact rollouts | Contact frames | Stability-event rollouts | Stable rollouts |','|---|---|---:|---:|---:|---:|']
for arm,a in pooled.items():
    for slot,v in a['per_object'].items():text.append(f'| {arm} | {slot} | {v["contact_episode"]}/150 | {v["contact_frames"]} | {v["stability_event"]}/150 | {v["stable"]}/150 |')
text+=['','Slots 08/09 are absent. The 40 proximity sensors cover link1–link6; the historical sensor audit does not establish clutter visibility.','',
       'The original reduced n=50 report and all three extension reports remain unchanged. The development gate remains missed and all authorization/qualification flags remain false. Extensions reused the original frozen full manifests, preserving every valid failure.','',
       'Source hashes, all rows and pairing-audit bindings are in `three_seed_full150.json`. Seed 3103 completed its final rollout by 01:54:59 UTC and finalized reporting by 01:56:40 UTC; the final monitor audit is clean and both supervisor and monitor have exited.']
(EXT/'THREE_SEED_FULL150.md').write_text('\n'.join(text)+'\n')
closure={**empty_authorization(),'utc':now(),'status':'SEED_3103_FULL50_VERIFIED_COMPLETE','pairs':50,'valid_rollouts':100,
         'passed_pair_audits':50,'original_three_seed_result_preserved':True,'seed3104_full50_result_preserved':True,
         'seed3105_full50_result_preserved':True,'extension_contract_sha256':contract['sha256'],
         'analysis_sha256':sha(EXT/'analysis.json'),'report_sha256':sha(EXT/'EVAL.md'),
         'ledger_sha256':sources[3103]['ledger_sha256'],'final_monitor_audit_utc':monitor['utc'],
         'supervisor_and_monitor_exited':True,'expected_running_workers':0,
         'successes':{arm:per_seed['3103'][arm]['task_success']for arm in ('ACT','PACT')},
         'full150_summary_sha256':sha(EXT/'three_seed_full150.json')}
atomic(EXT/'parent_closure.json',closure)
print({'closure':closure,'pooled':pooled,'paired':pair})
