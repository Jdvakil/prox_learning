"""Raw paired validation, prespecified development gate, and experiment report."""
from __future__ import annotations
import collections
from pact_wrist288_common import *

def rgb_difference(a,b):
    import numpy as np
    assert a.dtype==b.dtype==np.uint8 and a.shape==b.shape and a.ndim==3 and a.shape[-1]==3
    delta=np.abs(a.astype(np.int16)-b.astype(np.int16))
    changed=int(np.count_nonzero(delta)); maximum=int(delta.max())
    return {'max_abs_delta':maximum,'changed_channel_values':changed,'channel_values':int(a.size),
        'changed_fraction':changed/a.size,'passed':maximum<=2 and changed/a.size<=0.001},delta

def check_pair(a,b,destination):
    import h5py
    import numpy as np
    for key in ('episode_id','task_seed','row_sha256','seed','averaging_history'):
        assert a[key]==b[key],f'pair identity differs: {key}'
    audit={'episode_id':a['episode_id'],'exact_non_rgb':[],'rgb':{},'violations':[]}
    with h5py.File(ROOT/a['directory']/'initial_observation.h5') as ha,h5py.File(ROOT/b['directory']/'initial_observation.h5') as hb:
        names=[[],[]]
        for i,h in enumerate((ha,hb)):
            h.visititems(lambda n,o:names[i].append(n) if isinstance(o,h5py.Dataset) else None)
        assert names[0]==names[1] and names[0]
        for name in names[0]:
            av,bv=ha[name][()],hb[name][()]
            assert ha[name].shape==hb[name].shape and ha[name].dtype==hb[name].dtype,name
            if name=='observation/wrist_camera':
                rgb,delta=rgb_difference(av,bv);audit['rgb']=rgb
                destination.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(destination.with_suffix('.npz'),absolute_difference=delta)
                if not rgb['passed']: audit['violations'].append('RGB tolerance exceeded')
            else:
                if np.asarray(av).tobytes()!=np.asarray(bv).tobytes(): audit['violations'].append(name)
                else: audit['exact_non_rgb'].append(name)
    assert audit['rgb'] and any(n.startswith('physics/') for n in audit['exact_non_rgb'])
    audit['passed']=not audit['violations'];freeze(destination,audit)
    assert audit['passed'],audit
    return audit

def select_history(rows):
    candidates={}
    for history in (100,10):
        selected=[r for r in rows if r['averaging_history']==history]
        assert len(selected)==48 and collections.Counter(r['arm'] for r in selected)=={'ACT':24,'PACT':24}
        candidates[history]=(sum(r['task_success'] for r in selected),sum(r['collision_free_task_success'] for r in selected),
            -sum(r['hazard_frames'] for r in selected),int(history==100))
    chosen=max(candidates,key=candidates.get)
    successes={arm:sum(r['task_success'] for r in rows if r['averaging_history']==chosen and r['arm']==arm) for arm in ('ACT','PACT')}
    return {'history':chosen,'selection_scores':{str(k):list(v) for k,v in candidates.items()},'successes':successes,
        'passed':successes['PACT']>=13 and successes['PACT']-successes['ACT']>=3}

def validate_completions(records,jobs):
    from pact_wrist288_metrics import metrics
    assert len(records)==len(jobs)
    lookup={j['schedule']['rollout_id']:j for j in jobs}
    assert len(lookup)==len(jobs)==len({r['rollout_id'] for r in records})
    outcomes=[]
    for record in records:
        job=lookup[record['rollout_id']]
        assert record['returncode']==0 and record['valid_completion']
        directory=Path(job['directory'])
        receipt=read(directory/'exit_receipt.json')
        assert receipt['returncode']==0 and receipt['job_sha256']==digest(job)
        assert sha(directory/'exit_receipt.json')==record['exit_receipt_sha256']
        assert sha(directory/'result.json')==record['result_sha256']
        outcomes.append(metrics(directory,job['schedule']))
    return outcomes

def aggregate(rows):
    n=len(rows)
    keys=('task_success','collision_free_task_success','collision_free','hazard_contact_episode','hazard_frames',
        'target_touch','target_held','target_lifted_1cm','placement_support')
    result={'n':n,**{k:sum(r[k] for r in rows) for k in keys}}
    result['mean_hazard_frames']=result['hazard_frames']/n if n else None
    result['failure_stages']=dict(collections.Counter(r['failure_stage'] for r in rows))
    result['per_object']={slot:{key:sum(r['per_object'][slot][key] for r in rows) for key in
        ('contact_entries','contact_frames','contact_episode','stability_event','stable')} for slot in ACTIVE_CLUTTER_SLOTS}
    result['absent_slots']=['08','09']
    return result

def paired_counts(rows):
    pairs=collections.defaultdict(dict)
    for r in rows:
        key=(r['seed'],r['episode_id'],r['averaging_history'])
        assert r['arm'] not in pairs[key]
        pairs[key][r['arm']]=r
    assert all(set(pair)=={'ACT','PACT'} for pair in pairs.values())
    return {'ACT_only':sum(p['ACT']['task_success'] and not p['PACT']['task_success'] for p in pairs.values()),
        'PACT_only':sum(p['PACT']['task_success'] and not p['ACT']['task_success'] for p in pairs.values())}

def report(verdict,reason,rows=None):
    rows=rows or []
    state=read(WORK/'state.json') if (WORK/'state.json').exists() else {}
    lines_out=[f'# V10.10 wrist288 three-seed experiment', '',f'Verdict: **{verdict}**.', '',reason,'',
        f'Updated: {now()}. All `authorizes_*` flags remain false; historical qualification outcomes are unchanged.', '',
        f'Completed collection: {sum(r.get("accepted",False) for r in lines(WORK/"collection/ledger.jsonl"))}/288.',
        f'Converted episodes: {len(lines(WORK/"conversion_ledger.jsonl"))}/288.', '']
    for path in sorted((WORK/'checkpoints').glob('*/epoch_log.jsonl')):
        logs=lines(path); lines_out.append(f'- {path.parent.name}: {logs[-1]["global_step"] if logs else 0}/60,000 logged optimizer updates.')
    if rows:
        summaries={str(s):{arm:aggregate([r for r in rows if r['seed']==s and r['arm']==arm]) for arm in ('ACT','PACT')}
            for s in sorted({r['seed'] for r in rows})}
        pooled={arm:aggregate([r for r in rows if r['arm']==arm]) for arm in ('ACT','PACT')}
        lines_out += ['', '| Seed | ACT task success | PACT task success | PACT−ACT | ACT collision-free success | PACT collision-free success |',
            '|---|---:|---:|---:|---:|---:|']
        for seed,arms in summaries.items():
            a,p=arms['ACT'],arms['PACT'];n=a['n'];assert n==p['n']
            lines_out.append(f'| {seed} | {a["task_success"]}/{n} | {p["task_success"]}/{n} | {100*(p["task_success"]-a["task_success"])/n:+.1f} pp | {a["collision_free_task_success"]}/{n} | {p["collision_free_task_success"]}/{n} |')
        lines_out+=['','| Arm | Pooled success | Collision-free rollouts | Hazard-contact rollouts | Hazard frames | Mean frames/rollout |',
            '|---|---:|---:|---:|---:|---:|']
        for arm,a in pooled.items():
            lines_out.append(f'| {arm} | {a["task_success"]}/{a["n"]} | {a["collision_free"]}/{a["n"]} | {a["hazard_contact_episode"]}/{a["n"]} | {a["hazard_frames"]} | {a["mean_hazard_frames"]:.2f} |')
        paired=paired_counts(rows);lines_out+=['',f'Paired successes: ACT-only {paired["ACT_only"]}; PACT-only {paired["PACT_only"]}.','',
            '| Arm | Target touch | Held | Lift ≥1 cm | Receptacle support |','|---|---:|---:|---:|---:|']
        for arm,a in pooled.items(): lines_out.append(f'| {arm} | {a["target_touch"]} | {a["target_held"]} | {a["target_lifted_1cm"]} | {a["placement_support"]} |')
        lines_out+=['','| Arm | Slot | Contact rollouts | Contact frames | Stability-event rollouts | Stable rollouts |','|---|---|---:|---:|---:|---:|']
        for arm,a in pooled.items():
            for slot,stats in a['per_object'].items():
                lines_out.append(f'| {arm} | {slot} | {stats["contact_episode"]} | {stats["contact_frames"]} | {stats["stability_event"]} | {stats["stable"]} |')
        lines_out+=['','Slots 08/09 are absent; they are not zero-contact observations.','']
        if len(summaries)==3:
            means={arm:sum(s[arm]['task_success']/s[arm]['n'] for s in summaries.values())/3 for arm in ('ACT','PACT')}
            lines_out.append(f'Three-seed mean task success: ACT {100*means["ACT"]:.2f}%; PACT {100*means["PACT"]:.2f}%; difference {100*(means["PACT"]-means["ACT"]):+.2f} pp.')
        atomic(WORK/'analysis.json',{**empty_authorization(),'verdict':verdict,'per_seed':summaries,'pooled':pooled,'paired':paired,'rows':rows})
    lines_out += ['', 'Clutter was effectively invisible to the proximity skin in the prior resolvability audit: inbound vessel `max_w_perp_m = 0.000` in 7 of 8 variants. The 40 sensors cover link1–link6 only. A PACT advantage is not evidence that it senses the clutter.', '',
        'Action alignment follows the [MolmoSpaces data format](https://allenai.github.io/molmospaces/data_format/) and the [original ACT recorder](https://github.com/tonyzhaozh/act/blob/main/record_sim_episodes.py). Its effect is measured by this experiment.', '',
        'Artifacts: `config.json`, `manifests/`, `collection/ledger.jsonl`, `raw/`, `converted/`, `checkpoints/`, `evaluation/`, and `monitoring/hourly_run_check.jsonl`.']
    (WORK/'EVAL.md').write_text('\n'.join(lines_out)+'\n')
