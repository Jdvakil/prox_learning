#!/usr/bin/env python3
"""Frozen seed-3103 idealized VL53L8UX L8UX-inspired 16x16 sensor-profile transfer study."""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from pact_vl53l8ux_16x16_sensor import PROFILE

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'scripts'
spec = importlib.util.spec_from_file_location('_vl53l8ux_16x16_audit_helpers', CODE/'pact_v1010c_permuted_128d.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
OUT = ROOT/'diagnostics_output/pact_place_v1010c_vl53l8ux_16x16_s3103_v1'
SCHEMA = 'pact_v1010c_vl53l8ux_16x16_128d_s3103_v1'
SOURCE, DATA, PYTHON = base.SOURCE, base.DATA, base.PYTHON
EXPECTED, FORBIDDEN = base.EXPECTED, base.FORBIDDEN
N, HORIZON, SENSORS, DIM = 50, 900, 40, 128
ARMS = ('VL53L8UX_INSPIRED_16X16_MIN_15HZ',)
ANALYSIS_SEED = 2026091801
MARGIN = .10
IMPLEMENTATION = ('scripts/pact_v1010c_vl53l8ux_16x16_128d.py',
                  'scripts/pact_v1010c_vl53l8ux_16x16_128d_eval.py',
                  'scripts/pact_vl53l8ux_16x16_sensor.py',
                  'scripts/pact_v1010c_permuted_128d.py',
                  'scripts/pact_v1010c_core.py', 'scripts/pact_v1010c_eval.py')
base.OUT, base.SCHEMA, base.IMPLEMENTATION, base.ANALYSIS_SEED = OUT, SCHEMA, IMPLEMENTATION, ANALYSIS_SEED
now, read, sha, digest, require = base.now, base.read, base.sha, base.digest, base.require
inside, atomic, freeze, append, lines = base.inside, base.atomic, base.freeze, base.append, base.lines
self_hashed, checked_json = base.self_hashed, base.checked_json
implementation_hashes, verify_models = base.implementation_hashes, base.verify_models
resources = base.resources

def capacity_ok(sample):
    # Native 16x16 records need more storage than the four-zone studies.
    # Preserve a 4GiB reserve; never delete earlier artifacts to make room.
    return (sample['ram_fraction'] < .80 and sample['pid_fraction'] < .75 and
            sample['vram_fraction'] < .85 and sample['disk_free_bytes'] > 4*2**30)

completed_jobs, audit_metrics, paired_statistics = base.completed_jobs, base.audit_metrics, base.paired_statistics


def environment():
    env = base.environment()
    env.pop('PACT_PERMUTED_128D_OWNER', None)
    env['PACT_VL53L8UX_OWNER'] = SCHEMA
    return env


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    source = base.prepare_reference()
    freeze(OUT/'sensor_profile.json', self_hashed(dict(PROFILE, schema=SCHEMA)))
    freeze(OUT/'schedule.json', self_hashed(dict(
        source_manifest_sha256=source['sha256'],
        rows=[dict(scene_index=s['index'], arm=ARMS[0]) for s in source['scenes']],
        allocation='all 50 source scenarios in original order; no outcome selection')))
    print(json.dumps(dict(stage='prepared', scenarios=N, new_scientific_rollouts=N)), flush=True)


def verify_inputs(full=True):
    verify_models()
    source = checked_json(OUT/'source_manifest.json')
    profile = checked_json(OUT/'sensor_profile.json')
    schedule = checked_json(OUT/'schedule.json')
    require(profile == self_hashed(dict(PROFILE, schema=SCHEMA)), 'sensor profile drift')
    require(schedule['source_manifest_sha256'] == source['sha256'], 'manifest mismatch')
    require(schedule['rows'] == [dict(scene_index=i, arm=ARMS[0]) for i in range(N)], 'scenario selection drift')
    require(sha(OUT/'live_metrics.json') == source['live_metrics_sha256'], 'reference metrics drift')
    if full:
        for scene in source['scenes']:
            for name, expected in scene['artifacts'].items():
                require(sha(Path(scene['live_directory'])/name) == expected, 'historical artifact drift')
    return source, profile, schedule


def make_job(scene, arm, attempt, protocol=None):
    require(arm in (*ARMS, 'LIVE_PREFLIGHT'), 'unknown experiment arm')
    source = checked_json(OUT/'source_manifest.json')
    profile = checked_json(OUT/'sensor_profile.json')
    job_id = f'{arm.lower()}_{scene["index"]:03d}_{scene["scene_id"][:12]}'
    directory = OUT/('preflight' if arm == 'LIVE_PREFLIGHT' else 'rollouts')/job_id/f'attempt_{attempt:02d}'
    return self_hashed(dict(schema=SCHEMA, id=job_id, directory=str(directory), attempt=attempt,
        experiment_arm=arm, variant='readout60000', scene_id=scene['scene_id'], scene_index=scene['index'],
        checkpoint_sha256=EXPECTED['policy_last.ckpt'], encoder_sha256=EXPECTED['prox_encoder.pt'],
        source_manifest_sha256=source['sha256'], sensor_profile_sha256=profile['sha256'],
        implementation_hashes=implementation_hashes(), protocol_sha256=protocol))


def run_jobs(requests, workers, protocol=None):
    """Durable attempt receipts and bounded technical retry; mixed arm schedule."""
    import fcntl
    require(1 <= workers <= 6, 'workers must be between one and six')
    with (OUT / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = checked_json(OUT / 'source_manifest.json')
        accepted, pending = completed_jobs(), []
        for request in requests:
            scene = source['scenes'][request['scene_index']]
            job = make_job(scene, request['arm'], 0, protocol)
            if job['id'] in accepted:
                continue
            attempts = sorted(Path(job['directory']).parent.glob('attempt_*'))
            if attempts:
                require(len(attempts) == 1 and (attempts[0] / 'exit_receipt.json').exists(),
                        f'unresolved prior attempt: {job["id"]}; inspect, never rerun blindly')
                require(read(attempts[0] / 'exit_receipt.json').get('retryable'), 'prior attempt not retryable')
                job = make_job(scene, request['arm'], 1, protocol)
            pending.append((scene, job))
        active, stopped = {}, None
        baseline = resources(); last_status = 0.; last_resource = 0.
        try:
            while pending or active:
                sample = resources()
                if sample['oom_kill'] > baseline['oom_kill']:
                    stopped = 'OOM counter increased; drain workers and inspect'
                if time.monotonic() - last_resource >= 30:
                    append(OUT / 'resources.jsonl', dict(sample, active=len(active), pending=len(pending)))
                    last_resource = time.monotonic()
                if time.monotonic() - last_status >= 45:
                    accepted = completed_jobs()
                    status = dict(utc=now(), stage='evaluating', pending=len(pending),
                        active=[dict(pid=pid, id=e[2]['id'], directory=e[2]['directory']) for pid,e in active.items()],
                        accepted={arm:sum(r['arm']==arm for r in accepted.values()) for arm in (*ARMS,'LIVE_PREFLIGHT')},
                        resource_guard=not capacity_ok(sample), stopped=stopped)
                    atomic(OUT / 'progress.json', status)
                    print(json.dumps(status), flush=True); last_status = time.monotonic()
                for pid, entry in list(active.items()):
                    process, scene, job, log, start, killed = entry
                    if process.poll() is None and time.monotonic() - start > 3600:
                        if killed is None:
                            os.killpg(pid, signal.SIGTERM); entry[-1] = time.monotonic()
                        elif time.monotonic() - killed > 15:
                            os.killpg(pid, signal.SIGKILL)
                    rc = process.poll()
                    if rc is None:
                        continue
                    directory = Path(job['directory'])
                    failure = read(directory / 'failure.json') if (directory / 'failure.json').exists() else {}
                    retryable = bool(entry[-1] is not None or failure.get('retryable', False))
                    freeze(directory / 'exit_receipt.json', dict(utc=now(), id=job['id'], attempt=job['attempt'],
                        pid=pid, returncode=rc, observed_by_parent=True, elapsed_s=time.monotonic()-start,
                        timed_out=entry[-1] is not None, retryable=retryable, failure=failure))
                    log.close(); del active[pid]
                    if rc == 0:
                        result = read(directory / 'result.json')
                        require(result['status'] in ('complete','policy_failure') and
                                result['schedule_row_sha256'] == job['sha256'], 'invalid scientific outcome')
                        append(OUT / 'valid_ledger.jsonl', dict(id=job['id'], arm=job['experiment_arm'],
                            scene_id=scene['scene_id'], directory=str(directory),
                            receipt=str(directory/'exit_receipt.json'), receipt_sha256=sha(directory/'exit_receipt.json'),
                            result_path=str(directory/'result.json'), result_sha256=sha(directory/'result.json')))
                    elif retryable and job['attempt'] == 0:
                        pending.insert(0, (scene, make_job(scene, job['experiment_arm'], 1, protocol)))
                    else:
                        stopped = f'nonretryable/second technical failure: {job["id"]}'
                if stopped and not active:
                    raise RuntimeError(stopped)
                while not stopped and pending and len(active) < workers and capacity_ok(sample):
                    scene, job = pending.pop(0)
                    directory = inside(job['directory']); directory.mkdir(parents=True, exist_ok=False)
                    freeze(directory/'job.json', job)
                    log = (directory/'worker.log').open('x')
                    process = subprocess.Popen([str(PYTHON), str(CODE/'pact_v1010c_vl53l8ux_16x16_128d_eval.py'),
                        '--job', str(directory/'job.json')], cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    active[process.pid] = [process, scene, job, log, time.monotonic(), None]
                    time.sleep(3); sample = resources()
                if pending and not active and not capacity_ok(sample):
                    raise RuntimeError('Resource guard prevents launch; preserve pending jobs and inspect capacity')
                if pending or active:
                    time.sleep(2)
        finally:
            for pid, entry in active.items():
                process, _, job, log, start, _ = entry
                if process.poll() is None:
                    os.killpg(pid, signal.SIGTERM)
                try:
                    rc = process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(pid, signal.SIGKILL); rc = process.wait()
                log.close()
                path = Path(job['directory']) / 'exit_receipt.json'
                if not path.exists():
                    freeze(path, dict(utc=now(), id=job['id'], attempt=job['attempt'], pid=pid,
                        observed_by_parent=True, returncode=rc, retryable=False,
                        elapsed_s=time.monotonic()-start, supervisor_interrupted=True))



def preflight():
    source, _, _ = verify_inputs()
    tests = read(OUT/'preflight/unit_tests.json')
    require(tests['passed'] and tests['implementation_hashes'] == implementation_hashes() and
        tests['test_sha256'] == sha(ROOT/tests['test_path']), 'unit-test/code mismatch')
    evidence = OUT / 'preflight/offline.json'
    if not evidence.exists():
        evidence.parent.mkdir(parents=True, exist_ok=True)
        with (evidence.parent/'offline.log').open('a') as log:
            subprocess.run([str(PYTHON), str(CODE/'pact_v1010c_vl53l8ux_16x16_128d_eval.py'), '--offline'],
                cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    offline = read(evidence)
    require(offline['passed'] and offline['implementation_hashes'] == implementation_hashes(), 'offline/code mismatch')
    freeze(OUT/'preflight/report.json', dict(passed=True, implementation_hashes=implementation_hashes(),
        unit_tests_sha256=sha(OUT/'preflight/unit_tests.json'),
        offline_sha256=sha(evidence),source_manifest_sha256=source['sha256'],
        reference_full_rollouts_reaudited=50,
        interpretation='105-step controller identity and 10-step native 16x16 interface checks, with physical matching to the original source. All source rollouts reaudited; no redundant full source replay or outcome selection.'))
    print(json.dumps(dict(stage='preflight_passed')), flush=True)



def summarize(rows):
    import numpy as np
    full = [r for r in rows if not r.get('censored')]
    denominator = sum(r['physics_samples'] for r in full)
    return dict(n=len(rows), task=sum(r['task_success'] for r in rows),
        strict=sum(r['collision_free_task_success'] for r in rows), full_contact_rollouts=len(full),
        any_forbidden=sum(r['any_forbidden_contact'] for r in full),
        forbidden_frames=sum(r['forbidden_frames'] for r in full), physics_samples=denominator,
        forbidden_percent=100*sum(r['forbidden_frames'] for r in full)/denominator if denominator else None,
        mean_forbidden_frames=float(np.mean([r['forbidden_frames'] for r in full])) if full else None,
        target_touch=sum(r.get('target_touch',False) for r in full),
        target_lifted_1cm=sum(r.get('target_lifted_1cm',False) for r in full),
        contact_frames={c:sum(r['contact_frames'][c] for r in full) for c in FORBIDDEN},
        contact_episodes={c:sum(r['contact_frames'][c]>0 for r in full) for c in FORBIDDEN})


def freeze_protocol():
    import importlib.metadata
    source, profile, schedule = verify_inputs()
    pre = read(OUT/'preflight/report.json')
    require(pre['passed'] and pre['implementation_hashes'] == implementation_hashes(), 'preflight/code drift')
    bindings = dict(implementation_hashes=implementation_hashes(), sensor_profile_sha256=profile['sha256'],
        schedule_sha256=schedule['sha256'], source_manifest_sha256=source['sha256'],
        preflight_sha256=sha(OUT/'preflight/report.json'))
    path = OUT/'protocol.json'
    if path.exists():
        doc = checked_json(path)
        require(all(doc[k] == v for k,v in bindings.items()), 'frozen protocol drift')
        return doc
    require(not any(r['arm'] in ARMS for r in completed_jobs().values()), 'outcomes precede protocol')
    doc = self_hashed(dict(schema=SCHEMA, frozen_utc=now(), **bindings,
        training_seed=3103, checkpoint_selection='Original final update 60000, unchanged',
        checkpoint_hashes=EXPECTED, n_scenarios=N, scientific_rollouts=N, arms=list(ARMS),
        historical_live_reused=True, exposed_scenes=True, no_training=True,
        study_origin='exploratory follow-up selected after five sensor studies, including negative 2x2 results; all 50 exposed source scenes retained',
        primary='collision_free_task_success', primary_contrast='L8UX-inspired native 16x16 with fixed 8x8 minimum adapter minus original live PACT',
        hypothesis='same frozen policy/encoder retains performance under an idealized sensor-model change',
        observation=dict(camera=['wrist_camera'], camera_size=[240,320], sensors=40, readout_dim=128,
            history=8, pool='min', sensor_order='original canonical kinematic order',
            native_shape=[16,16], policy_spatial_shape=[8,8],
            adapter=PROFILE['adapter']),
        execution=dict(horizon=900, chunk=100, aggregation_history=100, query_every_control_step=True,
            early_success_stop=False, action_noise=False, physics_dt_s=.002,
            control_interval_s=.066, observation_poll_offsets_s=[.016,.032,.048,.064],
            native_frequency_hz=15, native_clock='t0 + ceil(100*k/3)*.002 seconds; complete grids acquired, no future values'),
        forbidden_classes=list(FORBIDDEN), allowed_classes=['grasp_target','place_receptacle'],
        statistics=dict(unit='matched scenario, conditional on this single training seed',
            retention_margin_absolute=MARGIN, confidence_level=.95,
            retention='one-sided conservative exact 95% lower bound for transfer minus live exceeds -0.10',
            retention_bound='L(gain probability) minus U(loss probability); one-sided 97.5% Clopper-Pearson components; union-bound coverage >=95%',
            degradation='one-sided conservative exact 95% upper bound is below -0.10',
            otherwise='inconclusive about retention; report point estimates, gains/losses and intervals',
            secondary='placement/contact and exposure metrics descriptive/exploratory; not independent confirmatory tests',
            bootstrap_replicates=20000, bootstrap_seed=ANALYSIS_SEED,
            descriptive_interval='pointwise paired bootstrap 95%', exact_difference_test='two-sided McNemar',
            nonsignificance_is_not_retention=True, simulator_samples_are_not_trials=True),
        failures=dict(worker_timeout_s=3600, technical_retries=1, retry_same_inputs=True,
            retry_policy_failure=False, invalid_policy_output='task/strict failure; contacts censored',
            unresolved_technical='incomplete, keep scene in planned denominator; never omit it',
            feasibility='all historical accepted scenarios; no scene rejection using shifted policy performance'),
        resource_limits=dict(memory_fraction=.80,pid_fraction=.75,vram_fraction=.85,disk_reserve_gib=4),
        source_documentation=PROFILE['sources'], documented_features=PROFILE['documented_features'],
        modeling_assumptions=PROFILE['assumed_features'],
        versions={p:importlib.metadata.version(p) for p in ('torch','torchvision','numpy','scipy','h5py','mujoco')},
        repository_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        limitation='Exploratory idealized L8UX-inspired profile study, one training seed and 50 previously exposed scenes. Public ST presentations document 16x16 zones, nominal 65-degree FoV, 3m dark range and advertised 60fps capability; no detailed public L8UX datasheet was verified. Square-pinhole diagonal interpretation, 15Hz experimental acquisition, 2cm floor, 1mm rounding, optical-axis depth and no-return conventions are declared assumptions. Fixed nonoverlapping 2x2 minima map native 16x16 to the frozen encoder 8x8 input. This is a complete profile-plus-adapter transfer with multiple simultaneous changes, not a pure resolution test, physical hardware validation, or unseen-environment generalization. It cannot attribute benefits to fine-tuning or prove extra resolution helps. All positive, negative and inconclusive outcomes must be retained.'))
    freeze(path, doc)
    return doc


def retention_statistics(live, transferred, margin=MARGIN):
    import numpy as np
    from scipy.stats import beta
    a,b=np.asarray(live,dtype=int),np.asarray(transferred,dtype=int)
    require(a.ndim == 1 and a.shape == b.shape and len(a)>0 and
        np.isin(a,[0,1]).all() and np.isin(b,[0,1]).all(), 'invalid paired binary outcomes')
    n=len(a); gains=int(((b==1)&(a==0)).sum()); losses=int(((a==1)&(b==0)).sum())
    def lower(k): return float(beta.ppf(.025,k,n-k+1)) if k else 0.
    def upper(k): return float(beta.ppf(.975,k+1,n-k)) if k<n else 1.
    lo,hi=lower(gains)-upper(losses),upper(gains)-lower(losses)
    descriptive=paired_statistics(b,a)
    descriptive.pop('live_only',None); descriptive.pop('permuted_only',None)
    return dict(descriptive, transfer_only=gains, live_only=losses, margin_absolute=margin,
        retention_lower95=lo, degradation_upper95=hi,
        # Each is a one-sided >=95% bound; the pair is NOT a two-sided 95% CI.
        decision='RETENTION_SUPPORTED' if lo > -margin else
                 'LOSS_EXCEEDS_MARGIN' if hi < -margin else 'INCONCLUSIVE_RETENTION')


def analyze():
    source, _, _ = verify_inputs()
    protocol=freeze_protocol(); valid=completed_jobs()
    live={r['scene_id']:r for r in read(OUT/'live_metrics.json')['rows']}
    rows,missing=[],[]
    for scene in source['scenes']:
        row=dict(scene_id=scene['scene_id'], scene_index=scene['index'], live=live[scene['scene_id']])
        job=make_job(scene,ARMS[0],0,protocol['sha256'])
        if job['id'] not in valid:
            missing.append(job['id']); rows.append(row); continue
        record=valid[job['id']]; directory=Path(record['directory'])
        actual=checked_json(directory/'job.json')
        require(actual == make_job(scene,ARMS[0],actual['attempt'],protocol['sha256']), 'job drift')
        result=read(directory/'result.json')
        require(read(directory/'initial_pairing.json')['passed'], 'unmatched scene')
        if result['status']=='policy_failure':
            row['transfer']=dict(task_success=False,collision_free_task_success=False,censored=True,
                directory=str(directory),failure=result['failure'])
        else:
            info=result['policy_info']; audit=info['native_sensor_audit']
            require(info['unchanged_weights'] and info['initial_pairing_passed'] and
                audit['consumed_control_frames']==audit['full_grid_delivery_checks']==HORIZON and
                audit['native_acquisitions_per_sensor']==892 and audit['native_zones_per_sensor']==256 and audit['spatial_minimum_checks']==HORIZON and
                audit['observation_polls_per_sensor']==3601 and audit['sampling_clock_exact'], 'sensor audit incomplete')
            row['transfer']=audit_metrics(directory)
            row['transfer']['native_sensor_sha256']=sha(directory/'native_multizone.h5')
        rows.append(row)
    subset=[r for r in rows if 'transfer' in r]
    primary=retention_statistics([r['live']['collision_free_task_success'] for r in subset],
        [r['transfer']['collision_free_task_success'] for r in subset]) if subset else None
    if primary and missing: primary['decision']='INCOMPLETE'
    endpoints=('task_success','any_forbidden_contact','forbidden_frames','forbidden_fraction',
        'hazard_contact_episode','clutter_contact_episode','hazard_frames','clutter_frames','target_touch','target_lifted_1cm')
    comparisons={}
    for metric in endpoints:
        available=[r for r in subset if metric in r['transfer']]
        if available:
            result=paired_statistics([r['transfer'][metric] for r in available],[r['live'][metric] for r in available])
            if 'live_only' in result:
                result['transfer_only']=result.pop('live_only'); result['live_only']=result.pop('permuted_only')
            comparisons[metric]=result
    report=dict(schema=SCHEMA,complete=not missing,protocol_sha256=protocol['sha256'],missing_jobs=missing,
        primary=primary,comparisons=comparisons,rows=rows,
        summaries={a:summarize([r[a] for r in rows if a in r]) for a in ('live','transfer')},
        contact_censoring_count=sum(r['transfer'].get('censored',False) for r in subset),
        interpretation_limit=protocol['limitation'])
    atomic(OUT/'analysis.json',report); write_report(report)
    print(json.dumps(dict(stage='analyzed',complete=report['complete'],primary=primary,summaries=report['summaries'])),flush=True)
    return report


def write_report(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms=('live','transfer'); labels=('Original live','L8UX-inspired 16x16 → 8x8')
    doc=['# PACT-128D L8UX-inspired 16x16 sensor-profile transfer: seed 3103','',
        f"Status: **{'complete' if report['complete'] else 'incomplete'}**; 50 source scenarios, unchanged weights.",'',
        'Idealized L8UX-inspired native 16x16 profile with fixed 2x2 minimum pooling. Undocumented settings are explicit assumptions; this is not hardware validation.','',
        '| Outcome | Original live | L8UX-inspired 16x16 → 8x8 |','|---|---:|---:|']
    for title,key in [('Placement success','task'),('Collision-free placement (primary)','strict'),
                      ('Any forbidden contact','any_forbidden'),('Target touch','target_touch'),('Target lift ≥1 cm','target_lifted_1cm')]:
        values=[]
        for arm in arms:
            s=report['summaries'][arm]; n=s['n'] if key in ('task','strict') else s['full_contact_rollouts']
            values.append(f'{s[key]}/{n} ({100*s[key]/n:.1f}%)' if n else 'Pending')
        doc.append('| '+title+' | '+' | '.join(values)+' |')
    for title,key in [('Forbidden-contact samples','forbidden_frames'),('Audited physics samples','physics_samples'),
                      ('Forbidden-contact exposure (%)','forbidden_percent')]:
        doc.append('| '+title+' | '+' | '.join(str(report['summaries'][a][key]) for a in arms)+' |')
    p=report['primary']
    if p:
        doc += ['',f"Primary difference (transfer − live): **{100*p['difference']:+.1f} pp**, "
            f"descriptive paired bootstrap 95% CI [{100*p['ci95'][0]:+.1f}, {100*p['ci95'][1]:+.1f}] pp. "
            f"Transfer-only successes: {p['transfer_only']}; live-only successes: {p['live_only']}; "
            f"two-sided exact McNemar p={p['exact_mcnemar_p']:.6g}.",'',
            f"Prespecified maximum loss: **10 pp**. The conservative one-sided 95% lower bound is "
            f"{100*p['retention_lower95']:+.2f} pp. Decision: **{p['decision']}**.",'',
            'The retention decision uses the exact gain/loss bound, not a nonsignificant difference test. '
            'The paired bootstrap interval is descriptive. These results condition on one trained checkpoint pair.']
    doc += ['',report['interpretation_limit'],'',
        f"Contact-censored policy failures: {report['contact_censoring_count']}. Scientific failures are retained. "
        'Contact exposure uses complete telemetry; simulator samples are not independent trials.','',
        '[Protocol](protocol.json) · [Sensor profile](sensor_profile.json) · [Analysis](analysis.json) · [Paired results](paired_results.csv) · [Preflight](preflight/report.json)','',
        '![Sensor transfer results](results.png)','']
    (OUT/'RESULTS.md').write_text('\n'.join(doc))
    keys=('task_success','collision_free_task_success','any_forbidden_contact','forbidden_frames','physics_samples','directory')
    fields=['scene_id']+[f'{a}_{k}' for a in arms for k in keys]
    with (OUT/'paired_results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for row in report['rows']:
            writer.writerow(dict(scene_id=row['scene_id'],**{f'{a}_{k}':row[a].get(k) for a in arms if a in row for k in keys}))
    fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for i,(key,label) in enumerate([('task','Placement'),('strict','Collision-free placement')]):
        values=[report['summaries'][a][key]/report['summaries'][a]['n'] if report['summaries'][a]['n'] else 0 for a in arms]
        axs[0].bar([x+(i-.5)*.35 for x in range(2)],values,.35,label=label)
    axs[0].set(xticks=[0,1],xticklabels=labels,ylim=(0,1),ylabel='Fraction of matched scenarios'); axs[0].legend()
    for row in report['rows']:
        if all(a in row and 'forbidden_frames' in row[a] for a in arms):
            axs[1].plot([0,1],[row[a]['forbidden_frames'] for a in arms],color='0.5',alpha=.3,marker='o',markersize=2)
    axs[1].set(xticks=[0,1],xticklabels=labels,ylabel='Forbidden-contact physics samples')
    axs[1].set_yscale('symlog',linthresh=10)
    fig.suptitle('Seed 3103 · frozen policy and encoder · idealized L8UX-inspired 16x16 sensor-profile transfer')
    fig.savefig(OUT/'results.png',dpi=180); fig.savefig(OUT/'results.pdf'); plt.close(fig)


def completion_review(report):
    """Recompute aggregation, delivered inputs, pairing and endpoints from disk."""
    import hashlib
    import h5py
    import numpy as np
    from pact_vl53l8ux_16x16_sensor import acquisition_tick, policy_grid
    source, _, _ = verify_inputs()
    protocol = checked_json(OUT/'protocol.json')
    records = completed_jobs()
    require(report['complete'] and len(report['rows']) == N, 'incomplete final review')
    rows = []
    for row in report['rows']:
        scene = source['scenes'][row['scene_index']]
        directory = Path(row['transfer']['directory'])
        job = checked_json(directory/'job.json')
        require(job == make_job(scene, ARMS[0], job['attempt'], protocol['sha256']), 'final job drift')
        require(Path(records[job['id']]['directory']) == directory, 'final ledger drift')
        pairing = read(directory/'initial_pairing.json')
        require(pairing['passed'] and not pairing['violations'] and
            sha(pairing['reference']) == pairing['reference_sha256'] and
            sha(pairing['comparison']) == pairing['comparison_sha256'], 'final pairing drift')
        result = read(directory/'result.json')
        if result['status'] == 'policy_failure':
            rows.append(dict(scene_id=row['scene_id'], attempt=job['attempt'],
                scientific_failure_retained=True, contacts_censored=True))
            continue
        info = result['policy_info']
        require(info['inference_only'] and info['unchanged_weights'], 'inference state invalid')
        require(sha(directory/'native_multizone.h5') == row['transfer']['native_sensor_sha256'],
                'native data drift')
        with h5py.File(directory/'native_multizone.h5', 'r') as h:
            values = h['native_distance_m'][()]
            aggregated = h['pooled_distance_m'][()]
            times = h['native_sim_time_s'][()]
            polls = h['observation_poll_sim_time_s'][()]
            indices = h['observation_poll_native_index'][()]
            require(values.shape == (892,40,16,16) and np.isfinite(values).all(), 'native shape/value error')
            require(aggregated.shape == (892,40,8,8), 'aggregate shape error')
            # Independent block slices avoid merely repeating the adapter implementation.
            for y in range(8):
                for x in range(8):
                    expected = values[...,2*y:2*y+2,2*x:2*x+2].min(axis=(-2,-1))
                    require(np.array_equal(aggregated[...,y,x], expected), 'block minimum mismatch')
            ticks = np.array([acquisition_tick(i) for i in range(892)])
            require(np.allclose(times-times[0], ticks*.002, atol=1e-8, rtol=0), 'native clock drift')
            expected_polls = np.r_[0., (np.arange(900)[:,None]*.066+
                                       np.array([.016,.032,.048,.064])).ravel()]
            require(np.allclose(polls-times[0], expected_polls, atol=1e-8, rtol=0), 'poll clock drift')
            require(np.array_equal(indices, np.searchsorted(times, polls+1e-9, side='right')-1),
                    'future/stale acquisition')
            # Reconstruct every consumed raw tensor, including four copies of the reset frame.
            delivered_hash = hashlib.sha256()
            for step in range(HORIZON):
                chosen = np.repeat(indices[0],4) if step == 0 else indices[1+4*(step-1):1+4*step]
                raw = policy_grid(values[chosen]).transpose(1,0,2,3)
                delivered_hash.update(np.ascontiguousarray(raw).tobytes())
            require(delivered_hash.hexdigest() == info['input_tensor_sha256'],
                    'saved native values do not reproduce actual encoder inputs')
        audited = audit_metrics(directory)
        require(all(audited[k] == row['transfer'][k] for k in ('task_success',
            'collision_free_task_success','forbidden_frames','physics_samples','contact_frames')),
            'final endpoint discrepancy')
        rows.append(dict(scene_id=row['scene_id'], attempt=job['attempt'], receipt_verified=True,
            initial_pairing_verified=True, metrics_verified=True,
            native_acquisition_verified=True, block_minimum_verified=True,
            complete_delivered_input_hash_verified=True))
    atomic(OUT/'completion_review.json', dict(utc=now(), complete=True, scientific_rollouts=N,
        technical_retries=sum(r['attempt'] for r in rows), contact_censoring=report['contact_censoring_count'],
        analysis_sha256=sha(OUT/'analysis.json'), protocol_sha256=protocol['sha256'],
        implementation_hashes=implementation_hashes(), rows=rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','preflight','run','analyze','all'))
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if args.command in ('prepare','all'): prepare()
    if args.command in ('preflight','all'): preflight()
    if args.command in ('run','all'):
        protocol=freeze_protocol(); schedule=checked_json(OUT/'schedule.json')['rows']
        run_jobs(schedule[:1],1,protocol['sha256'])
        run_jobs(schedule,args.workers,protocol['sha256'])
        report=analyze(); require(report['complete'],'study incomplete')
        completion_review(report)
        freeze(OUT/'finished.json',dict(utc=now(),scientific_rollouts=N,analysis_sha256=sha(OUT/'analysis.json')))
        atomic(OUT/'progress.json',dict(utc=now(),stage='complete',pending=0,active=[],
            accepted={arm:sum(r['arm']==arm for r in completed_jobs().values()) for arm in (*ARMS,'LIVE_PREFLIGHT')},
            analysis_sha256=sha(OUT/'analysis.json')))
    elif args.command=='analyze': analyze()


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        import traceback
        OUT.mkdir(parents=True,exist_ok=True)
        append(OUT/'supervisor_errors.jsonl',dict(utc=now(),type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
