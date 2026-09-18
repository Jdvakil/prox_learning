#!/usr/bin/env python3
"""Frozen seed-3103 live-versus-unrelated-readout study; never trains a model.

The source run and its artifacts are read-only.  Only this study's output root
is writable.  ``prepare`` builds donors, ``preflight`` checks/replays the live
adapter, ``run`` freezes the protocol and evaluates, and ``analyze`` audits both
arms.  The historical 32-D frozen_directory is deliberately never used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'scripts'
SOURCE = ROOT / 'diagnostics_output/pact_place_v1010c_readout_s3103'
DATA = ROOT / 'diagnostics_output/pact_place_v1010_wrist288_s3_v1'
OUT = ROOT / 'diagnostics_output/pact_place_v1010c_permuted_s3103_v1'
PYTHON = Path('/root/act_retrain_venv/bin/python')
SCHEMA = 'pact_v1010c_permuted_128d_s3103_v1'
DONOR_SEED, ANALYSIS_SEED = 2026091603, 2026091604
N, HORIZON, SENSORS, DIM = 50, 900, 40, 128
FORBIDDEN = ('hazard_bar', 'clutter', 'other_environment', 'mounted_fixture')
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
    'OMP_THREAD_LIMIT', 'BLIS_NUM_THREADS', 'RAYON_NUM_THREADS')}
os.environ.update(THREAD_ENV)
EXPECTED = {
    'policy_last.ckpt': '3d01957cd3d86bb95db13bb4b96db05ebe07794134e4f5f64b683205afd46dcd',
    'prox_encoder.pt': 'c1127d13cc195197c7375b5bb03081b65f9a44dabe477cb28d9408b5308e7cbc',
    'dataset_stats.pkl': 'c15e9673e5c619ba1c86d54ab45d8a90843ec296ed768dfeb4e588808aaae107',
}
IMPLEMENTATION = ('scripts/pact_v1010c_permuted_128d.py',
                  'scripts/pact_v1010c_permuted_128d_eval.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def inside(path):
    path = Path(path).resolve()
    require(path.is_relative_to(OUT.resolve()), f'write outside study: {path}')
    return path


def atomic(path, value):
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def freeze(path, value):
    path = inside(path)
    if path.exists():
        require(read(path) == value, f'immutable artifact mismatch: {path}')
    else:
        atomic(path, value)


def append(path, value):
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x] if Path(path).exists() else []


def self_hashed(doc):
    return {**doc, 'sha256': digest(doc)}


def checked_json(path):
    doc = read(path)
    require(doc['sha256'] == digest({k: v for k, v in doc.items() if k != 'sha256'}),
            f'self-hash mismatch: {path}')
    return doc


def implementation_hashes():
    return {p: sha(ROOT / p) for p in IMPLEMENTATION}


def environment():
    # Import establishes exactly the original module search order and EGL setup.
    import pact_v1010c_core as core
    env = core.environment()
    env['PACT_PERMUTED_128D_OWNER'] = SCHEMA
    return env


def verify_models():
    for name, expected in EXPECTED.items():
        require(sha(SOURCE / 'checkpoint' / name) == expected, f'checkpoint drift: {name}')
    pair = read(SOURCE / 'checkpoint/checkpoint_pairs.json')['policy_last.ckpt']
    require(pair['global_step'] == 60000 and pair['encoder'] == 'prox_encoder.pt', 'wrong checkpoint pair')
    require(pair['policy_sha256'] == EXPECTED['policy_last.ckpt'] and
            pair['encoder_sha256'] == EXPECTED['prox_encoder.pt'], 'wrong pair hashes')
    require(sha(DATA / 'dataset_stats.pkl') == EXPECTED['dataset_stats.pkl'], 'normalization drift')
    import pact_v1010c_core as core
    core.verify_upstream()
    # Check source evaluator and its original task/controller dependencies.
    for contract in (read(SOURCE / 'contract.json')['code_hashes'],
                     read(DATA / 'config.json')['file_hashes']):
        for path, expected in contract.items():
            require(sha(ROOT / path) == expected, f'source implementation drift: {path}')
    return pair


def contact_summary(counts, classes, task_success):
    """Union of all forbidden classes; overlapping contacts count only once."""
    import numpy as np
    counts = np.asarray(counts)
    require(counts.ndim == 2 and counts.shape[1] == len(classes) and len(counts) > 0,
            'invalid contact sample array')
    require(set(FORBIDDEN) <= set(classes), 'missing forbidden contact class')
    require(np.isfinite(counts).all() and (counts >= 0).all(), 'invalid contact counts')
    masks = counts > 0
    forbidden = masks[:, [classes.index(c) for c in FORBIDDEN]].any(axis=1)
    return dict(physics_samples=len(counts), forbidden_frames=int(forbidden.sum()),
        forbidden_fraction=float(forbidden.mean()), any_forbidden_contact=bool(forbidden.any()),
        collision_free=bool(not forbidden.any()),
        collision_free_task_success=bool(task_success and not forbidden.any()),
        contact_frames={c: int(masks[:, i].sum()) for i, c in enumerate(classes)})


def audit_metrics(directory):
    """One independent endpoint/contact implementation for historical and new runs."""
    import h5py
    import pact_v1010c_core  # establish original imports, without training
    from pact_wrist288_metrics import metrics
    directory = Path(directory)
    result = read(directory / 'result.json')
    raw = metrics(directory)  # reconstruct judge, contact identities, lift and touch
    with h5py.File(directory / 'telemetry.h5', 'r') as h:
        summary = contact_summary(h['contacts/class_entries'][()],
            json.loads(h['contacts'].attrs['class_names']), raw['task_success'])
    require(summary['physics_samples'] == 29701, 'incomplete full-horizon telemetry')
    require(summary['collision_free_task_success'] == raw['collision_free_task_success'],
            'strict endpoint disagreement')
    return {**raw, **summary, 'clutter_frames': summary['contact_frames']['clutter'],
            'directory': str(directory), 'scene_id': result['scene_id'],
            'physical_row_digest': result['physical_row_digest'],
            'result_sha256': sha(directory / 'result.json'),
            'experiment_arm': result.get('experiment_arm', 'LIVE_HISTORICAL')}


def prepare_reference():
    verify_models()
    manifest = checked_json(SOURCE / 'evaluation_manifest.json')
    rows = read(SOURCE / 'comparison.json')['readout_rows']
    by_scene = {r['scene_id']: r for r in rows}
    require(len(by_scene) == len(rows) == len(manifest['scenes']) == N, 'expected all 50 scenes')
    ledger = {r['result_path']: r for r in lines(SOURCE / 'valid_ledger.jsonl')}
    references, summaries = [], []
    for index, scene in enumerate(manifest['scenes']):
        row = by_scene[scene['scene_id']]
        directory = Path(row['directory'])
        result = read(directory / 'result.json')
        receipt_entry = ledger[str(directory / 'result.json')]
        receipt = read(receipt_entry['receipt'])
        require(receipt['observed_by_parent'] and receipt['returncode'] == 0, 'invalid historical exit')
        require(sha(receipt_entry['receipt']) == receipt_entry['receipt_sha256'], 'historical receipt drift')
        require(sha(directory / 'result.json') == row['result_sha256'] == receipt_entry['result_sha256'],
                'historical result drift')
        require(result['checkpoint_sha256'] == EXPECTED['policy_last.ckpt'] and
                result['surface_encoder_sha256'] == EXPECTED['prox_encoder.pt'] and
                result['stats_sha256'] == EXPECTED['dataset_stats.pkl'], 'not the finetuned 128-D run')
        require(result['variant'] == 'readout60000' and result['training_seed'] == 3103, 'wrong baseline')
        info = result['policy_info']
        require(info['proximity_feature_dim'] == 128 and info['prox_policy_tap'] == 'readout'
                and info['prox_pool'] == 'min' and info['averaging_history'] == 100, 'observation mismatch')
        require(result['seed'] == scene['selected_seed'], 'selected retry mismatch')
        require(result['physical_row_digest'] == digest(scene['physical_row']), 'physical row mismatch')
        raw = audit_metrics(directory)
        require(all(raw[k] == row[k] for k in ('task_success', 'collision_free_task_success', 'contact_frames')),
                'historical audit disagreement')
        summaries.append(raw)
        references.append(dict(index=index, scene_id=scene['scene_id'], physical_row=scene['physical_row'],
            selected_retry=scene['selected_retry'], selected_seed=scene['selected_seed'],
            live_directory=str(directory), live_receipt=receipt_entry,
            artifacts={name: sha(directory / name) for name in
                       ('result.json', 'initial_observation.h5', 'trajectory.h5', 'telemetry.h5', 'actions.npz')}))
    require(sum(r['task_success'] for r in summaries) == 27 and
            sum(r['collision_free_task_success'] for r in summaries) == 24, 'live reference totals differ')
    freeze(OUT / 'live_metrics.json', {'n': N, 'rows': summaries})
    source = self_hashed(dict(schema=SCHEMA, scenes=references, checkpoint_hashes=EXPECTED,
        original_manifest_sha256=sha(SOURCE / 'evaluation_manifest.json'),
        original_comparison_sha256=sha(SOURCE / 'comparison.json'),
        live_metrics_sha256=sha(OUT / 'live_metrics.json'),
        selection='All 50 original readout60000 records; exposed regression scenes, user-selected seed3103'))
    freeze(OUT / 'source_manifest.json', source)
    return source


def select_sources(episodes, timesteps, rows=N, horizon=HORIZON, seed=DONOR_SEED):
    """Original whole-frame algorithm, parameterized for the 128-D data split."""
    import numpy as np
    episodes, timesteps = np.asarray(episodes), np.asarray(timesteps)
    require(len(episodes) == len(timesteps) and len(episodes) >= horizon, 'insufficient donor frames')
    rng = np.random.default_rng(seed)
    selected = np.stack([rng.choice(len(episodes), size=horizon, replace=False) for _ in range(rows)])
    for row in selected:
        for pos in range(1, horizon):
            if episodes[row[pos]] == episodes[row[pos - 1]]:
                later = next((i for i in range(pos + 1, horizon)
                              if episodes[row[i]] != episodes[row[pos - 1]]), None)
                require(later is not None, 'cannot separate adjacent source episodes')
                row[pos], row[later] = row[later], row[pos]
    require(all(len(set(r)) == horizon for r in selected), 'repeated donor within row')
    require((episodes[selected][:, 1:] != episodes[selected][:, :-1]).all(), 'adjacent episode repeated')
    return episodes[selected], timesteps[selected]


def prepare_donors(source):
    import numpy as np
    import h5py
    import pact_v1010c_core as core
    import torch
    split = read(DATA / 'split_manifest.json')
    require(split['split_manifest_sha256'] == digest({k: v for k, v in split.items()
            if k != 'split_manifest_sha256'}), 'split manifest hash mismatch')
    training = sorted((e for e in split['episodes'] if e['split'] == 'train'),
                      key=lambda e: e['act_episode_index'])
    require(len(training) == 240, 'wrong training partition')
    require(not ({e['episode_id'] for e in training} & {s['scene_id'] for s in source['scenes']}),
            'donor and evaluation episode identities overlap')
    views = {e['file']: e for e in read(SOURCE / 'data_views_manifest.json')['episodes']}
    episodes, timesteps, provenance = [], [], []
    for ep in training:
        i = ep['act_episode_index']; name = f'episode_{i}.hdf5'; view = views[name]
        require(sha(SOURCE / 'data_views' / name) == view['view_sha256'], f'data view drift: {name}')
        length = view['shape'][0]
        require(view['shape'][1:] == [40, 8, 8], 'expected min-pooled views')
        episodes.extend([i] * length); timesteps.extend(range(length))
        provenance.append(dict(episode_index=i, episode_id=ep['episode_id'], frames=length,
                               path=str(SOURCE / 'data_views' / name), sha256=view['view_sha256']))
    require(len(episodes) == 117966, 'training frame population changed')
    selected_ep, selected_t = select_sources(episodes, timesteps)
    plan = self_hashed(dict(schema=SCHEMA, seed=DONOR_SEED, rows=N, horizon=HORIZON,
        encoder_sha256=EXPECTED['prox_encoder.pt'], shape=[N, HORIZON, SENSORS, DIM], dtype='float32',
        split_sha256=sha(DATA / 'split_manifest.json'), frame_population=len(episodes),
        source_manifest_sha256=source['sha256'], training_episodes=provenance,
        selected_episodes=selected_ep.tolist(), selected_timesteps=selected_t.tolist(),
        selection='Uniform frames without replacement within row, then separate adjacent source episodes',
        preprocessing='min over four subframes; causal eight-frame history; repeat-in-encoder; readout128'))
    freeze(OUT / 'donor_plan.json', plan)
    bank_path = OUT / 'permuted_tokens.npy'
    if (OUT / 'donor_bank.json').exists():
        bank = read(OUT / 'donor_bank.json')
        require(bank['plan_sha256'] == plan['sha256'] and sha(bank_path) == bank['tokens_sha256'],
                'frozen donor bank mismatch')
        return bank
    require(not bank_path.exists(), 'unpublished final bank exists; inspect before resuming')
    # An interrupted derived cache can be rebuilt from the same immutable selection.
    partial = OUT / 'permuted_tokens.partial.npy'
    bank = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32,
                                    shape=(N, HORIZON, SENSORS, DIM))
    flat_bank = bank.reshape(-1, SENSORS, DIM)
    ep_flat, t_flat = selected_ep.ravel(), selected_t.ravel()
    encoder = core.make_encoder(SOURCE / 'checkpoint/prox_encoder.pt', train=False)
    encoder.requires_grad_(False)
    started, unique_count = time.monotonic(), 0
    with torch.inference_mode():
        for ep in provenance:
            positions = np.flatnonzero(ep_flat == ep['episode_index'])
            if not len(positions):
                continue
            with h5py.File(ep['path'], 'r') as h:
                pooled = h['observations/proximity'][()]
                require(h.attrs['v1010c_proximity_pool'] == 'min', 'wrong pooling')
            require(np.isfinite(pooled).all(), 'nonfinite source proximity')
            for t in np.unique(t_flat[positions]):
                history = core.causal_pooled_window(pooled, int(t))
                features = core.encode_for_act(encoder, torch.from_numpy(history).cuda().unsqueeze(0))
                require(tuple(features.shape) == (1, SENSORS, DIM), 'wrong encoder readout')
                values = features.cpu().numpy()[0]
                require(np.isfinite(values).all(), 'nonfinite donor readout')
                flat_bank[positions[t_flat[positions] == t]] = values
                unique_count += 1
            bank.flush()
            atomic(OUT / 'donor_progress.json', dict(utc=now(), last_episode=ep['episode_index'],
                encoded_unique_frames=unique_count, elapsed_s=time.monotonic() - started))
            print(json.dumps({'stage': 'donors', 'episode': ep['episode_index'],
                              'unique_frames': unique_count}), flush=True)
    require(all(np.isfinite(bank[i]).all() for i in range(N)), 'incomplete donor bank')
    del flat_bank, bank, encoder
    torch.cuda.empty_cache()
    os.replace(partial, bank_path)
    result = dict(plan_sha256=plan['sha256'], tokens_sha256=sha(bank_path),
        shape=[N, HORIZON, SENSORS, DIM], dtype='float32', unique_frames=unique_count,
        encoder_sha256=EXPECTED['prox_encoder.pt'], elapsed_s=time.monotonic() - started,
        encoded_batch_size=1, no_training=True)
    freeze(OUT / 'donor_bank.json', result)
    return result


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    source = prepare_reference()
    bank = prepare_donors(source)
    print(json.dumps({'stage': 'prepared', 'live_task': 27, 'live_strict': 24, **bank}), flush=True)


def verify_inputs(full=True):
    verify_models()
    source = checked_json(OUT / 'source_manifest.json')
    plan = checked_json(OUT / 'donor_plan.json')
    bank = read(OUT / 'donor_bank.json')
    require(plan['source_manifest_sha256'] == source['sha256'] and
            bank['plan_sha256'] == plan['sha256'], 'source/donor pairing changed')
    require(sha(OUT / 'live_metrics.json') == source['live_metrics_sha256'], 'baseline drift')
    if full:
        require(sha(OUT / 'permuted_tokens.npy') == bank['tokens_sha256'], 'donor tensor hash mismatch')
        for scene in source['scenes']:
            for name, expected in scene['artifacts'].items():
                require(sha(Path(scene['live_directory']) / name) == expected, 'historical artifact changed')
    return source, plan, bank


def resources():
    from pact_v1010c_train import resource_sample
    return resource_sample()  # read-only NVML/cgroup observation; never invokes train()


def capacity_ok(sample):
    return (sample['ram_fraction'] < .80 and sample['pid_fraction'] < .75 and
            sample['vram_fraction'] < .85 and sample['disk_free_bytes'] > 10 * 2**30)


def make_job(scene, arm, attempt, protocol=None):
    job_id = f'{arm.lower()}_{scene["index"]:03d}_{scene["scene_id"][:12]}'
    directory = OUT / ('preflight' if arm == 'LIVE_PREFLIGHT' else 'rollouts') / job_id / f'attempt_{attempt:02d}'
    return self_hashed(dict(schema=SCHEMA, id=job_id, directory=str(directory), attempt=attempt,
        experiment_arm=arm, variant='readout60000', scene_id=scene['scene_id'], scene_index=scene['index'],
        checkpoint_sha256=EXPECTED['policy_last.ckpt'], encoder_sha256=EXPECTED['prox_encoder.pt'],
        source_manifest_sha256=checked_json(OUT / 'source_manifest.json')['sha256'],
        donor_plan_sha256=checked_json(OUT / 'donor_plan.json')['sha256'],
        donor_bank_sha256=read(OUT / 'donor_bank.json')['tokens_sha256'],
        implementation_hashes=implementation_hashes(), protocol_sha256=protocol))


def completed_jobs():
    valid = {}
    for record in lines(OUT / 'valid_ledger.jsonl'):
        require(record['id'] not in valid, 'duplicate accepted scientific attempt')
        receipt = read(record['receipt'])
        require(receipt['observed_by_parent'] and receipt['returncode'] == 0, 'unobserved worker exit')
        require(sha(record['receipt']) == record['receipt_sha256'] and
                sha(record['result_path']) == record['result_sha256'], 'completed record drift')
        valid[record['id']] = record
    return valid


def run_jobs(scenes, arm, workers, protocol=None):
    """Observed subprocess exits, immutable attempts, one transient technical retry."""
    import fcntl
    require(1 <= workers <= 6, 'workers must be between one and six')
    lock = (OUT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending = []
    accepted = completed_jobs()
    for scene in scenes:
        job = make_job(scene, arm, 0, protocol)
        if job['id'] in accepted:
            continue
        attempts = sorted((Path(job['directory']).parent).glob('attempt_*'))
        if attempts:
            require(len(attempts) == 1 and (attempts[0] / 'exit_receipt.json').exists(),
                    f'unresolved attempts for {job["id"]}')
            failure = read(attempts[0] / 'exit_receipt.json')
            require(failure.get('retryable', False), f'non-retryable prior attempt: {job["id"]}')
            job = make_job(scene, arm, 1, protocol)
        pending.append((scene, job))
    active, stopped = {}, None
    baseline = resources(); last_sample = 0.; last_progress = 0.
    try:
        while pending or active:
            sample = resources()
            if time.monotonic() - last_sample > 30:
                append(OUT / 'resources.jsonl', dict(sample, arm=arm, active=len(active), pending=len(pending)))
                last_sample = time.monotonic()
            if sample['oom_kill'] > baseline['oom_kill']:
                stopped = 'cgroup OOM observed; drain owned workers and investigate'
            if time.monotonic() - last_progress > 45:
                print(json.dumps({'stage': arm, 'active': len(active), 'pending': len(pending),
                    'accepted_total': len(completed_jobs()), 'utc': now(), 'resource_guard': not capacity_ok(sample)}), flush=True)
                last_progress = time.monotonic()
            for pid, entry in list(active.items()):
                process, scene, job, log, start, killed_at = entry
                if process.poll() is None and time.monotonic() - start > 3600:
                    if killed_at is None:
                        os.killpg(pid, signal.SIGTERM); entry[-1] = time.monotonic()
                    elif time.monotonic() - killed_at > 15:
                        os.killpg(pid, signal.SIGKILL)
                rc = process.poll()
                if rc is None:
                    continue
                log.close(); del active[pid]
                directory = Path(job['directory'])
                failure = read(directory / 'failure.json') if (directory / 'failure.json').exists() else {}
                retryable = bool(entry[-1] is not None or failure.get('retryable', False))
                receipt = dict(utc=now(), id=job['id'], attempt=job['attempt'], pid=pid,
                    returncode=rc, observed_by_parent=True, elapsed_s=time.monotonic() - start,
                    timed_out=entry[-1] is not None, retryable=retryable, failure=failure)
                freeze(directory / 'exit_receipt.json', receipt)
                if rc == 0:
                    result = read(directory / 'result.json')
                    require(result['status'] in ('complete', 'policy_failure'), 'missing scientific outcome')
                    require(result['schedule_row_sha256'] == job['sha256'], 'job/result mismatch')
                    record = dict(id=job['id'], arm=arm, scene_id=scene['scene_id'], directory=str(directory),
                        receipt=str(directory / 'exit_receipt.json'), receipt_sha256=sha(directory / 'exit_receipt.json'),
                        result_path=str(directory / 'result.json'), result_sha256=sha(directory / 'result.json'))
                    append(OUT / 'valid_ledger.jsonl', record)
                elif retryable and job['attempt'] == 0:
                    pending.insert(0, (scene, make_job(scene, arm, 1, protocol)))
                else:
                    stopped = f'non-retryable/second technical failure: {job["id"]}; {directory}/worker.log'
            if stopped and not active:
                raise RuntimeError(stopped)
            while not stopped and pending and len(active) < workers and capacity_ok(sample):
                scene, job = pending.pop(0)
                directory = inside(job['directory']); directory.mkdir(parents=True, exist_ok=False)
                freeze(directory / 'job.json', job)
                log = (directory / 'worker.log').open('x')
                command = [str(PYTHON), str(CODE / 'pact_v1010c_permuted_128d_eval.py'),
                           '--job', str(directory / 'job.json')]
                process = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                active[process.pid] = [process, scene, job, log, time.monotonic(), None]
                # Avoid a burst of renderer initialization before resource use is visible.
                time.sleep(3)
                sample = resources()
            if pending and not active and not capacity_ok(sample):
                raise RuntimeError('resource guard prevents launch; rerun when capacity is available')
            if pending or active:
                time.sleep(2)
    finally:
        for pid, (process, *_) in active.items():
            if process.poll() is None:
                os.killpg(pid, signal.SIGTERM)
        for process, _, _, log, _, _ in active.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            log.close()
        lock.close()


def preflight():
    source, _, _ = verify_inputs()
    evidence = OUT / 'preflight/offline.json'
    if not evidence.exists():
        evidence.parent.mkdir(parents=True, exist_ok=True)
        command = [str(PYTHON), str(CODE / 'pact_v1010c_permuted_128d_eval.py'), '--offline']
        with (evidence.parent / 'offline.log').open('a') as log:
            subprocess.run(command, cwd=ROOT, env=environment(), stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=1800)
    offline = read(evidence)
    require(offline['passed'] and offline['implementation_hashes'] == implementation_hashes(),
            'offline checks missing or code changed')
    run_jobs(source['scenes'][:1], 'LIVE_PREFLIGHT', 1)
    valid = [r for r in completed_jobs().values() if r['arm'] == 'LIVE_PREFLIGHT']
    require(len(valid) == 1, 'expected one live preflight')
    result = read(valid[0]['result_path'])
    require(result['status'] == 'complete', 'live infrastructure preflight did not finish')
    require(read(Path(valid[0]['directory']) / 'initial_pairing.json')['passed'], 'initial replay mismatch')
    historical = read(Path(source['scenes'][0]['live_directory']) / 'result.json')
    report = dict(passed=True, implementation_hashes=implementation_hashes(), offline_sha256=sha(evidence),
        live_record=valid[0], historical_task=historical['task_success'], replay_task=result['task_success'],
        historical_strict=historical['collision_free_task_success'], replay_strict=result['collision_free_task_success'],
        interpretation='Physical/input/functional parity is required; no selection by replay outcome.')
    freeze(OUT / 'preflight/report.json', report)
    print(json.dumps(report), flush=True)


def freeze_protocol():
    source, plan, bank = verify_inputs()
    pre = read(OUT / 'preflight/report.json')
    require(pre['passed'] and pre['implementation_hashes'] == implementation_hashes(), 'preflight/code mismatch')
    path = OUT / 'protocol.json'
    if path.exists():
        protocol = checked_json(path)
        require(protocol['implementation_hashes'] == implementation_hashes(), 'scientific code changed')
        require(protocol['source_manifest_sha256'] == source['sha256'] and
                protocol['donor_plan_sha256'] == plan['sha256'] and
                protocol['donor_bank_sha256'] == bank['tokens_sha256'], 'scientific inputs changed')
        return protocol
    require(not any(r['arm'] == 'PACT_PERMUTED_128D' for r in completed_jobs().values()),
            'scientific outcomes precede protocol')
    versions = {}
    for package in ('torch', 'torchvision', 'numpy', 'scipy', 'h5py', 'mujoco'):
        versions[package] = importlib.metadata.version(package)
    protocol = self_hashed(dict(schema=SCHEMA, frozen_utc=now(), training_seed=3103,
        checkpoint_selection='Original final update 60000, unchanged', checkpoint_hashes=EXPECTED,
        implementation_hashes=implementation_hashes(), versions=versions,
        repository_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_manifest_sha256=source['sha256'], donor_plan_sha256=plan['sha256'],
        donor_bank_sha256=bank['tokens_sha256'], preflight_sha256=sha(OUT / 'preflight/report.json'),
        n_pairs=N, historical_live_reused=True, exposed_scenes=True, repeats_per_arm=1,
        intervention='One unrelated complete 40x128 final-encoder frame at every control step',
        observation=dict(camera=['wrist_camera'], camera_size=[240, 320], sensor_count=40,
            proximity_pool='min', history=8, tap='readout', no_normalization_refit=True),
        execution=dict(horizon=900, chunk=100, aggregation_history=100, query_every_control_step=True,
            early_success_stop=False, action_noise=False, physics_dt_s=.002),
        primary='collision_free_task_success', forbidden_classes=list(FORBIDDEN),
        allowed_contact_classes=['grasp_target', 'place_receptacle'],
        secondary=['task_success', 'any_forbidden_contact', 'forbidden_frames', 'hazard_frames',
                   'clutter_frames', 'target_touch', 'target_lifted_1cm'],
        statistics=dict(unit='matched scenario', bootstrap_replicates=20000, bootstrap_seed=ANALYSIS_SEED,
            difference='live minus permuted', confidence_level=.95, primary_test='two-sided exact McNemar',
            positive_primary='difference CI lower > 0 AND exact p < .05', multiple_testing='secondary exploratory'),
        failures=dict(technical_retries=1, worker_timeout_s=3600, retry_same_inputs=True,
            retry_policy_failure=False, unresolved_technical='incomplete study',
            invalid_policy_output='task and strict failure; partial contact exposure censored'),
        limitation='One selected checkpoint, exposed scenes, one donor plan; temporal and cross-modal correspondence both disrupted'))
    freeze(path, protocol)
    return protocol


def paired_statistics(live, permuted):
    import numpy as np
    from scipy.stats import binomtest
    a, b = np.asarray(live, dtype=float), np.asarray(permuted, dtype=float)
    require(a.ndim == 1 and a.shape == b.shape and len(a) > 0, 'invalid paired arrays')
    differences = a - b
    draws = np.random.default_rng(ANALYSIS_SEED).integers(0, len(a), (20000, len(a)))
    lo, hi = np.quantile(differences[draws].mean(axis=1), [.025, .975])
    result = dict(n=len(a), difference=float(differences.mean()), ci95=[float(lo), float(hi)])
    if np.isin(a, [0, 1]).all() and np.isin(b, [0, 1]).all():
        live_only, permuted_only = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        result.update(live_only=live_only, permuted_only=permuted_only,
            exact_mcnemar_p=float(binomtest(live_only, live_only + permuted_only, .5).pvalue)
                if live_only + permuted_only else 1.)
    return result


def analyze():
    import numpy as np
    source, _, _ = verify_inputs()
    protocol = checked_json(OUT / 'protocol.json')
    require(protocol['implementation_hashes'] == implementation_hashes(), 'analysis code changed after freeze')
    valid = completed_jobs()
    candidates = {r['scene_id']: r for r in valid.values() if r['arm'] == 'PACT_PERMUTED_128D'}
    paired, missing = [], []
    for scene in source['scenes']:
        if scene['scene_id'] not in candidates:
            missing.append(scene['scene_id']); continue
        record = candidates[scene['scene_id']]
        directory = Path(record['directory']); result = read(directory / 'result.json')
        job = checked_json(directory / 'job.json')
        require(job['protocol_sha256'] == protocol['sha256'], 'result belongs to a different protocol')
        require(read(directory / 'initial_pairing.json')['passed'], 'unmatched scene')
        live = audit_metrics(scene['live_directory'])
        if result['status'] == 'policy_failure':
            permuted = dict(task_success=False, collision_free_task_success=False,
                censored=True, directory=str(directory), failure=result['failure'])
        else:
            permuted = audit_metrics(directory)
            require(result['policy_info']['replacement_projection_checks'] == HORIZON,
                    'donor consumption not verified')
        paired.append(dict(scene_id=scene['scene_id'], live=live, permuted=permuted))
    require(paired, 'no valid paired outcomes')
    endpoints = ('collision_free_task_success', 'task_success', 'any_forbidden_contact',
                 'forbidden_frames', 'forbidden_fraction', 'hazard_contact_episode',
                 'clutter_contact_episode', 'hazard_frames', 'clutter_frames',
                 'target_touch', 'target_lifted_1cm')
    comparisons = {}
    for key in endpoints:
        subset = [p for p in paired if key in p['permuted']]
        if subset:
            comparisons[key] = paired_statistics([p['live'][key] for p in subset],
                                                 [p['permuted'][key] for p in subset])
    summaries = {}
    for arm in ('live', 'permuted'):
        rows = [p[arm] for p in paired]
        full = [r for r in rows if not r.get('censored', False)]
        summaries[arm] = dict(n=len(rows), task=sum(r['task_success'] for r in rows),
            strict=sum(r['collision_free_task_success'] for r in rows), full_contact_rollouts=len(full),
            any_forbidden=sum(r['any_forbidden_contact'] for r in full),
            forbidden_frames=sum(r['forbidden_frames'] for r in full),
            physics_samples=sum(r['physics_samples'] for r in full),
            forbidden_percent=100 * sum(r['forbidden_frames'] for r in full) /
                sum(r['physics_samples'] for r in full) if full else None,
            mean_forbidden_frames=float(np.mean([r['forbidden_frames'] for r in full])) if full else None,
            median_forbidden_frames=float(np.median([r['forbidden_frames'] for r in full])) if full else None,
            contact_frames={c: sum(r['contact_frames'][c] for r in full) for c in FORBIDDEN},
            contact_episodes={c: sum(r['contact_frames'][c] > 0 for r in full) for c in FORBIDDEN},
            target_touch=sum(r.get('target_touch', False) for r in full),
            target_lifted_1cm=sum(r.get('target_lifted_1cm', False) for r in full))
    primary = comparisons['collision_free_task_success']
    if missing:
        decision = 'INCOMPLETE'
    elif primary['ci95'][0] > 0 and primary['exact_mcnemar_p'] < .05:
        decision = 'POSITIVE_PRIMARY'
    elif primary['ci95'][1] < 0 and primary['exact_mcnemar_p'] < .05:
        decision = 'NEGATIVE_PRIMARY'
    else:
        decision = 'INCONCLUSIVE_PRIMARY'
    report = dict(schema=SCHEMA, protocol_sha256=protocol['sha256'], n_pairs=len(paired),
        missing_scenes=missing, decision=decision, summaries=summaries, comparisons=comparisons,
        rows=paired, contact_censoring_count=sum(p['permuted'].get('censored', False) for p in paired),
        interpretation_limit=protocol['limitation'])
    atomic(OUT / 'analysis.json', report)
    write_report(report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)


def write_report(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    pairs, summaries = report['rows'], report['summaries']
    fields = ['scene_id'] + [f'{arm}_{key}' for arm in ('live', 'permuted') for key in
        ('task_success', 'collision_free_task_success', 'any_forbidden_contact', 'forbidden_frames',
         'physics_samples', 'target_touch', 'target_lifted_1cm', 'directory')]
    with (OUT / 'paired_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for pair in pairs:
            writer.writerow({'scene_id': pair['scene_id'], **{f'{arm}_{key}': value
                for arm in ('live', 'permuted') for key, value in pair[arm].items()
                if f'{arm}_{key}' in fields}})
    n = len(pairs)
    a, b = summaries['live'], summaries['permuted']
    primary = report['comparisons']['collision_free_task_success']
    contact = report['comparisons'].get('forbidden_frames')
    lines = ['# PACT-128D proximity dependence: seed 3103', '',
        f"Status: **{report['decision']}**. {n}/50 matched pairs; historical live results reused.", '',
        '| Outcome | Live | Unrelated embeddings |', '|---|---:|---:|']
    for label, key in (('Placement success', 'task'), ('Collision-free placement', 'strict'),
                       ('Any forbidden contact', 'any_forbidden'), ('Target touch', 'target_touch'),
                       ('Target lift ≥1 cm', 'target_lifted_1cm')):
        den_a = n if key in ('task', 'strict') else a['full_contact_rollouts']
        den_b = n if key in ('task', 'strict') else b['full_contact_rollouts']
        rate_a = f'{a[key]}/{den_a} ({100*a[key]/den_a:.1f}%)' if den_a else 'Not observed'
        rate_b = f'{b[key]}/{den_b} ({100*b[key]/den_b:.1f}%)' if den_b else 'Not observed'
        lines.append(f'| {label} | {rate_a} | {rate_b} |')
    lines += [f"| Forbidden contact samples | {a['forbidden_frames']:,} | {b['forbidden_frames']:,} |",
        f"| Audited physics samples | {a['physics_samples']:,} | {b['physics_samples']:,} |",
        f"| Forbidden-contact exposure (%) | {a['forbidden_percent']} | {b['forbidden_percent']} |", '',
        f"Collision-free placement difference (live − unrelated): **{100*primary['difference']:+.1f} pp**, "
        f"95% paired bootstrap CI [{100*primary['ci95'][0]:+.1f}, {100*primary['ci95'][1]:+.1f}] pp. "
        f"Exact two-sided McNemar p = {primary['exact_mcnemar_p']:.6g}; discordant pairs "
        f"{primary['live_only']} live-only / {primary['permuted_only']} unrelated-only.", '']
    if contact:
        lines += [f"Mean forbidden-contact sample difference: {contact['difference']:+.2f} per rollout, "
            f"95% paired CI [{contact['ci95'][0]:+.2f}, {contact['ci95'][1]:+.2f}]. "
            'Physics samples are not independent statistical observations.', '']
    lines += ['| Contact class | Live episodes / samples | Unrelated episodes / samples |', '|---|---:|---:|']
    for c in FORBIDDEN:
        lines.append(f"| {c} | {a['contact_episodes'][c]} / {a['contact_frames'][c]:,} | "
                     f"{b['contact_episodes'][c]} / {b['contact_frames'][c]:,} |")
    lines += ['', 'Categories can overlap; forbidden exposure is the union. Target/receptacle contact is excluded.', '',
        'The policy and its corresponding finetuned encoder were fixed at update 60,000. '
        'At every control step, the intervention supplied a complete 40×128 frame encoded from an unrelated '
        'training episode. Within-frame sensor identity is preserved; temporal and cross-modal correspondence are disrupted.', '',
        report['interpretation_limit'] + '. A null result does not establish non-use or equivalence. '
        'This is not a finetuning-causality or environmental-generalization experiment.', '',
        f"Contact-censored policy failures: {report['contact_censoring_count']}. "
        'Any censored exposure is excluded from full-horizon contact comparisons, and its task/strict outcomes are failures.', '',
        '## Evidence', '', '[Frozen protocol](protocol.json) · [Analysis](analysis.json) · '
        '[Matched records](paired_results.csv) · [Preflight](preflight/report.json) · [Donor plan](donor_plan.json)', '',
        '![Placement and paired contact exposure](results.png)', '']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.arange(2)
    axes[0].bar(x-.18, [a['task']/n, a['strict']/n], .36, label='Live')
    axes[0].bar(x+.18, [b['task']/n, b['strict']/n], .36, label='Unrelated')
    axes[0].set(xticks=x, xticklabels=['Placement', 'Collision-free placement'], ylim=(0, 1), ylabel='Fraction of matched scenarios')
    axes[0].legend()
    for pair in pairs:
        if 'forbidden_frames' in pair['permuted']:
            axes[1].plot([0, 1], [pair['live']['forbidden_frames'], pair['permuted']['forbidden_frames']],
                         color='0.55', alpha=.35, marker='o', markersize=3)
    axes[1].set(xticks=[0, 1], xticklabels=['Live', 'Unrelated'], ylabel='Forbidden-contact physics samples')
    axes[1].set_yscale('symlog', linthresh=10)
    fig.suptitle(f'Seed 3103 · {n} matched scenarios · fixed policy and finetuned encoder')
    fig.savefig(OUT / 'results.png', dpi=180); fig.savefig(OUT / 'results.pdf'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'preflight', 'run', 'analyze'))
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'preflight':
        preflight()
    elif args.command == 'run':
        protocol = freeze_protocol()
        scenes = checked_json(OUT / 'source_manifest.json')['scenes']
        # One full scientific rollout validates infrastructure before concurrency.
        run_jobs(scenes[:1], 'PACT_PERMUTED_128D', 1, protocol['sha256'])
        run_jobs(scenes, 'PACT_PERMUTED_128D', args.workers, protocol['sha256'])
    else:
        analyze()


if __name__ == '__main__':
    main()
