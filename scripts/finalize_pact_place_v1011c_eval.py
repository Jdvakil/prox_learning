"""Descriptive, per-seed results reconstructed from raw V10.11c artifacts."""
import argparse
import json
import os
import signal
import time
from pact_place_v1011c_experiment import *
from pact_place_v1011c_metrics import metrics
from pact_place_v1011c_pairing import check_pair


def cleanup_orphan_workers():
    cleaned = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            status = (proc/'stat').read_text().split()
            command = (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if status[3]!='1' or 'multiprocessing.spawn' not in command:
                continue
            cwd = Path(os.readlink(proc/'cwd')).resolve()
            if not cwd.is_relative_to(ROOT):
                continue
            os.kill(int(proc.name),signal.SIGTERM)
            cleaned.append({'pid':int(proc.name),'cwd':str(cwd),'command':command,'signal':'SIGTERM'})
        except (OSError,UnicodeError,ProcessLookupError):
            continue
    return cleaned


def report_failure():
    failures = []
    for file in sorted((WORK/'stages').glob('*.json')):
        row = read(file)
        if row['returncode'] or row.get('failed'):
            failures.append(row)
    if not failures:
        return
    lines = ['\n## V10.11c pipeline failure\n',
             'The following stage failed. Dependent stages have not been reported as completed.\n']
    for failure in failures:
        lines += [f"Stage `{failure['stage']}` exited `{failure['returncode']}`. Log: `{failure['log']}`.\n",
                  'Actual output (tail):\n', '```text\n'+failure['output_tail'][-6000:]+'\n```\n']
    lines += ['No evaluation conclusion is available from unfinished stages.\n']
    text = '\n'.join(lines)
    failure_id=digest(failures)[:16]
    freeze(WORK/f'pipeline_failure_{failure_id}.json',{**empty_authorization(),'failures':failures})
    (WORK/f'pipeline_failure_{failure_id}.md').write_text(text)
    path=ROOT/'EVAL.md'
    old=path.read_text()
    if text not in old:
        path.write_text(old+text)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--failure-only',action='store_true')
    args=p.parse_args()
    if args.failure_only:
        report_failure()
        return
    manifest=load_eval_manifest(EVAL/'eval_manifest.json')
    run=read(EVAL/'full_run.json')
    assert run['infrastructure_healthy'] and run['rollouts_complete']==run['rollouts_expected']==300
    ledger=[json.loads(x) for x in (EVAL/'full_ledger.jsonl').read_text().splitlines()]
    assert len(ledger)==300 and len({r['rollout_id'] for r in ledger})==300
    assert all(r['status']=='complete' and r['returncode']==0 for r in ledger)
    entries=[metrics(ROOT/r['directory'],r) for r in ledger]
    expected={(r['checkpoint_seed'],r['episode_id'],a) for r in manifest['rows'] for a in ('ACT','PACT')}
    assert {(r['seed'],r['episode_id'],r['arm']) for r in entries}==expected
    by_id={}
    for row in entries:
        by_id.setdefault(row['episode_id'],{})[row['arm']]=row
    pair_checks={key:check_pair(pair['ACT'],pair['PACT']) for key,pair in by_id.items()}
    assert len(pair_checks)==150
    source=read(WORK/'source_manifest.json')
    for path,expected_hash in source['protected_source_files'].items():
        assert sha256_file(ROOT/path)==expected_hash, f'source drift: {path}'
    models=read(WORK/'training_verification.json')['models']
    resolution=read(ROOT/'diagnostics_output/pact_place_v9_w1_resolvability_full/resolvability.json')
    widths=[v['roles']['inbound_vessel']['max_w_perp_m'] for v in resolution['retrodiction']['per_variant']]
    assert len(widths)==8 and sum(w==0 for w in widths)==7
    assert len(CANONICAL_SENSOR_NAMES)==40
    assert set(n.split('_')[0] for n in CANONICAL_SENSOR_NAMES)=={'link1','link2','link3','link4','link5','link6'}
    groups={}
    for seed in (*SEEDS,'pooled'):
        groups[str(seed)]={}
        for arm in ('ACT','PACT'):
            rows=[r for r in entries if r['arm']==arm and (seed=='pooled' or r['seed']==seed)]
            n=len(rows)
            assert n==(150 if seed=='pooled' else 50)
            groups[str(seed)][arm]={'n':n,**{key:sum(r[key] for r in rows) for key in (
                'task_success','collision_free_task_success','hazard_contact_episode','hazard_frames','hazard_entries',
                'clutter_contact_episode','clutter_stability_events')},
                'per_object':{slot:{key:sum(r['per_object'][slot][key] for r in rows)
                    for key in ('contact_episode','contact_frames','contact_entries','stability_event','stable')}
                    for slot in ('01','08','09')}}
    cleaned=cleanup_orphan_workers()
    doc={**empty_authorization(),'verified':True,'training_chunk':100,'evaluation_chunk':100,
        'training_seeds':list(SEEDS),'groups':groups,'rows':entries,'pair_checks':pair_checks,
        'full_ledger_sha256':sha256_file(EVAL/'full_ledger.jsonl'),
        'source_preservation_files_verified':len(source['protected_source_files']),
        'skin_inbound_vessel_widths_m':widths,'orphan_workers_cleaned':cleaned,
        'statistical_tests_omitted_at_owner_request':True}
    freeze(EVAL/'analysis.json',doc)
    lines=['# V10.11c ACT/PACT training and paired evaluation\n',
        'Completed six chunk-100 models and **300/300 raw-verified scientific rollouts**: '
        'ACT and PACT, seeds 3103/3104/3105, 50 paired instances per seed. '
        'All 150 pairs have identical non-RGB initial observations and selected environment seeds, with wrist RGB checked against the disclosed tolerance below. '
        'The smoke completed 8/8 rollouts on four separate instances, initially failed strict RGB equality, and was reconciled under that rule without rerunning any rollout; eight additional frozen smoke rows were reserved but not selected.\n',
        'The owner narrowed scope on 2026-09-05 to chunk 100 only and omitted chunk 25, '
        'gripper-status analysis, Wilson intervals and McNemar tests. Results below are descriptive. '
        'They do not establish statistical superiority.\n',
        '## Corpus and training\n',
        'The read-only collection ledger contains **482 attempts, 99 accepted strict-clean episodes**. '
        'All 24 cells have 4–5 episodes. Row directories resolve as `attempt_id[:16]`; no closeout was required. '
        'The split is **75 train / 24 validation**, taking the lowest SHA256(split seed:attempt_id) per cell for validation. '
        'Split seed: 2026090401. Converted T=176–546, sum 37,871; '
        '1,514,840 windows were encoded, with independent source-preservation checks.\n',
        'The reused embedding verifier retains V10.9 schema labels and a V10.8 '
        '`note_on_window_count` paragraph. That legacy paragraph does not describe this corpus; '
        'the numerical episode/timestep/window fields and source reconstruction are authoritative.\n',
        'Both arms used 2,000 epochs, batch 8, learning rate 1e-5, KL 10, chunk 100, hidden 512, '
        'feed-forward 3,200, 7 encoder/7 decoder layers, 8 heads, wrist ResNet-18, and state/action dimensions 9/8. '
        'Training episode horizon was 635. The command comparison allowed only the checkpoint directory and five PACT flags. '
        'The frozen encoder hash is `'+ENCODER_SHA256+'`. '
        'All six models passed strict checkpoint reload, offline inference and split/dataset provenance checks; '
        'PACT proximity consumption was checked using real versus zeroed embeddings.\n',
        '| Seed | Arm | Epochs | Best epoch | Best validation loss | Training minutes |\n|---|---|---:|---:|---:|---:|']
    for seed in SEEDS:
        for arm in ('act','pact'):
            m=models[f'{arm}_seed{seed}'];v=m['verification']
            lines.append(f"| {seed} | {arm.upper()} | {v['epochs_recorded']} | {v['best_epoch']} | {v['best_val_loss']:.6f} | {m['elapsed_minutes']:.1f} |")
    lines += ['\n## Paired outcomes\n',
        'The overall comparison includes **all three independently trained seeds**: 150 rollouts per arm. '
        'For every episode-level rate, the pooled numerator divided by 150 is exactly the arithmetic mean of '
        'the three seed-level rates, because each seed contributes 50 instances. Frame and contact-entry '
        'columns are summed counts across the indicated seed(s), not percentages or per-rollout averages. '
        'Each arm is paired on the same instances within each seed; the three seed blocks use disjoint held-out instances. '
        'Thus variation between seed rows reflects both training randomness and held-out instance variation.\n',
        'Collision-free success means final task success with zero hazard-bar, other-environment, clutter or mounted-fixture '
        'contact entries. Task success was independently read from the final trajectory `success` field. '
        'Contact episodes have at least one physics sample with that contact. A frame below is an audited physics sample '
        '(2 ms plus episode boundaries), not a contact-pair entry or a rendered video frame.\n',
        '| Seed | Arm | Collision-free success | Task success | Hazard-contact episodes | Hazard frames | Hazard entries | Clutter-contact episodes |\n'
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for seed in (*SEEDS,'pooled'):
        for arm in ('ACT','PACT'):
            g=groups[str(seed)][arm];n=g['n']
            rate=lambda key:f"{g[key]}/{n} ({100*g[key]/n:.1f}%)"
            lines.append(f"| {seed} | {arm} | {rate('collision_free_task_success')} | {rate('task_success')} | {rate('hazard_contact_episode')} | {g['hazard_frames']} | {g['hazard_entries']} | {rate('clutter_contact_episode')} |")
    lines += ['\n## Per-object contact and stability\n',
        'Slot 01 is the tall route cylinder; 08 the tall near-target cylinder; 09 the tall near-target box. '
        'An object is stable only if it never exceeds 2 cm translation or 25° rotation from its settled baseline at any retained '
        'control sample. Object contact attribution was reconstructed from retained geom/body pair identities and counts; '
        'stability was reconstructed from retained poses and rotations.\n',
        '| Seed | Slot | Arm | Contact episodes | Contact frames | Contact entries | Stable episodes |\n|---|---|---|---:|---:|---:|---:|']
    for seed in (*SEEDS,'pooled'):
        for slot in ('01','08','09'):
            for arm in ('ACT','PACT'):
                g=groups[str(seed)][arm];o=g['per_object'][slot];n=g['n']
                lines.append(f"| {seed} | {slot} | {arm} | {o['contact_episode']}/{n} ({100*o['contact_episode']/n:.1f}%) | {o['contact_frames']} | {o['contact_entries']} | {o['stable']}/{n} ({100*o['stable']/n:.1f}%) |")
    lines += ['\n## Interpretation and provenance\n',
        'The original smoke stage exited 1 after 8/8 successful rollouts because three wrist RGB pairs failed exact byte equality: '
        '`AssertionError: initial observation mismatch: observation/wrist_camera`. '
        'Those pairs differed in 15, 24 and 48 of 658,944 uint8 channel values, each by exactly one intensity level; '
        'all 449 non-RGB observation datasets matched exactly. Before scientific evaluation, the pairing rule was amended to allow '
        'at most 1/255 intensity difference in at most 0.1% of RGB channel values while requiring exact equality for every other field. '
        'The cause of the sparse RGB differences was not conclusively established. Original failures, raw images and '
        '`smoke_reconciliation.json` are retained. No smoke performance outcome influenced this change.\n',
        'The outer coordinator was lost in a Codex crash during training; the six-model trainer survived. A detached coordinator '
        'reconnected without restarting a model. Its completion record explicitly marks the lost coordinator exit code as unavailable '
        'and relies on all six actual model exit codes and independent verification.\n',
        'Clutter here is effectively invisible to the proximity skin: inbound vessel `max_w_perp_m = 0.000` in **7/8** variants '
        'of `diagnostics_output/pact_place_v9_w1_resolvability_full/resolvability.json`; the 40 sensors cover **link1–link6 only**. '
        'Any PACT–ACT difference is **not evidence that PACT senses the clutter**. This historical resolvability check is not a new '
        'visibility measurement on the held-out evaluation instances.\n',
        'Evaluation used `molmo_spaces.tasks.enclosure_reach.PactPlaceCorridorV1011C33PctTallerPrimitiveSampler` from '
        '`experiment/pact-vs-act-remediation-v2`, with its implementation hash checked against the source freeze. '
        'The original chunk-100 temporal ensemble, 127.5 gripper threshold, 900-step horizon, disabled action noise and '
        '`end_on_success=false` were preserved. All five native thread pools were capped at 1. '
        'Initial and retry evaluation seeds were checked against 28,222 historical/collection-stream seeds and against each other.\n',
        f"Scientific evaluation took {run['elapsed_hours']:.2f} hours with worker-count history {run.get('worker_history',[run['workers']])}. Full H5 trajectories, actions, "
        'initial observations and raw telemetry are retained under `diagnostics_output/pact_place_v1011c_eval/`.\n',
        'Periodic training snapshots and optimizer resume bundles were omitted for the initial disk constraint. '
        'Verified best checkpoints are retained; each redundant final checkpoint was pruned after verification. '
        f"Source-preservation checks rehashed {len(source['protected_source_files'])} files. "
        'Every authorization flag remains false.\n',
        f"Orphan `multiprocessing.spawn` workers belonging to this worktree cleaned: {len(cleaned)}.\n",
        'Machine-readable report: `diagnostics_output/pact_place_v1011c_eval/analysis.json`. '
        'Completion was reconciled against `full_ledger.jsonl` and every raw result, not a progress summary.\n']
    if run.get('scheduler_resize_receipt'):
        lines += ['The owner requested increased concurrency after observing resource headroom. The original scheduler '
            'was paused while all active rollouts finished normally. Their actual Linux exit codes and raw artifacts '
            'were retained in `scheduler_resize_01/`; only the drained scheduler was terminated (stage '
            '`11_eval_full` actual exit -9). The resumed stage retained the original frozen schedule and completed '
            'the remaining instances without repeating or discarding a scientific rollout. The interrupted stage '
            'is not reported as successful.\n']
    text='\n'.join(lines)
    (EVAL/'report.md').write_text(text)
    path=ROOT/'EVAL.md'; prior=path.read_text()
    marker='# V10.11c ACT/PACT experiment — in progress'
    assert marker in prior
    path.write_text(prior.split(marker,1)[0]+text)
    print('EVAL.md written from 300 verified rollouts and 150 pairs passing the disclosed observation checks.',flush=True)


if __name__ == '__main__':
    main()
