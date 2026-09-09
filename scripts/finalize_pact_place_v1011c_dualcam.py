"""Re-derive final numbers from every retained trajectory/contact/pose artifact."""
from pact_place_v1011c_dualcam import *
from pact_place_v1011c_dualcam_metrics import metrics, check_pair, aggregate
from pact_place_v1011c_metrics import metrics as baseline_metrics
from prepare_pact_place_v1011c_dualcam import check_hash
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone


def authorization_check(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key.startswith('authorizes_'):
                assert item is False, (key,item)
            authorization_check(item)
    elif isinstance(value,list):
        for item in value:
            authorization_check(item)


def main():
    manifest = load_eval_manifest(EVAL/'eval_manifest.json')
    selection = read(EVAL/'checkpoint_selection.json')
    step = selection['selected_global_step']
    final_ids = {r['episode_id'] for r in manifest['rows'] if r['role']=='final'}
    dev_ids = {r['episode_id'] for r in manifest['rows'] if r['role']=='development'}
    assert len(final_ids)==50 and len(dev_ids)==12 and not final_ids&dev_ids
    ledger = [json.loads(x) for x in (EVAL/'final_ledger.jsonl').read_text().splitlines() if x.strip()]
    schedule = read(EVAL/'final_schedule.json')['jobs']
    assert len(ledger)==len(schedule)==100
    expected = {j['schedule']['rollout_id']:j for j in schedule}
    assert len(expected)==100 and {r['rollout_id'] for r in ledger}==set(expected)
    rows, pairs = [],{}
    for receipt in ledger:
        assert receipt['returncode']==0 and receipt['status']=='complete'
        job = expected[receipt['rollout_id']]
        assert receipt == read(Path(job['output'])/'worker_completion.json')
        row = metrics(Path(job['output']),job['schedule'])
        assert row['seed']==3103 and row['training_global_step']==step
        assert row['episode_id'] in final_ids
        assert row['arm'] not in pairs.setdefault(row['episode_id'],{})
        pairs[row['episode_id']][row['arm']] = row
        rows.append(row)
    pairing = [check_pair(p['ACT'],p['PACT']) for p in pairs.values()]
    assert len(pairing)==50
    aggregates = {a:aggregate([r for r in rows if r['arm']==a]) for a in ('ACT','PACT')}
    assert all(a['n']==50 for a in aggregates.values())
    discordance = {key:{'both':sum(p['ACT'][key] and p['PACT'][key] for p in pairs.values()),
        'ACT_only':sum(p['ACT'][key] and not p['PACT'][key] for p in pairs.values()),
        'PACT_only':sum(p['PACT'][key] and not p['ACT'][key] for p in pairs.values()),
        'neither':sum(not p['ACT'][key] and not p['PACT'][key] for p in pairs.values())}
        for key in ('task_success','collision_free_task_success','collision_free')}
    # Same training seed from the old experiment, re-read raw endpoints. Its
    # test stream is different: this is descriptive context, not a paired test.
    previous = [json.loads(x) for x in (original.EVAL/'full_ledger.jsonl').read_text().splitlines() if x.strip()]
    previous = [r for r in previous if r['checkpoint_seed']==3103]
    assert len(previous)==100 and all(r['status']=='complete' and r['returncode']==0 for r in previous)
    old_rows = [baseline_metrics(ROOT/r['directory'],r) for r in previous]
    baseline = {a:{'n':50,**{key:sum(r[key] for r in old_rows if r['arm']==a)
        for key in ('task_success','collision_free_task_success','hazard_contact_episode','clutter_contact_episode')},
        'hazard_frames_total':sum(r['hazard_frames'] for r in old_rows if r['arm']==a)} for a in ('ACT','PACT')}
    source = read(WORK/'source_manifest.json')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check_hash,(source['protected_source_files']|source['implementation_hashes']).items()))
        old_conversion = read(original.WORK/'conversion_manifest_encoded.json')
        list(pool.map(check_hash,[(str((original.DATA/e['act_file']).relative_to(ROOT)),e['act_file_sha256'])
                                 for e in old_conversion['episodes']]))
        conversion = read(WORK/'conversion_manifest_encoded.json')
        list(pool.map(check_hash,[(str((DATA/e['act_file']).relative_to(ROOT)),e['act_file_sha256']) for e in conversion['episodes']]))
    for path in WORK.rglob('*.json'):
        authorization_check(read(path))
    resolvability_path = ROOT/'diagnostics_output/pact_place_v9_w1_resolvability_full/resolvability.json'
    resolvability = read(resolvability_path)
    variants = resolvability['retrodiction']['per_variant']
    widths = [v['roles']['inbound_vessel']['max_w_perp_m'] for v in variants]
    assert len(widths)==8 and sum(v==0.0 for v in widths)==7
    assert len(CANONICAL_SENSOR_NAMES)==40 and {n.split('_')[0] for n in CANONICAL_SENSOR_NAMES}=={f'link{i}' for i in range(1,7)}
    document = {**empty_authorization(),'verified':True,'seed':3103,'selected_global_step':step,
        'training_updates_per_arm':30000,'final_pairs':50,'final_rollouts':100,
        'aggregate':aggregates,'paired_endpoint_counts':discordance,'raw_metrics':rows,'pair_checks':pairing,
        'previous_v1011c_seed3103_raw_recomputed':baseline,
        'previous_comparison_is_paired':False,'final_ledger_sha256':sha256_file(EVAL/'final_ledger.jsonl'),
        'manifest_sha256':manifest['manifest_sha256'],'checkpoint_selection':selection,
        'source_and_parent_conversion_unchanged':True,'all_authorization_flags_false':True,
        'resolvability':{'source_sha256':sha256_file(resolvability_path),'inbound_widths_m':widths,'zero_variants':7,'variants':8},
        'finished_utc':datetime.now(timezone.utc).isoformat(),
        'limitations':['Single training seed, not a three-seed paper result.',
            'Alignment, second RGB view and training budget form a joint intervention; their individual effects are not isolated.',
            '24-case checkpoint comparison uses 12 development pairs, not the final 50 pairs.',
            'Low contacts can reflect failure to approach; interpret alongside target touch and task success.',
            'Proximity input sensitivity is not proof that PACT senses clutter.',
            'No new collection, chunk-25 experiment, shorter-averaging ablation, Wilson interval or McNemar test was run.',
            'Inherited conversion metadata fields are clarified in conversion_metadata_clarification.json; as-run manifests and actual training directory/tree are preserved.']}
    recovery_path = WORK/'recovery_03/repair_completion.json'
    recovery = read(recovery_path) if recovery_path.exists() else None
    if recovery:
        assert recovery['complete'] and recovery['pairing_tolerance_unchanged']
        assert recovery['additional_rollouts'] == 2
        document['pairing_recovery'] = recovery
        document['original_first_attempt_aggregate'] = read(WORK/'recovery_03/original_outcomes.json')['aggregate']
        document['limitations'].append('One pair failed the initial RGB gate on its first attempt. Both arms were repeated once on the same instance; original and repeat results are disclosed, not pooled.')
    freeze(EVAL/'analysis.json',document)
    def rate(arm,key):
        item = aggregates[arm]
        return f"{item['counts'][key]}/50 ({100*item['rates'][key]:.1f}%)"
    text = '\n'.join([
        '# V10.11c aligned dual-camera pilot — seed 3103', '',
        f"Completed 100/100 final rollouts with actual exit 0 and 50/50 initial-state pair checks. Selected snapshot: {step:,} updates for both arms; each model was trained through 30,000 updates.", '',
        '| Endpoint | ACT | PACT |', '|---|---:|---:|',
        *[f'| {label} | {rate("ACT",key)} | {rate("PACT",key)} |' for label,key in [
            ('Task success','task_success'),('Collision-free task success','collision_free_task_success'),
            ('Collision-free episode','collision_free'),('Hazard-bar contact','hazard_contact_episode'),
            ('Any clutter contact','clutter_contact_episode'),('Task object touched','task_object_touched')]],
        f"| Hazard-bar contact frames, total | {aggregates['ACT']['hazard_contact_frames_total']} | {aggregates['PACT']['hazard_contact_frames_total']} |",
        f"| Hazard-bar contact frames, mean/episode | {aggregates['ACT']['hazard_contact_frames_mean']:.2f} | {aggregates['PACT']['hazard_contact_frames_mean']:.2f} |", '',
        'Contact frames are physics-audit samples, not RGB frames or the 900 policy-control steps.', '',
        '| Slot | Arm | Contact episodes | Contact frames | Stable episodes |', '|---|---|---:|---:|---:|',
        *[f"| {slot} | {arm} | {aggregates[arm]['per_object'][slot]['contact_episodes']}/50 | {aggregates[arm]['per_object'][slot]['contact_frames_total']} | {50-aggregates[arm]['per_object'][slot]['stability_event_episodes']}/50 |"
          for slot in ('01','08','09') for arm in ('ACT','PACT')], '',
        'Recipe: existing 99 strict-clean episodes; ledger-derived 75/24 split; observation[t] paired with commanded_action[t+1]; two terminal status-only transitions excluded. Wrist and recorded table RGB are supplied to both policies. Chunk 100, original history-100 temporal ensemble, batch 8, learning rate 1e-5, frozen encoder SHA 6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206, and 127.5 threshold are unchanged. No new data collection was needed.', '',
        'Evaluation uses enclosure_reach.PactPlaceCorridorV1011C33PctTallerPrimitiveSampler on experiment/pact-vs-act-remediation-v2, not the pact_place.py port. Both RGB images permit at most one uint8 intensity level in at most 0.1% of channel values; every other initial field must match exactly. All native thread pools are capped at one; H5-only storage retains raw endpoints.', '',
        'Checkpoint selection used the same 12 development pairs at 20k and 30k, with a preregistered symmetric two-arm rule. Development and smoke instances are excluded from the 50-pair final test. This is a one-seed pilot, not a pooled three-seed result.', '',
        f"Previous V10.11c seed 3103, recomputed from raw artifacts on a different test stream: ACT task {baseline['ACT']['task_success']}/50, collision-free success {baseline['ACT']['collision_free_task_success']}/50; PACT task {baseline['PACT']['task_success']}/50, collision-free success {baseline['PACT']['collision_free_task_success']}/50. This comparison is descriptive, not paired across experiments.", '',
        'Clutter here is effectively invisible to the proximity skin: inbound vessel max_w_perp_m = 0.000 in 7 of 8 variants in diagnostics_output/pact_place_v9_w1_resolvability_full/resolvability.json; the 40 sensors cover link1–link6 only. Any PACT–ACT difference is not evidence that PACT senses clutter. Both arms can use the added RGB camera.', '',
        'All source and parent-conversion hashes were rechecked after evaluation. Every authorization flag remains false. As-run conversion metadata contains inherited parent-directory/token annotations; conversion_metadata_clarification.json records the correct derivative directory and 1,514,760 reused embeddings. Actual training manifests bind the new directory and verified new H5 tree.', '',
        'No Wilson intervals, McNemar tests, chunk-25 runs or gripper-status analysis were performed. Full stage receipts, failures (if any), checkpoint selection, resource samples and machine-readable analysis are retained beside this report.', ''])
    if recovery:
        text += '\n## Disclosed pairing-gate recovery\n\n'
        text += ('The first final stage exited 1 despite 100/100 rollout workers exiting 0: 49/50 pairs passed the strict initial-image check. '
                 'For one pair, all 459 non-RGB datasets and table RGB matched exactly, but three of 658,944 wrist-image channel values differed by two intensity levels, exceeding the one-level maximum. '
                 'The original failed stage, original ledger, and both original attempts are preserved under `../recovery_03/`. '
                 'A bounded repeat of both arms on that same frozen instance passed the unchanged gate. No seeds, models, inference settings or tolerance were changed; no task outcome was used to decide which pair to repeat. '
                 'The main table uses the repaired 50-pair set; the two attempts are not pooled.\n\n')
        text += '| Arm | Original task success | Repeat task success | Original collision-free task success | Repeat collision-free task success |\n|---|---:|---:|---:|---:|\n'
        for arm in ('ACT','PACT'):
            old, new = recovery['original_pair_raw_metrics'][arm], recovery['repeat_pair_raw_metrics'][arm]
            text += f"| {arm} | {int(old['task_success'])} | {int(new['task_success'])} | {int(old['collision_free_task_success'])} | {int(new['collision_free_task_success'])} |\n"
        text += '\nOriginal-first-attempt aggregate task successes: ' + ', '.join(
            f"{arm} {document['original_first_attempt_aggregate'][arm]['counts']['task_success']}/50" for arm in ('ACT','PACT')) + '. Full original/repeat contact and stability endpoints are retained in `analysis.json`.\n'
    report = EVAL/'REPORT.md'
    assert not report.exists()
    with report.open('x') as stream:
        stream.write(text)
    print(json.dumps({'aggregate':aggregates,'previous':baseline,'selected_global_step':step}),flush=True)


if __name__ == '__main__':
    main()
