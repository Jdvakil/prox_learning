"""Regression checks for environment identity, accounting, and object attribution."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from pact_place_v1011c_experiment import EVAL, WORK, SEEDS, load_eval_manifest


def test_split_comes_from_actual_ledger_population():
    source = json.loads((WORK/'source_manifest.json').read_text())
    split = json.loads((WORK/'split_manifest.json').read_text())
    rows = source['rows']
    assert len(rows)==99
    assert split['counts']['train']['total']==75
    assert split['counts']['validation']['total']==24
    assert {e['episode_id'] for e in split['episodes']} == {r['attempt_id'] for r in rows}
    for cell in {r['cell'] for r in rows}:
        members = [e for e in split['episodes'] if e['cell']==cell]
        assert sum(e['split']=='validation' for e in members)==1
        assert sum(e['split']=='train' for e in members)==len(members)-1


def test_manifest_has_fifty_disjoint_instances_per_seed():
    doc = load_eval_manifest(EVAL/'eval_manifest.json')
    assert doc['num_queries']==100 and doc['scientific_rollouts']==300
    assert doc['seed_audit']['disjoint']
    for seed in SEEDS:
        rows = [r for r in doc['rows'] if r['checkpoint_seed']==seed]
        assert len(rows)==50 and len({r['cell'] for r in rows})==24
        assert all(r['task_sampler_class']=='PactPlaceCorridorV1011C33PctTallerPrimitiveSampler' for r in rows)


def test_per_object_physics_frames_are_not_contact_entries(monkeypatch):
    import eval_pact_place_v1011c_row as row
    pair = {'geom1':'robot_0/finger','geom2':'pact_clutter_08/geom',
        'body1':'robot_0/hand','body2':'pact_clutter_08/body',
        'root1':'robot_0/','root2':'pact_clutter_08/body','distance_m':-0.001}
    monkeypatch.setattr(row.audit_module,'place_environment_contact_pairs',lambda env:[dict(pair),dict(pair)])
    audit = row.V1011CContactAudit()
    env = SimpleNamespace(current_data=SimpleNamespace(time=0.0))
    audit.observe(env,0)
    audit.observe(env,0)  # Same physics timestamp must not be double-counted.
    env.current_data.time=0.002
    audit.observe(env,0)
    summary = audit.summary()
    assert summary['sample_count']==2
    assert summary['per_object']['08']=={'contact_entries':4,'contact_frames':2}
    assert summary['per_object']['01']=={'contact_entries':0,'contact_frames':0}
    assert audit.raw_pair_counts==[(0,0,2),(1,0,2)]


def test_sampler_refuses_wrong_environment_and_decoder_is_inherited():
    import eval_pact_place_v1011c_row as row
    with pytest.raises(AssertionError):
        row.sampler_for({'task_sampler_class':'PactPlaceCorridorV1010FourObjectSampler'},object)
    cls=row.sampler_for({'task_sampler_class':'PactPlaceCorridorV1011C33PctTallerPrimitiveSampler'},object)
    assert cls.__module__=='molmo_spaces.tasks.enclosure_reach'
    assert row.V1011CInferencePolicy.model_output_to_action is row.v109.PactPlaceV109InferencePolicy.model_output_to_action


def test_missing_terminal_result_is_not_a_completed_rollout(tmp_path):
    from pact_place_v1011c_metrics import metrics
    with pytest.raises(FileNotFoundError):
        metrics(tmp_path)
    (tmp_path/'result.json').write_text(json.dumps({'status':'infrastructure_failure'}))
    with pytest.raises(AssertionError):
        metrics(tmp_path)


def test_eval_config_binds_collection_scene_and_full_horizon(tmp_path,monkeypatch):
    import eval_pact_place_v1011c_row as evaluator
    row=load_eval_manifest(EVAL/'eval_manifest.json')['rows'][0]
    monkeypatch.setattr(evaluator.v109,'_ACTIVE_SCENE',str(ROOT/row['pact_v1011_scene_relative']))
    monkeypatch.setattr(evaluator.v109,'_ACTIVE_SCENE_SHA256',row['pact_v106_scene_sha256'])
    policy=evaluator.V1011CPolicyConfig(arm='ACT',checkpoint_dir=str(tmp_path),
        checkpoint_path=str(tmp_path/'policy_best.ckpt'),stats_path=str(tmp_path/'dataset_stats.pkl'),
        checkpoint_seed=3103,sensor_names=evaluator.CANONICAL_SENSOR_NAMES)
    config=evaluator.V1011CEvalConfig(output_dir=tmp_path,num_workers=1,policy_config=policy)
    assert config.task_horizon==900 and config.end_on_success is False
    assert config.policy_config.num_queries==100
    assert config.task_sampler_config.task_sampler_class.__name__==row['task_sampler_class']
    assert all(Path(path).name=='pact_place_corridor_v10_7_neg5.xml' for path in config.task_sampler_config.scene_xml_paths)
    assert config.robot_config.action_noise_config.enabled is False


@pytest.mark.parametrize('delta,changed,non_rgb_changed,allowed',[
    (1,1,False,True),(2,1,False,False),(1,4,False,False),(0,0,True,False)])
def test_rgb_tolerance_is_narrow_and_non_rgb_stays_exact(tmp_path,delta,changed,non_rgb_changed,allowed):
    import h5py
    from pact_place_v1011c_pairing import check_pair
    rows=[]
    for arm in ('ACT','PACT'):
        directory=tmp_path/arm;directory.mkdir()
        image=np.zeros((32,32,3),dtype=np.uint8)
        if arm=='PACT':image.ravel()[:changed]=delta
        with h5py.File(directory/'initial_observation.h5','w') as h:
            h.create_dataset('observation/wrist_camera',data=image)
            h.create_dataset('observation/qpos',data=np.array([1.0 if arm=='PACT' and non_rgb_changed else 0.0]))
        rows.append({'episode_id':'same','task_seed':1,'row_sha256':'same','directory':str(directory)})
    if allowed:
        assert check_pair(*rows)['rgb']['changed_channel_values']==changed
    else:
        with pytest.raises(AssertionError):check_pair(*rows)


def test_resume_preserves_real_exit_receipts_and_never_repeats_completed_jobs(tmp_path,monkeypatch):
    import run_pact_place_v1011c_eval as runner
    monkeypatch.setattr(runner,'metrics',lambda *args:{'validated':True})
    job={'schedule':{'rollout_id':'frozen-job','arm':'ACT','checkpoint_seed':3103},'output':str(tmp_path)}
    row={**job['schedule'],'directory':str(tmp_path),'status':'complete','returncode':0,
         'adopted_after_scheduler_drain':True,'exit_code_evidence':{
             'source':'/proc/PID/stat field 52 while zombie','linux_wait_status':0}}
    ledger=tmp_path/'ledger.jsonl'
    ledger.write_text('')
    records,added=runner.reconcile_resume([job],ledger,[row])
    assert records==added==[row]
    ledger.write_text(json.dumps(row)+'\n')
    records,added=runner.reconcile_resume([job],ledger,[row])
    assert records==[row] and added==[]
    for invalid in ({**row,'returncode':1},{**row,'reused_validated_result':True},
                    {**row,'checkpoint_seed':3104},{**row,'status':'failed'}):
        with pytest.raises(AssertionError):runner.reconcile_resume([job],ledger,[invalid])


def test_result_file_alone_does_not_establish_successful_process_exit(tmp_path):
    import run_pact_place_v1011c_eval as runner
    (tmp_path/'result.json').write_text('{}')
    # The path only needs to belong to the worktree for the pre-existing guard.
    job={'schedule':{},'output':str(WORK/'test_unstarted_guard')}
    from unittest.mock import patch
    with patch.object(runner.Path,'relative_to',return_value=Path('test')), \
         patch.object(runner.Path,'mkdir'), patch.object(runner.Path,'exists',return_value=True):
        with pytest.raises(AssertionError,match='actual exit-code receipt'):
            runner.run_one(job)
