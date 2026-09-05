"""The v12 hub label includes post-sampler geometry changes."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'submodules/act')]
from pact_v12_adapter import row_payload, apply_overlay, install_sampler_overlay
from pact_workflow import prepare_contract
from pact import convert_command


def fixture():
    profile = json.loads((ROOT / 'configs/pact_datasets.json').read_text())['datasets']['v12']
    clutter = json.loads((ROOT / profile['clutter_config']).read_text())
    family = 'F1_inner_panel_stagger'
    layout = deepcopy(clutter['layouts'][f'{family}|right'])
    layout['objects'] = [o for o in layout['objects'] if o['palette_slot'] in ('01','03','04','06')]
    jx, jy = {'01': -.005, '06': .003}, {'01': .003, '06': -.006}
    for o in layout['objects']:
        o['center_m'][0] += jx.get(o['palette_slot'], 0.)
        o['center_m'][1] += jy.get(o['palette_slot'], 0.)
    layout.update(applied_clutter_x_jitter_m=jx, applied_clutter_y_jitter_m=jy)
    return profile, {'family_id': family, 'intrusion_side': 'right', 'pose_id': 'center',
                     'scene_params': {'pact_clutter_palette': clutter['palette'], 'pact_clutter_layout': layout,
                                      'pact_v106_scene_sha256': profile['scene_sha256'],
                                      'pact_v106_x_m': .8, 'pact_v106_r_neg_m': .33, 'pact_v106_r_pos_m': .3}}


def test_nominal_row_retains_parked_slots_and_applies_jitter_only_once():
    profile, template = fixture()
    before = deepcopy(template)
    row = row_payload(profile, template)
    assert len(row['pact_clutter_layout']['objects']) == 8
    source = next(o for o in row['pact_clutter_layout']['objects'] if o['palette_slot'] == '01')
    observed = next(o for o in template['scene_params']['pact_clutter_layout']['objects'] if o['palette_slot'] == '01')
    assert source['center_m'][0] + row['clutter_x_jitter_m']['01'] == observed['center_m'][0]
    assert template == before
    assert row['overlay'] == 'v12_preview_household'
    assert row['pact_v106_scene_sha256'] == profile['scene_sha256']


def test_mismatched_geometry_or_unseen_pose_cannot_silently_use_v12():
    profile, template = fixture()
    template['pose_id'] = 'neg5'
    with pytest.raises(ValueError, match='center-pose'):
        row_payload(profile, template)
    template['pose_id'] = 'center'
    template['scene_params']['pact_clutter_layout']['objects'][0]['center_m'][0] += .01
    with pytest.raises(ValueError, match='does not reproduce'):
        row_payload(profile, template)


def test_sampler_overlay_adds_assets_before_compilation(monkeypatch):
    calls = []
    overlay = SimpleNamespace(_install_preview_contact_classes=lambda: calls.append('taxonomy'),
                              _attach_standing_kitchen=lambda spec: calls.append(('extras', spec)) or ['extra'])
    monkeypatch.setitem(sys.modules, 'render_pact_place_v12_clutter', overlay)
    monkeypatch.setenv('MOLMOSPACES_PACT_V1010', 'old')
    sampler = SimpleNamespace(add_auxiliary_objects=lambda spec: calls.append(('original', spec)))
    module, names = install_sampler_overlay(sampler, Path('/tmp/pinned_runtime'))
    sampler.add_auxiliary_objects('spec')
    assert calls == ['taxonomy', ('original', 'spec'), ('extras', 'spec')]
    assert module is overlay and names == ['extra']


def test_overlay_parks_bottle_refreshes_settle_and_reports_lane_error(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, 'mujoco', SimpleNamespace(mj_forward=lambda m,d: calls.append('forward')))
    overlay = SimpleNamespace(
        _hide_primitive_colliders=lambda m: calls.append('hide'),
        _apply_preview_household=lambda m,d: calls.append('household'),
        _refresh_clutter_settle=lambda t,m,d: calls.append('settle'),
        _place_standing_kitchen=lambda m,d,n: calls.append('place') or ['bad_extra'],
        STANDING_KITCHEN=(), extras_overlap_motion_lane=lambda m,d,n: ['bad_extra'])
    task = SimpleNamespace(env=SimpleNamespace(current_model=object(),current_data=object()),scene_params={})
    with pytest.raises(ValueError, match='motion lane'):
        apply_overlay(task, overlay, ['bad_extra'])
    assert calls == ['hide','household','settle','place','forward']


def test_converter_command_targets_v12_without_reusing_v1011d():
    profile, _ = fixture()
    command = convert_command('v12', profile)
    assert command[1:3] == ['-m','scripts.convert_pact_place_to_act']
    assert command[command.index('--dst')+1].endswith('/data/v12')
    assert '--with_proximity' in command
    assert command[command.index('--prox_pool')+1] == 'min'


def test_v12_suite_has_eight_dev_and_48_test_rows_with_no_unseen_poses(monkeypatch):
    import pact_workflow
    profile, template = fixture()
    cells = [f'F{i}|{side}|center' for i in range(4) for side in ('left','right')]
    episodes = [{'id':i,'cell':cells[i % 8],'selected_seed':i,'requested_seed':i} for i in range(40)]
    templates = {c:dict(template, family_id=c.split('|')[0], intrusion_side=c.split('|')[1]) for c in cells}
    monkeypatch.setattr(pact_workflow, 'inventory', lambda p: ({'image_h':240,'image_w':320,'prox_pool':'min'},episodes,templates))
    def eval_row(p,t,role,index,forbidden):
        return {'role':role,'index':index,'pose_id':t['pose_id'],'family_id':t['family_id'],'side':t['intrusion_side']}
    monkeypatch.setattr(pact_workflow, 'evaluation_row', eval_row)
    contract = prepare_contract('v12', profile)
    assert len(contract['evaluation']['dev']) == 8
    assert len(contract['evaluation']['test']) == 48
    assert all(r['pose_id']=='center' for r in contract['evaluation']['test'])
