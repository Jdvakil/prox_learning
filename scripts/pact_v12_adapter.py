"""v12 collection variant: one retained bottle and standing kitchen objects.

The raw scene_params retain the inherited V1010 marker and pre-overlay layout.
The dataset manifest and collection helpers define the actual deployed scene.
"""
from __future__ import annotations
from copy import deepcopy
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SLOTS = ('01', '03', '04', '06')


def row_payload(profile, template):
    """Recover the nominal eight-slot row, avoiding double application of jitter."""
    params = template['scene_params']
    if template['pose_id'] != 'center':
        raise ValueError('The v12 dataset contains center-pose demonstrations only')
    clutter = json.loads((ROOT / profile['clutter_config']).read_text())
    key = f"{template['family_id']}|{template['intrusion_side']}"
    nominal = deepcopy(clutter['layouts'][key])
    observed = params['pact_clutter_layout']
    x_jitter = observed['applied_clutter_x_jitter_m']
    y_jitter = observed['applied_clutter_y_jitter_m']
    by_slot = {o['palette_slot']: o for o in nominal['objects']}
    recorded = {o['palette_slot']: o for o in observed['objects']}
    if set(recorded) != set(ACTIVE_SLOTS):
        raise ValueError('v12 inherited layout must have exactly the four recorded slots')
    for slot in ACTIVE_SLOTS:
        source, actual = by_slot[slot], recorded[slot]
        expected = [source['center_m'][0] + x_jitter.get(slot, 0.),
                    source['center_m'][1] + y_jitter.get(slot, 0.), source['center_m'][2]]
        if source['uid'] != actual['uid'] or any(
                not math.isclose(a, b, abs_tol=1e-9, rel_tol=0.) for a, b in zip(expected, actual['center_m'])):
            raise ValueError(f'v12 nominal layout does not reproduce recorded slot {slot}')
        # Preserve collection precision; the checked-in geometry has a few
        # sub-femtometer serialization differences in support height.
        source['center_m'][2] = actual['center_m'][2]
        source['half_m'] = deepcopy(actual['half_m'])
        source['quat_wxyz'] = deepcopy(actual['quat_wxyz'])
    # Use the palette stored in the demonstration, including physical dimensions.
    row = {'pact_clutter_palette': deepcopy(params['pact_clutter_palette']),
           'pact_clutter_layout': nominal, 'clutter_x_jitter_m': deepcopy(x_jitter),
           'clutter_y_jitter_m': deepcopy(y_jitter), 'target_x_jitter_m': 0., 'target_y_jitter_m': 0.,
           'overlay': profile['overlay'], 'dataset_environment_version': profile['dataset_environment_version']}
    for name in ('pact_v106_scene_sha256', 'pact_v106_x_m', 'pact_v106_r_neg_m', 'pact_v106_r_pos_m'):
        row[name] = params[name]
    return row


def install_sampler_overlay(sampler, runtime):
    """Attach the collection's kitchen assets before model compilation."""
    # The existing collection helper honors this path before importing MolmoSpaces.
    os.environ['MOLMOSPACES_PACT_V1010'] = str(runtime)
    import render_pact_place_v12_clutter as overlay
    overlay._install_preview_contact_classes()
    names = []
    original_add = sampler.add_auxiliary_objects

    def add_auxiliary_objects(spec):
        original_add(spec)
        names.extend(overlay._attach_standing_kitchen(spec))

    sampler.add_auxiliary_objects = add_auxiliary_objects
    return overlay, names


def apply_overlay(task, overlay, names):
    """Apply collection geometry only; expert motion corrections are not a policy input."""
    import mujoco
    model, data = task.env.current_model, task.env.current_data
    overlay._hide_primitive_colliders(model)
    overlay._apply_preview_household(model, data)
    overlay._refresh_clutter_settle(task, model, data)
    placed = overlay._place_standing_kitchen(model, data, names)
    mujoco.mj_forward(model, data)
    behind = tuple(item['uid'] for item in overlay.STANDING_KITCHEN if item.get('behind_grasp'))
    hits = [name for name in overlay.extras_overlap_motion_lane(model, data, placed)
            if not any(uid in name for uid in behind)]
    if hits:
        raise ValueError(f'v12 kitchen extras overlap the motion lane: {hits}')
    task.scene_params['pact_evaluation_variant'] = 'pact_place_corridor_v10_11_preview_onebottle'
    return {'variant': task.scene_params['pact_evaluation_variant'],
            'parked_household': list(overlay.PARK_HOUSEHOLD),
            'retained_bottle': overlay.KEEP_BOTTLE,
            'placed_kitchen_bodies': placed,
            'kitchen_positions_m': {name: data.xpos[model.body(name).id].tolist() for name in placed}}
