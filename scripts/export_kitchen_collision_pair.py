#!/usr/bin/env python3
"""Contact-checked kitchen figure illustrations from the original MuJoCo scene.

The arm poses are constructed with IK, not recorded policy outcomes. Objects,
camera and lighting are restored from the same source as the kitchen figure.
"""
import argparse
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import mujoco
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from sensor_grid_image import bordered_sensor_grid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'images/whole_body/kitchen_collision_pair'
ROW = Path('/mnt/laptop/data/pact_pick_n_place_v2/data/v12/rows/000_2f77fef1863bdeb3')
SCENE = ROOT / 'custom_scenes/pact_place_corridor_v12.xml'
SENSORS = ['link6_sensor_0', 'link6_sensor_3', 'link5_back_sensor_4']
CROP = (380, 0, 2115, 1408)


def scene(snapshot):
    if snapshot is None:
        from export_table_occlusion_pair import main
        m, d, table, k, _, _, _ = main(return_scene=True, row=ROW, scene_path=SCENE, kitchen=True)
        m.vis.global_.fovy = np.rad2deg(2*np.arctan(k[1, 2]/k[1, 1]))
        return m, d, table
    m = mujoco.MjModel.from_binary_path(str(snapshot/'model.mjb'))
    d = mujoco.MjData(m)
    with np.load(snapshot/'state.npz') as state:
        for key in ['qpos', 'qvel', 'mocap_pos', 'mocap_quat']:
            getattr(d, key)[:] = state[key]
        table = state['table'].copy()
    mujoco.mj_forward(m, d)
    return m, d, table


def contacts(m, d):
    result = []
    for contact in d.contact:
        bodies = [m.body(int(m.geom_bodyid[g])).name for g in contact.geom]
        robot = [name.startswith('robot_0/') for name in bodies]
        if any(robot) and not all(robot):
            result.append(dict(bodies=bodies, geom_ids=contact.geom.tolist(),
                               signed_distance_m=float(contact.dist), position_world=contact.pos.tolist()))
    return result


def main(snapshot=None):
    from export_sensor_modalities import depth_rgb, NEAR, FAR
    OUT.mkdir(parents=True, exist_ok=True)
    m, d, table = scene(snapshot)
    joints = [m.joint(f'robot_0/fr3_joint{i+1}') for i in range(7)]
    adr = [j.qposadr[0] for j in joints]
    provenance = json.loads((ROOT/'images/whole_body/kitchen_items/manifest.json').read_text())['provenance']
    q0 = np.array(provenance['adjusted_simulation_pose']['adjusted_arm_qpos'])
    d.qpos[adr] = q0
    mujoco.mj_forward(m, d)
    cid = m.camera('robot_0/link6_sensor_4').id
    p0 = d.cam_xpos[cid].copy()
    rotation = d.cam_xmat[cid].reshape(3, 3).copy()
    nonrobot = [i for i in range(m.nbody) if not m.body(i).name.startswith('robot_0/')]
    scene_positions = d.xpos[nonrobot].copy()
    scene_rotations = d.xmat[nonrobot].copy()
    robot_geoms = [i for i in range(m.ngeom)
                   if m.body(int(m.geom_bodyid[i])).name.startswith('robot_0/')
                   and m.body(int(m.geom_bodyid[i])).name != 'robot_0/base'
                   and int(m.geom_contype[i] | m.geom_conaffinity[i])]
    hazard_geoms = [i for i in range(m.ngeom)
                    if m.body(int(m.geom_bodyid[i])).name == 'pact_intrusion_right'
                    and int(m.geom_contype[i] | m.geom_conaffinity[i])]

    def place(delta):
        def residual(q):
            d.qpos[adr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
            dr = Rotation.from_matrix(d.cam_xmat[cid].reshape(3, 3) @ rotation.T).as_rotvec()
            return np.r_[10*(d.cam_xpos[cid]-p0-delta), dr, .002*(q-q0)]
        fit = least_squares(residual, q0, max_nfev=100,
                            bounds=([j.range[0] for j in joints], [j.range[1] for j in joints]))
        residual(fit.x)
        mujoco.mj_forward(m, d)
        assert fit.success and np.linalg.norm(d.cam_xpos[cid]-p0-delta) < 1e-5
        assert np.array_equal(d.xpos[nonrobot], scene_positions)
        assert np.array_equal(d.xmat[nonrobot], scene_rotations)
        return fit.x

    table_opt = mujoco.MjvOption(); table_opt.geomgroup[3:] = 0; table_opt.sitegroup[:] = 0
    rgb_opt = mujoco.MjvOption(); rgb_opt.geomgroup[2] = 0; rgb_opt.geomgroup[4] = 0; rgb_opt.sitegroup[:] = 0
    depth_opt = mujoco.MjvOption(); depth_opt.geomgroup[2] = 0; depth_opt.geomgroup[4] = 1; depth_opt.sitegroup[:] = 0
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    records = []
    with mujoco.Renderer(m, 1408, 2496) as external, mujoco.Renderer(m, 1024, 1024) as dense, mujoco.Renderer(m, 8, 8) as native:
        for label, delta in [('collision', [.047, -.07, .12]), ('avoidance', [.047, .14, .12])]:
            folder = OUT/label; folder.mkdir(exist_ok=True)
            q = place(np.array(delta))
            actual_contacts = contacts(m, d)
            distances = [(float(mujoco.mj_geomDistance(m, d, a, b, .5, None)), a, b)
                         for a in robot_geoms for b in hazard_geoms]
            distance, robot_gid, hazard_gid = min(distances)
            if label == 'collision':
                assert len(actual_contacts) == 1
                assert actual_contacts[0]['bodies'] == ['pact_intrusion_right', 'robot_0/fr3_link7']
                assert -.004 < distance < -.001
            else:
                assert not actual_contacts and distance > .08
            np.savez_compressed(folder/'state.npz', qpos=d.qpos, qvel=d.qvel,
                                mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat)
            external.update_scene(d, camera=cam, scene_option=table_opt)
            for c in external.scene.camera:
                c.pos = table[:3, 3]; c.forward = table[:3, 2]; c.up = -table[:3, 1]
            full = Image.fromarray(external.render().copy())
            full.save(folder/'scene_full.png')
            full.crop(CROP).save(OUT/f'{label}.png')
            # Same crop in both images, useful for placing a contact inset in Canva.
            full.crop((920, 400, 1610, 910)).save(folder/'detail.png')
            sensor_records = []
            all_names = sorted(m.camera(i).name for i in range(m.ncam) if '_sensor_' in m.camera(i).name)
            all_depths = []
            for camera_name in all_names:
                name = camera_name.split('/')[-1]
                native.update_scene(d, camera=camera_name, scene_option=depth_opt)
                native.enable_depth_rendering(); depth = native.render().copy(); native.disable_depth_rendering()
                assert depth.shape == (8, 8) and np.all(np.isfinite(depth)) and np.all(depth > 0)
                all_depths.append(depth)
                if name not in SENSORS + ['link6_sensor_4', 'link2_sensor_3']:
                    continue
                sensor_dir = folder/name; sensor_dir.mkdir(exist_ok=True)
                np.save(sensor_dir/'readout_8x8_metres.npy', depth)
                bordered_sensor_grid(Image.fromarray(depth_rgb(depth))).save(sensor_dir/'readout_8x8.png')
                native.enable_segmentation_rendering(); seg = native.render().copy(); native.disable_segmentation_rendering()
                hits = []
                for (gid, kind), z in zip(seg.reshape(-1, 2), depth.ravel()):
                    if int(gid) in hazard_geoms and kind == int(mujoco.mjtObj.mjOBJ_GEOM):
                        hits.append(float(z))
                dense.update_scene(d, camera=camera_name, scene_option=rgb_opt)
                Image.fromarray(dense.render().copy()).save(sensor_dir/'rgb.png')
                dense.update_scene(d, camera=camera_name, scene_option=depth_opt)
                dense.enable_depth_rendering(); dd = dense.render().copy(); dense.disable_depth_rendering()
                np.save(sensor_dir/'dense_depth_metres.npy', dd)
                Image.fromarray(depth_rgb(dd)).save(sensor_dir/'depth.png')
                sensor_records.append(dict(sensor=name, panel_hit_count=len(hits),
                                           closest_panel_axial_depth_m=min(hits) if hits else None))
            assert len(all_depths) == 40
            np.savez_compressed(folder/'all_40_sensors.npz', sensor_names=np.array(all_names), depth_metres=np.stack(all_depths))
            figure = Image.new('RGB', (2496, 1120), 'white')
            figure.paste(full.crop(CROP).resize((1380, 1120), Image.Resampling.LANCZOS), (0, 0))
            for row, sensor in enumerate(SENSORS):
                for col, filename in enumerate(['rgb.png', 'depth.png', 'readout_8x8.png']):
                    with Image.open(folder/sensor/filename) as panel:
                        mode = Image.Resampling.NEAREST if col == 2 else Image.Resampling.LANCZOS
                        figure.paste(panel.resize((350, 350), mode), (1416+360*col, 23+365*row))
            figure.save(OUT/f'first_page_{label}.png')
            records.append(dict(label=label, arm_qpos=q.tolist(),
                                translation_from_previous_figure_sensor4_m=delta,
                                contacts=actual_contacts, minimum_robot_panel_distance_m=distance,
                                closest_geom_pair=[robot_gid, hazard_gid], sensor_evidence=sensor_records))
            print(label, 'contact count', len(actual_contacts), 'panel distance', distance, flush=True)
    manifest = dict(
        type='Constructed, contact-checked MuJoCo pose illustrations; not recorded policy outcomes.',
        source_row=str(ROW), source_frame=0, scene_xml=str(SCENE),
        source_figure='images/whole_body/first_page_kitchen_items.png',
        source_provenance=provenance, camera_pose_cv=table.tolist(), crop_pixels=CROP,
        method='Original scene restored; link6 sensor4 translated with IK preserving orientation. No physics steps, image generation, object changes, highlights or text overlays.',
        interpretation='Collision shows actual MuJoCo contact. Avoidance shows a contact-free alternative pose, not a validated trajectory or a learned-policy rollout. This pair cannot establish that live proximity improves a policy.',
        sensors='All 40 sensors freshly rendered at native 8x8 resolution for each pose. RGB and dense depth are diagnostic views from sensor poses; proximity hardware does not output RGB.',
        first_page_sensor_rows=SENSORS,
        colour_scale_metres=[NEAR, FAR], grids='Solid black cell boundaries and outer borders.',
        unchanged_scene_verified=True, poses=records)
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    with ZipFile(OUT/'kitchen_collision_pair_assets.zip', 'w', ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix != '.zip':
                z.write(path, path.relative_to(OUT))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, help='Optional cache made by the source scene restoration utility.')
    main(parser.parse_args().snapshot)
