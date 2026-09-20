#!/usr/bin/env python3
"""Clean multi-environment figure assets from original MuJoCo scenes and sensors.

No synthetic point samples or image generation. All display arrays are freshly
rendered at the exported pose; source-recorded arrays are saved separately.
"""
import argparse
import json
from pathlib import Path
import zipfile

from export_table_occlusion_pair import main as restore_scene, ROOT
from export_sensor_modalities import depth_rgb, NEAR, FAR
from sensor_grid_image import bordered_sensor_grid
import mujoco
import numpy as np
import h5py
import cv2
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

OUT = ROOT / 'images/whole_body'


def lower_arm(m, d, dz):
    joints = [m.joint(f'robot_0/fr3_joint{i+1}') for i in range(7)]
    adr = [j.qposadr[0] for j in joints]
    q0 = d.qpos[adr].copy()
    cid = m.camera('robot_0/link6_sensor_4').id
    p0 = d.cam_xpos[cid].copy()
    rotation = d.cam_xmat[cid].reshape(3, 3).copy()
    def error(q):
        d.qpos[adr] = q
        mujoco.mj_forward(m, d)
        dr = Rotation.from_matrix(d.cam_xmat[cid].reshape(3, 3) @ rotation.T).as_rotvec()
        return np.r_[10*(d.cam_xpos[cid]-p0-[0, 0, dz]), dr, .002*(q-q0)]
    fit = least_squares(error, q0, bounds=([j.range[0] for j in joints],
                                          [j.range[1] for j in joints]), max_nfev=250)
    error(fit.x)
    assert fit.success and np.linalg.norm(d.cam_xpos[cid]-p0-[0, 0, dz]) < 1e-5
    for contact in d.contact:
        bodies = [m.body(int(m.geom_bodyid[g])).name for g in contact.geom]
        if (any(b.startswith('robot_0/') for b in bodies)
                and not all(b.startswith('robot_0/') for b in bodies)):
            assert contact.dist >= -.001, (bodies, contact.dist)
    return dict(vertical_shift_m=dz, original_arm_qpos=q0.tolist(), adjusted_arm_qpos=fit.x.tolist())


def geom_label(m, gid):
    body = m.body(int(m.geom_bodyid[gid])).name
    return body if body != 'world' else m.geom(gid).name


def render_external(renderer, m, d, opt, camera, pose=None):
    renderer.update_scene(d, camera=camera, scene_option=opt)
    if pose is not None:
        for c in renderer.scene.camera:
            c.pos = pose[:3, 3]
            c.forward = pose[:3, 2]
            c.up = -pose[:3, 1]
    return renderer.render().copy()


def export_scene(label, m, d, camera, selected, provenance, table_pose=None, recorded=None):
    out = OUT / label
    out.mkdir(parents=True, exist_ok=True)
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, 2496)
    m.vis.global_.offheight = max(m.vis.global_.offheight, 1408)
    table_opt = mujoco.MjvOption(); table_opt.geomgroup[3:] = 0; table_opt.sitegroup[:] = 0
    sensor_opt = mujoco.MjvOption(); sensor_opt.geomgroup[2] = 0; sensor_opt.geomgroup[4] = 1; sensor_opt.sitegroup[:] = 0
    rgb_opt = mujoco.MjvOption(); rgb_opt.geomgroup[2] = 0; rgb_opt.geomgroup[4] = 0; rgb_opt.sitegroup[:] = 0
    robot_ids = [i for i in range(m.ngeom) if m.body(int(m.geom_bodyid[i])).name.startswith('robot_0/')
                 and m.body(int(m.geom_bodyid[i])).name != 'robot_0/base']
    groups = m.geom_group.copy()
    with mujoco.Renderer(m, 1408, 2496) as r:
        for hidden, filename in [(False, 'scene_with_robot.png'), (True, 'scene_without_robot.png')]:
            m.geom_group[:] = groups
            if hidden: m.geom_group[robot_ids] = 5
            Image.fromarray(render_external(r, m, d, table_opt, camera, table_pose)).save(out / filename)
    m.geom_group[:] = groups
    wrist = 'robot_0/gripper/wrist_camera'
    wid = m.camera(wrist).id
    wrist_rotation_cv = d.cam_xmat[wid].reshape(3, 3) @ np.diag([1, -1, -1])
    tan_y = np.tan(np.deg2rad(m.cam_fovy[wid])/2)
    with mujoco.Renderer(m, 704, 1248) as r:
        r.update_scene(d, camera=wrist, scene_option=table_opt)
        Image.fromarray(r.render().copy()).save(out / 'wrist_rgb.png')
    names = sorted(m.camera(i).name for i in range(m.ncam) if '_sensor_' in m.camera(i).name)
    arrays, records, points, sensor_ids = [], [], [], []
    with mujoco.Renderer(m, 8, 8) as native, mujoco.Renderer(m, 1024, 1024) as dense:
        for sensor_index, camera_name in enumerate(names):
            name = camera_name.split('/')[-1]
            folder = out / name; folder.mkdir(exist_ok=True)
            if name not in selected:
                for filename in ['rgb.png', 'depth.png', 'dense_depth_metres.npy', 'rgb_depth_8x8.png']:
                    (folder / filename).unlink(missing_ok=True)
            cid = m.camera(camera_name).id
            native.update_scene(d, camera=camera_name, scene_option=sensor_opt)
            native.enable_depth_rendering(); depth = native.render().copy(); native.disable_depth_rendering()
            native.enable_segmentation_rendering(); seg = native.render().copy(); native.disable_segmentation_rendering()
            assert depth.shape == (8, 8) and np.all(np.isfinite(depth)) and np.all(depth > 0)
            arrays.append(depth)
            small = Image.fromarray(depth_rgb(depth))
            small.save(folder / 'readout_8x8_native.png')
            bordered_sensor_grid(small).save(folder / 'readout_8x8.png')
            np.save(folder / 'readout_8x8_metres.npy', depth)
            np.savetxt(folder / 'readout_8x8_metres.csv', depth, delimiter=',', fmt='%.9g')
            # Back-project all 64 actual rays, including background returns.
            v, u = np.indices((8, 8)); focal = 4 / np.tan(np.deg2rad(m.cam_fovy[cid])/2)
            local = np.stack([(u+.5-4)*depth/focal, (v+.5-4)*depth/focal, depth], -1)
            rot_cv = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])
            world = local.reshape(-1, 3) @ rot_cv.T + d.cam_xpos[cid]
            points.append(world); sensor_ids.extend([sensor_index]*64)
            wp = (world-d.cam_xpos[wid]) @ wrist_rotation_cv
            inside = ((wp[:, 2] > 0) & (abs(wp[:, 0]) < wp[:, 2]*tan_y*1248/704)
                      & (abs(wp[:, 1]) < wp[:, 2]*tan_y))
            counts = {}; outside = {}; closest = {}
            for ray, ((gid, kind), z) in enumerate(zip(seg.reshape(-1, 2), depth.ravel())):
                if gid < 0 or kind != int(mujoco.mjtObj.mjOBJ_GEOM) or z > .65: continue
                hit = geom_label(m, gid)
                if hit.startswith('robot_0/'): continue
                counts[hit] = counts.get(hit, 0) + 1
                closest[hit] = min(closest.get(hit, float('inf')), float(z))
                if not inside[ray]: outside[hit] = outside.get(hit, 0) + 1
            record = dict(sensor=name, nearby_surface_hits=counts, axial_depth_min_by_surface_m=closest,
                          nearby_hits_outside_wrist_fov=outside, sensor_position_world=d.cam_xpos[cid].tolist(),
                          sensor_rotation_cv=rot_cv.tolist(), fovy_degrees=float(m.cam_fovy[cid]))
            if recorded is not None:
                original = recorded[name]
                np.save(folder / 'source_recorded_8x8_metres.npy', original)
                record['difference_from_source_recorded_mae_m'] = float(np.mean(abs(original-depth)))
            records.append(record)
            if name in selected:
                dense.update_scene(d, camera=camera_name, scene_option=rgb_opt)
                Image.fromarray(dense.render().copy()).save(folder / 'rgb.png')
                dense.update_scene(d, camera=camera_name, scene_option=sensor_opt)
                dense.enable_depth_rendering(); dd = dense.render().copy(); dense.disable_depth_rendering()
                Image.fromarray(depth_rgb(dd)).save(folder / 'depth.png')
                np.save(folder / 'dense_depth_metres.npy', dd)
                triptych = Image.new('RGB', (3096, 1024), 'white')
                for j, file in enumerate(['rgb.png', 'depth.png', 'readout_8x8.png']):
                    with Image.open(folder / file) as panel: triptych.paste(panel, (1036*j, 0))
                triptych.save(folder / 'rgb_depth_8x8.png')
                print(label, name, counts, 'outside wrist', outside, flush=True)
    assert len(arrays) == 40
    np.savez_compressed(out / 'all_40_sensors.npz', depth_metres=np.stack(arrays),
                        sensor_names=np.array(names), points_world=np.concatenate(points),
                        point_sensor_ids=np.array(sensor_ids))
    all_grids = Image.new('RGB', (8*264-8, 5*264-8), 'white')
    for index, depth in enumerate(arrays):
        grid = bordered_sensor_grid(Image.fromarray(depth_rgb(depth)), size=256, line_width=2)
        all_grids.paste(grid, ((index % 8)*264, (index // 8)*264))
    all_grids.save(out / 'all_40_sensor_grids.png')
    # Text-free figure candidate; each separate source panel remains available.
    figure = Image.new('RGB', (2496, 1408 + 24 + 816), 'white')
    with Image.open(out / 'scene_with_robot.png') as im: figure.paste(im, (0, 0))
    for i, name in enumerate(selected[:3]):
        with Image.open(out / name / 'readout_8x8_native.png') as im:
            figure.paste(bordered_sensor_grid(im, size=816, line_width=4), (i*840, 1432))
    figure.save(out / 'scene_and_three_body_readouts.png')
    manifest = dict(environment=label, provenance=provenance, selected_sensors=selected,
                    sensor_count=40, native_samples_per_sensor=64, native_samples_total=2560,
                    rgb='Diagnostic views from proximity sensor poses; proximity hardware does not output RGB. Cosmetic skin and invisible collision proxies are hidden in RGB.',
                    depth='Fresh MuJoCo depth at exported pose; native 8x8 rendered independently, not downsampled dense depth. Sensor renderer hides cosmetic skin and includes original sensor geometry.',
                    grid='Solid black cell boundaries and outer border; no interpolation of range bins.',
                    colour_scale_metres=[NEAR, FAR], colour_map='turbo; near red, far blue; display saturates but arrays are unclipped',
                    evidence='Surface IDs come from native 8x8 segmentation. Nearby means axial depth <=0.65 m. Wrist exclusion is a geometric FOV test, not an occlusion test. Counts illustrate spatial coverage, not learned-policy efficacy.',
                    camera='External paired images use identical scene/camera; second image hides robot geometry only. Pedestal remains.',
                    records=records)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def recorded_kitchen():
    row = Path('/mnt/laptop/data/pact_pick_n_place_v2/data/v12/rows/000_2f77fef1863bdeb3')
    m, d, table_pose, k, opt, ids, meta = restore_scene(
        return_scene=True, row=row, scene_path=ROOT/'custom_scenes/pact_place_corridor_v12.xml', kitchen=True)
    m.vis.global_.fovy = np.rad2deg(2*np.arctan(k[1, 2]/k[1, 1]))
    camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    with h5py.File(row/'trajectory.h5') as h:
        raw = {name: h['traj_0/obs/proximity/'+name][0, -1] for name in h['traj_0/obs/proximity']}
        # Restore calibrated wrist optics as well as the arm-mounted skin.
        # Camera-system configuration can differ from the nominal MJCF camera.
        pose = h['traj_0/obs/sensor_param/wrist_camera/cam2world_gl'][0]
        wk = h['traj_0/obs/sensor_param/wrist_camera/intrinsic_cv'][0]
        wid = m.camera('robot_0/gripper/wrist_camera').id
        bid = m.cam_bodyid[wid]; br = d.xmat[bid].reshape(3, 3)
        m.cam_pos[wid] = br.T @ (pose[:3, 3]-d.xpos[bid])
        rotation = br.T @ pose[:3, :3] @ np.diag([1, -1, -1])
        mujoco.mju_mat2Quat(m.cam_quat[wid], rotation.ravel())
        m.cam_fovy[wid] = np.rad2deg(2*np.arctan(wk[1, 2]/wk[1, 1]))
        mujoco.mj_forward(m, d)
        assert np.linalg.norm(d.cam_xpos[wid]-pose[:3, 3]) < 1e-6
    adjustment = lower_arm(m, d, -.16)
    export_scene('kitchen_items', m, d, camera,
                 ['link6_sensor_3', 'link5_back_sensor_4', 'link2_sensor_3'],
                 dict(source_row=str(row), source_frame=0, adjusted_simulation_pose=adjustment,
                      kitchen_auxiliary_geometry=meta['figure_kitchen_overlay'],
                      note='Original recorded scene reconstructed with its existing kitchen overlay. Link 6 lowered 16 cm to expose bottles in sensor 3. All displayed readings recomputed; source recorded arrays saved separately.'),
                 table_pose=table_pose, recorded=raw)
    for name in ['exo_camera_1', 'wrist_camera']:
        cap = cv2.VideoCapture(str(row/f'episode_00000000_{name}.mp4'))
        ok, bgr = cap.read(); cap.release(); assert ok
        Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(OUT/'kitchen_items'/f'source_recorded_{name}_frame000.png')


def recorded_fumehood():
    m, d, pose, k, opt, ids, meta = restore_scene(return_scene=True)
    m.vis.global_.fovy = np.rad2deg(2*np.arctan(k[1, 2]/k[1, 1]))
    camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    adjustment = lower_arm(m, d, -.08)
    export_scene('fumehood_clutter', m, d, camera,
                 ['link6_sensor_4', 'link6_sensor_3', 'link5_back_sensor_4'],
                 dict(source_row='/mnt/laptop/data/pact_pick_n_place_v2/data/v1011d/rows/000_187ba0ce76cb3011',
                      source_frame=0, adjusted_simulation_pose=adjustment,
                      note='Original v1011d recorded scene geometry; link 6 lowered 8 cm. Fresh simulation readings at this pose.'),
                 table_pose=pose)


def export_snapshot(path, selected):
    m = mujoco.MjModel.from_binary_path(str(path/'model.mjb')); d = mujoco.MjData(m)
    with np.load(path/'state.npz') as state:
        for name in ['qpos', 'qvel', 'mocap_pos', 'mocap_quat']: getattr(d, name)[:] = state[name]
    mujoco.mj_forward(m, d)
    meta = json.loads((path/'metadata.json').read_text())
    cam = mujoco.MjvCamera(); cam.lookat = meta['camera']['lookat']; cam.distance = meta['camera']['distance']
    cam.azimuth = meta['camera']['presentation_views'][0]['azimuth']; cam.elevation = -18
    export_scene(path.name, m, d, cam, selected,
                 dict(scene=meta['scene'], config=meta['config'], house=meta['house'], seed=2026,
                      source_snapshot=str(path), note=meta['new_render_provenance']))


def package():
    cases = [(p.parent, json.loads(p.read_text())) for p in sorted(OUT.glob('*/manifest.json'))]
    html = ['<!doctype html><meta charset="utf-8"><title>Whole-body sensing examples</title>',
            '<style>body{font:17px Arial;max-width:1400px;margin:35px auto;line-height:1.5}img{width:100%}section{margin:60px 0}a{color:#1468b3}</style>',
            '<h1>Whole-body sensing: figure assets</h1><p>Original simulation scene geometry and native sensor depth. '
            'These images illustrate coverage around the arm; they do not measure policy improvement. '
            'Every PNG is text-free; 8×8 panels have black borders. RGB is a diagnostic sensor viewpoint.</p>']
    for folder, manifest in cases:
        label = folder.name
        html += [f'<section><h2>{label.replace("_", " ").title()}</h2>',
                 f'<p>{manifest["provenance"]["note"]}</p>',
                 f'<a href="{label}/scene_with_robot.png"><img src="{label}/scene_with_robot.png"></a>',
                 f'<p><a href="{label}/scene_without_robot.png">Without robot</a> · <a href="{label}/wrist_rgb.png">Wrist view</a> · <a href="{label}/all_40_sensor_grids.png">All 40 sensor grids</a></p>']
        for name in manifest['selected_sensors']:
            record = next(r for r in manifest['records'] if r['sensor'] == name)
            hits = '; '.join(f'{surface}: {count}' for surface, count in record['nearby_hits_outside_wrist_fov'].items())
            html += [f'<h3>{name}: RGB / depth / 8×8</h3>',
                     f'<img src="{label}/{name}/rgb_depth_8x8.png">',
                     f'<p>Nearby ray hits outside wrist FOV: {hits or "none"}. See manifest for surface IDs and depth values.</p>']
        html += ['</section>']
    (OUT/'index.html').write_text('\n'.join(html))
    with zipfile.ZipFile(OUT/'whole_body_figure_assets.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix not in ['.zip', '.mjb'] and 'source_snapshots' not in path.parts:
                z.write(path, path.relative_to(OUT))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitchen', action='store_true')
    parser.add_argument('--fumehood', action='store_true')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--sensors', nargs=3)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.kitchen: recorded_kitchen()
    if args.fumehood: recorded_fumehood()
    if args.snapshot: export_snapshot(args.snapshot, args.sensors)
    package()
