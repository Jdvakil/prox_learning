#!/usr/bin/env python3
"""Counterfactual sensor-stop demo and exact native 8x8 point overlays.

Same motor trajectory with guard disabled/enabled. The guard uses only live
link-6 depth samples; object IDs are used only for visualization/validation.
"""
import argparse
import json
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

os.environ.setdefault('MPLCONFIGDIR', '/tmp/prox-sensor-plot')
from export_kitchen_collision_pair import ROOT, ROW, scene
from export_kitchen_contact_dynamics import environment_contacts
import export_kitchen_motion_sequences as layout
from proximity_pointcloud import native_link6, recorded_link6, save_cloud
from sensor_grid_image import bordered_sensor_grid
import mujoco
import numpy as np
from PIL import Image

OUT = ROOT/'images/whole_body/sensor_guard_pointcloud'
TIMES = [0., .4, .8, 2., 3.6, 4.6]
VIEWS = {
    'side_view': dict(lookat=[.49, -.12, .98], distance=.88, azimuth=62, elevation=-12, fovy=42),
    'context': dict(lookat=[.40, 0, .92], distance=1.55, azimuth=-40, elevation=-20, fovy=42),
}
COLORS = np.array([[.05,.85,.95,1], [.7,.35,.95,1], [.10,.85,.42,1],
                   [1,.28,.10,1], [1,.76,.04,1], [.95,.22,.65,1]])


def simulate(m, initial, threshold):
    adr = [m.joint(f'robot_0/fr3_joint{i+1}').qposadr[0] for i in range(7)]
    act = [m.actuator(f'robot_0/fr3_joint{i+1}').id for i in range(7)]
    gripper = m.actuator('robot_0/gripper/fingers_actuator').id
    targets = np.load(ROOT/'images/whole_body/kitchen_contact_dynamics/collision_physics_states.npz')['target_arm_waypoints']
    step_period = round(.02/m.opt.timestep)
    assert abs(step_period*m.opt.timestep-.02) < 1e-9
    results = {}
    with mujoco.Renderer(m, 8, 8) as native:
        for label, enabled in [('collision', False), ('sensor_stop', True)]:
            d = mujoco.MjData(m)
            for key, value in initial.items(): getattr(d, key)[:] = value
            d.qpos[adr] = targets[0]; d.qvel[:] = 0; d.ctrl[act] = targets[0]; d.ctrl[gripper] = 0
            mujoco.mj_forward(m, d)
            for _ in range(400):
                mujoco.mj_step(m, d)
                assert not environment_contacts(m, d)
            mujoco.mj_forward(m, d)
            latched = None; event = None; clouds = []; states = []; observations = []; logs = []
            for step in range(round(5/m.opt.timestep)+1):
                t = step*m.opt.timestep
                if step % step_period == 0:
                    cloud = native_link6(m, d, native)  # No segmentation/object IDs.
                    depth = cloud['depth_m']
                    valid = np.isfinite(depth) & (depth > 0)
                    trigger = valid & (depth <= threshold)
                    if enabled and latched is None and np.any(trigger):
                        s, v, u = np.unravel_index(np.argmin(np.where(valid, depth, np.inf)), depth.shape)
                        latched = d.qpos[adr].copy()
                        event = dict(time_s=float(t), sensor=str(cloud['sensor_names'][s]),
                                     pixel_u=int(u), pixel_v=int(v), depth_m=float(depth[s, v, u]),
                                     threshold_m=threshold, observation_index=len(clouds))
                    clouds.append(cloud)
                    states.append({key: getattr(d, key).copy() for key in ['qpos', 'qvel', 'ctrl']})
                    observations.append(dict(time_s=float(t), minimum_depth_m=float(depth[valid].min()),
                                             guard_latched=latched is not None))
                phase = min(t/4, 1)*5; seg = min(int(phase), 4); f = phase-seg; f = f*f*(3-2*f)
                commanded = (1-f)*targets[seg]+f*targets[seg+1]
                d.ctrl[act] = commanded if latched is None else latched
                mujoco.mj_step(m, d); mujoco.mj_forward(m, d)
                contacts = environment_contacts(m, d)
                if enabled: assert not contacts, (t, contacts)
                logs.append(dict(time_s=float(t), contacts=contacts,
                                 normal_force_N=sum(c['normal_force_N'] for c in contacts)))
            assert (event is not None) if enabled else any(r['contacts'] for r in logs)
            data = {key: np.stack([s[key] for s in states]) for key in ['qpos', 'qvel', 'ctrl']}
            data.update({key: np.stack([c[key] for c in clouds]) for key in
                         ['depth_m', 'intrinsic_cv', 'camera_to_world_cv', 'points_world_m']})
            data.update(sensor_names=clouds[0]['sensor_names'], times_s=np.array([o['time_s'] for o in observations]),
                        mocap_pos=d.mocap_pos.copy(), mocap_quat=d.mocap_quat.copy())
            np.savez_compressed(OUT/f'{label}_live_observations.npz', **data)
            (OUT/f'{label}_contacts.json').write_text(json.dumps(logs, indent=2))
            (OUT/f'{label}_observations.json').write_text(json.dumps(observations, indent=2))
            results[label] = dict(data=data, event=event, logs=logs)
            print(label, 'trigger', event, 'contact steps', sum(bool(r['contacts']) for r in logs), flush=True)
    stop = results['sensor_stop']['event']['observation_index']
    for key in ['qpos', 'qvel', 'depth_m']:
        assert np.array_equal(results['collision']['data'][key][:stop+1], results['sensor_stop']['data'][key][:stop+1])
    return results


def add_points(m, render_scene, cloud, rays=False, radius=.003):
    """Place markers at measured XYZs; never move/snap them onto a mesh."""
    for s, points in enumerate(cloud['points_world_m']):
        origin = cloud['camera_to_world_cv'][s, :3, 3]
        ids = cloud.get('geom_ids')
        for v in range(8):
            for u in range(8):
                point = points[v, u]
                if not np.all(np.isfinite(point)) or cloud['depth_m'][s, v, u] <= 0: continue
                gid = int(ids[s, v, u]) if ids is not None else None
                if gid == -1: continue  # Renderer background/far-plane, not a surface return.
                geom = render_scene.geoms[render_scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius]*3),
                                   point, np.eye(3).ravel(), COLORS[s % len(COLORS)])
                geom.emission = .5; render_scene.ngeom += 1
                # Optional line layer only for rays known to hit the panel.
                # Object identity is used here, never by the live guard.
                if rays and gid is not None and m.body(int(m.geom_bodyid[gid])).name == 'pact_intrusion_right':
                    geom = render_scene.geoms[render_scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                                       np.eye(3).ravel(), COLORS[s % len(COLORS)])
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1.4, origin, point)
                    render_scene.ngeom += 1


def validate_cloud(m, d, cloud):
    """Independent surface check against the original analytic panel box."""
    errors = []; counts = []
    for s in range(len(cloud['sensor_names'])):
        hits = 0
        for p, gid in zip(cloud['points_world_m'][s].reshape(-1, 3), cloud['geom_ids'][s].ravel()):
            if gid < 0 or m.body(int(m.geom_bodyid[gid])).name != 'pact_intrusion_right': continue
            local = (p-d.geom_xpos[gid]) @ d.geom_xmat[gid].reshape(3, 3)
            q = np.abs(local)-m.geom_size[gid]
            signed_distance = np.linalg.norm(np.maximum(q, 0))+min(float(q.max()), 0)
            errors.append(abs(float(signed_distance))); hits += 1
        counts.append(hits)
    assert errors and max(errors) < 1e-4, errors
    return dict(panel_hits_by_sensor=counts, panel_hit_count=sum(counts),
                maximum_panel_surface_error_m=max(errors), total_native_samples=int(cloud['depth_m'].size))


def render(m, initial, results):
    d = mujoco.MjData(m); opt = mujoco.MjvOption(); opt.geomgroup[3:] = 0; opt.sitegroup[:] = 0
    evidence = []
    def restore(data, i):
        for key in ['qpos', 'qvel', 'ctrl']: getattr(d, key)[:] = data[key][i]
        for key in ['mocap_pos', 'mocap_quat']: getattr(d, key)[:] = data[key]
        mujoco.mj_forward(m, d)
    with mujoco.Renderer(m, 8, 8) as native, mujoco.Renderer(m, 1440, 1920) as external:
        # Direct HDF5 example: these 384 samples predate this demonstration.
        for key, value in initial.items(): getattr(d, key)[:] = value
        mujoco.mj_forward(m, d)
        recorded = recorded_link6(ROW/'trajectory.h5', frame=0)
        save_cloud(OUT/'recorded_dataset_frame000', recorded)
        (OUT/'recorded_dataset_frame000/source.json').write_text(json.dumps(dict(
            hdf5=str(ROW/'trajectory.h5'), frame=0, temporal_substep=3,
            depth_address='traj_0/obs/proximity/link6_sensor_i[0,3,:,:]',
            camera_address='traj_0/obs/sensor_param/link6_sensor_i/{intrinsic_cv,cam2world_gl}[0]',
            samples=int(recorded['depth_m'].size), provenance='Unmodified recorded simulated readings, not hardware data.'), indent=2))
        cam = mujoco.MjvCamera()
        for key, value in VIEWS['context'].items():
            if key == 'fovy': m.vis.global_.fovy = value
            else: setattr(cam, key, value)
        external.update_scene(d, camera=cam, scene_option=opt)
        add_points(m, external.scene, recorded)
        Image.fromarray(external.render().copy()).save(OUT/'recorded_dataset_frame000/points_on_scene.png')
        for view, settings in VIEWS.items():
            cam = mujoco.MjvCamera()
            for key, value in settings.items():
                if key == 'fovy': m.vis.global_.fovy = value
                else: setattr(cam, key, value)
            for label, result in results.items():
                data = result['data']; folder = OUT/view/label; folder.mkdir(parents=True, exist_ok=True)
                strip = Image.new('RGB', (5840, 720), 'white')
                for frame, time in enumerate(TIMES):
                    index = int(np.argmin(abs(data['times_s']-time))); restore(data, index)
                    cloud = native_link6(m, d, native, segmentation=True)
                    assert np.array_equal(cloud['depth_m'], data['depth_m'][index])
                    assert np.allclose(cloud['points_world_m'], data['points_world_m'][index], atol=1e-9, rtol=0)
                    check = validate_cloud(m, d, cloud)
                    if view == 'side_view':
                        save_cloud(OUT/'samples'/label/f'frame_{frame+1:02d}', cloud)
                        evidence.append(dict(sequence=label, frame=frame+1, time_s=float(data['times_s'][index]), **check))
                    external.update_scene(d, camera=cam, scene_option=opt)
                    add_points(m, external.scene, cloud)
                    im = Image.fromarray(external.render().copy()); im.save(folder/f'frame_{frame+1:02d}.png')
                    strip.paste(im.resize((960, 720), Image.Resampling.LANCZOS), (976*frame, 0))
                strip.save(OUT/view/f'{label}_strip.png')
            both = Image.new('RGB', (5840, 1456), 'white')
            for row, label in enumerate(results):
                with Image.open(OUT/view/f'{label}_strip.png') as im: both.paste(im, (0, row*736))
            both.save(OUT/view/'comparison.png')
            # Exact trigger observation, with a separate optional beam layer.
            guarded = results['sensor_stop']; index = guarded['event']['observation_index']
            restore(guarded['data'], index); cloud = native_link6(m, d, native, segmentation=True)
            if view == 'side_view':
                save_cloud(OUT/'trigger_measurements', cloud)
                evidence.append(dict(sequence='trigger', time_s=guarded['event']['time_s'], **validate_cloud(m, d, cloud)))
                # Native arrays remain separate; grid colours are only display.
                from matplotlib import colormaps
                grid = Image.new('RGB', (6*256+5*8, 256), 'white')
                for s, z in enumerate(cloud['depth_m']):
                    rgb = (colormaps['turbo'](1-np.clip((z-.005)/.195, 0, 1))[..., :3]*255).astype(np.uint8)
                    grid.paste(bordered_sensor_grid(Image.fromarray(rgb), 256, 2), (s*264, 0))
                grid.save(OUT/'trigger_6_sensors_8x8.png')
            for rays, name in [(False, 'points_at_trigger.png'), (True, 'rays_at_trigger.png')]:
                external.update_scene(d, camera=cam, scene_option=opt); add_points(m, external.scene, cloud, rays=rays)
                Image.fromarray(external.render().copy()).save(OUT/view/name)
            print('Rendered', view, flush=True)
    return evidence


def main(snapshot=None, threshold=.14):
    OUT.mkdir(parents=True, exist_ok=True)
    m, d, _ = scene(snapshot); m.vis.global_.offheight = max(m.vis.global_.offheight, 1440)
    initial = {key: getattr(d, key).copy() for key in ['qpos', 'qvel', 'mocap_pos', 'mocap_quat']}
    results = simulate(m, initial, threshold)
    evidence = render(m, initial, results)
    layout.OUT = OUT; layout.VIEWS = {name: c['azimuth'] for name, c in VIEWS.items()}
    layout.PATHS = {'collision': None, 'sensor_stop': None}
    layout.deck(description='Same scripted motor command with live sensor guard OFF (top) and ON (bottom). Points are native link6 8x8 readings backprojected with camera calibration. New simulator demonstration, not learned-policy evidence.')
    metrics = {}
    for label, result in results.items():
        hits = [row for row in result['logs'] if row['contacts']]
        metrics[label] = dict(contact_steps=len(hits), first_contact_s=hits[0]['time_s'] if hits else None,
                              peak_normal_force_N=max(r['normal_force_N'] for r in result['logs']))
    manifest = dict(
        source_scene_row=str(ROW), type='New sensor-guard counterfactual in original simulated scene; plus separately exported original HDF5 readings.',
        claim='Enabling this explicit live proximity stop prevented the contact caused by the same approach command in this one simulation. This is not PACT/ACT evaluation or a general safety guarantee.',
        guard='At 50 Hz, latch current arm joint positions if ANY finite positive link6 native depth <= threshold. Once latched, hold those motor targets. No RGB, object IDs, contact flags, object poses or time-index trigger is used in the decision.',
        threshold_m=threshold, depth_units='Metres along sensor optical Z; not Euclidean ray length.',
        trigger=results['sensor_stop']['event'], metrics=metrics,
        pairing='Same initial state, dynamics, target trajectory, gripper command and sensor sampling. Joint positions, velocities and depth observations match exactly through the trigger observation.',
        source_renderer='molmo_spaces/env/env.py: record_proximity_depths; native square 8x8, group2 hidden, group4 enabled, skybox disabled.',
        points='Six link6 sensors x 8x8 = 384 samples per snapshot. Every XYZ derives from its pixel depth and calibrated pose. No random samples, surface fitting, mesh sampling, interpolation, densification or snapping.',
        display='3 mm radius spheres centered on the measured XYZs. Colour identifies sensor. All finite surface-return points are submitted to the depth-tested 3D renderer; field of view and occlusion determine visibility. Background misses are not drawn but all 384 raw samples remain saved. Optional lines connect real sensor origins to actual panel returns.',
        recorded_example='recorded_dataset_frame000 reads unmodified traj_0/obs/proximity/link6_sensor_i[0,-1] and stored calibration directly from the original trajectory.h5. These are recorded simulated readings, not physical hardware measurements.',
        live_example='All other clouds are fresh native simulated sensor readings from the motor-driven demonstration, saved at the exact observed robot state. They are not falsely labelled as original dataset samples.',
        image_order='Top collision with guard disabled; bottom sensor-triggered stop with guard enabled; six time-matched frames left to right.',
        frame_times_s=TIMES, cameras=VIEWS, surface_checks=evidence)
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    code = OUT/'code'; code.mkdir(exist_ok=True)
    for name in ['proximity_pointcloud.py', 'export_sensor_guard_pointcloud.py']:
        (code/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    with ZipFile(OUT/'sensor_guard_pointcloud_assets.zip', 'w', ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix != '.zip': z.write(path, path.relative_to(OUT))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, help='Optional cache of the restored original scene.')
    parser.add_argument('--stop-distance', type=float, default=.14, help='Positive axial-depth threshold in metres.')
    args = parser.parse_args()
    if not np.isfinite(args.stop_distance) or args.stop_distance <= 0:
        parser.error('--stop-distance must be finite and positive')
    main(args.snapshot, args.stop_distance)
