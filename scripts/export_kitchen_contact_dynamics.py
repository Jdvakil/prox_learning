#!/usr/bin/env python3
"""Motor-driven contact demonstration in the original kitchen scene.

The source objects/panel are static. This is a new MuJoCo demonstration with
scripted joint targets, not a recorded dataset episode or learned-policy result.
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from export_kitchen_collision_pair import ROOT, ROW, SCENE, scene
import export_kitchen_motion_sequences as layout
import mujoco
import numpy as np
from PIL import Image, ImageDraw

OUT = ROOT/'images/whole_body/kitchen_contact_dynamics'
TIMES = [0., .8, 2., 3.4, 3.6, 4.6]
CAMERAS = {
    'side_view': dict(lookat=[.47, -.09, .93], distance=.76, azimuth=80, elevation=-3, fovy=38),
    'context': dict(lookat=[.32, 0, .89], distance=1.65, azimuth=55, elevation=-20, fovy=42),
}


def environment_contacts(m, d):
    rows = []
    for i, c in enumerate(d.contact):
        bodies = [m.body(int(m.geom_bodyid[g])).name for g in c.geom]
        robot = [name.startswith('robot_0/') for name in bodies]
        if not (any(robot) and not all(robot)):
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, wrench)
        rows.append(dict(bodies=bodies, geom_ids=c.geom.tolist(),
                         distance_m=float(c.dist), position_world=c.pos.tolist(),
                         normal_world=c.frame[:3].tolist(), normal_force_N=float(wrench[0])))
    return rows


def simulate(m, initial):
    qadr = [m.joint(f'robot_0/fr3_joint{i+1}').qposadr[0] for i in range(7)]
    act = [m.actuator(f'robot_0/fr3_joint{i+1}').id for i in range(7)]
    gripper = m.actuator('robot_0/gripper/fingers_actuator').id
    results = {}
    for label in ['collision', 'avoidance']:
        d = mujoco.MjData(m)
        for key in ['qpos', 'qvel', 'mocap_pos', 'mocap_quat']:
            getattr(d, key)[:] = initial[key]
        qs = np.load(ROOT/'images/whole_body/kitchen_motion_sequences'/f'{label}_path.npz')['frame_arm_qpos'].copy()
        if label == 'collision':
            # Continue the command beyond first touch. Only motor targets are
            # extended; actual robot joints evolve under mj_step and contact.
            qs[-1] += 3*(qs[-1]-qs[-2])
        for j, adr in enumerate(qadr):
            bounds = m.jnt_range[m.actuator_trnid[act[j], 0]]
            assert np.all((qs[:, j] >= bounds[0]) & (qs[:, j] <= bounds[1]))
        d.qpos[qadr] = qs[0]; d.qvel[:] = 0
        d.ctrl[act] = qs[0]; d.ctrl[gripper] = 0
        mujoco.mj_forward(m, d)
        for _ in range(400):
            mujoco.mj_step(m, d)
            assert not environment_contacts(m, d), 'Initial settling touched the environment'
        fixed = [i for i in range(m.nbody) if not m.body(i).name.startswith('robot_0/')]
        fixed_pos = d.xpos[fixed].copy(); fixed_rot = d.xmat[fixed].copy()
        traces = []; states = []; velocities = []; controls = []; snapshots = []
        target_knots = qs.copy()
        count = round(5/m.opt.timestep)
        for step in range(count+1):
            requested_time = step*m.opt.timestep
            phase = min(requested_time/4, 1)*5
            segment = min(int(phase), 4); f = phase-segment
            f = f*f*(3-2*f)
            target = (1-f)*qs[segment]+f*qs[segment+1]
            d.ctrl[act] = target
            mujoco.mj_step(m, d)
            # Synchronize force/geometry queries with the integrated state.
            mujoco.mj_forward(m, d)
            cs = environment_contacts(m, d)
            assert np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))
            assert np.array_equal(d.xpos[fixed], fixed_pos)
            assert np.array_equal(d.xmat[fixed], fixed_rot)
            if label == 'avoidance':
                assert not cs, (requested_time, cs)
            else:
                assert all(c['bodies'] == ['pact_intrusion_right', 'robot_0/fr3_link7'] for c in cs)
            rec = dict(time_s=float(requested_time), contacts=cs,
                       summed_normal_force_N=sum(c['normal_force_N'] for c in cs),
                       arm_tracking_error_rad=float(np.linalg.norm(d.qpos[qadr]-target)))
            traces.append(rec)
            if step % 10 == 0:
                states.append(d.qpos.copy()); velocities.append(d.qvel.copy())
                controls.append(d.ctrl.copy()); snapshots.append(rec)
        if label == 'collision':
            assert any(r['summed_normal_force_N'] > 1 for r in traces)
            assert traces[-1]['contacts'], 'Expected sustained blocked contact'
        data = dict(qpos=np.array(states), qvel=np.array(velocities), ctrl=np.array(controls),
                    times_s=np.array([r['time_s'] for r in snapshots]),
                    mocap_pos=d.mocap_pos.copy(), mocap_quat=d.mocap_quat.copy(),
                    target_arm_waypoints=target_knots)
        np.savez_compressed(OUT/f'{label}_physics_states.npz', **data)
        (OUT/f'{label}_contact_log.json').write_text(json.dumps(traces, indent=2))
        with (OUT/f'{label}_contact_force.csv').open('w') as f:
            writer = csv.writer(f); writer.writerow(['time_s', 'contact_count', 'normal_force_N', 'arm_tracking_error_rad'])
            writer.writerows([r['time_s'], len(r['contacts']), r['summed_normal_force_N'], r['arm_tracking_error_rad']] for r in traces)
        results[label] = (data, snapshots, traces)
        print(label, 'contact steps', sum(bool(r['contacts']) for r in traces),
              'peak normal force N', max(r['summed_normal_force_N'] for r in traces), flush=True)
    assert np.array_equal(results['collision'][0]['qpos'][:41], results['avoidance'][0]['qpos'][:41])
    return results


def mark_contact(scene, contact):
    point = np.array(contact['position_world'])
    normal = np.array(contact['normal_world'])
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.004]*3),
                       point, np.eye(3).ravel(), np.array([1., .05, .02, 1.]))
    scene.ngeom += 1
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                       np.eye(3).ravel(), np.array([1., .05, .02, 1.]))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .0025, point, point+.055*normal)
    scene.ngeom += 1


def contact_overlay(image, scene, contacts, fovy):
    """Project actual solver contacts; draw diagnostic marks above occluding mesh."""
    result = image.copy(); draw = ImageDraw.Draw(result)
    eye = np.mean([c.pos for c in scene.camera], axis=0)
    forward = np.mean([c.forward for c in scene.camera], axis=0)
    forward /= np.linalg.norm(forward)
    up = np.mean([c.up for c in scene.camera], axis=0)
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    focal = image.height/(2*np.tan(np.deg2rad(fovy)/2))
    def project(point):
        delta = point-eye; depth = delta@forward
        assert depth > 0
        return np.array([image.width/2+focal*(delta@right)/depth,
                         image.height/2-focal*(delta@up)/depth])
    for contact in contacts:
        point = np.array(contact['position_world'])
        start = project(point); end = project(point+.055*np.array(contact['normal_world']))
        direction = (end-start)/np.linalg.norm(end-start)
        cross = np.array([-direction[1], direction[0]])
        red = (235, 40, 25)
        draw.line([tuple(start), tuple(end)], fill=red, width=6)
        draw.polygon([tuple(end), tuple(end-20*direction+9*cross), tuple(end-20*direction-9*cross)], fill=red)
        draw.ellipse((start[0]-13, start[1]-13, start[0]+13, start[1]+13), outline=red, width=5)
    return result


def export(m, results):
    d = mujoco.MjData(m)
    opt = mujoco.MjvOption(); opt.geomgroup[3:] = 0; opt.sitegroup[:] = 0
    def restore(data, index):
        for key in ['qpos', 'qvel', 'ctrl']: getattr(d, key)[:] = data[key][index]
        for key in ['mocap_pos', 'mocap_quat']: getattr(d, key)[:] = data[key]
        mujoco.mj_forward(m, d)
    for view, settings in CAMERAS.items():
        cam = mujoco.MjvCamera()
        for key, value in settings.items():
            if key == 'fovy': m.vis.global_.fovy = value
            else: setattr(cam, key, value)
        with mujoco.Renderer(m, 1440, 1920) as r:
            for label, (data, snapshots, _) in results.items():
                folder = OUT/view/label; folder.mkdir(parents=True, exist_ok=True)
                strip = Image.new('RGB', (5840, 720), 'white')
                for frame, time in enumerate(TIMES):
                    index = int(np.argmin(abs(data['times_s']-time)))
                    restore(data, index); r.update_scene(d, camera=cam, scene_option=opt)
                    image = Image.fromarray(r.render().copy())
                    image.save(folder/f'frame_{frame+1:02d}.png')
                    strip.paste(image.resize((960, 720), Image.Resampling.LANCZOS), (976*frame, 0))
                    if view == 'side_view' and label == 'collision' and frame == 5:
                        contact_overlay(image, r.scene, snapshots[index]['contacts'], settings['fovy']).save(OUT/'contact_debug.png')
                        for contact in snapshots[index]['contacts']: mark_contact(r.scene, contact)
                        Image.fromarray(r.render().copy()).save(OUT/'contact_debug_depth_tested.png')
                strip.save(OUT/view/f'{label}_strip.png')
        both = Image.new('RGB', (5840, 1456), 'white')
        for row, label in enumerate(results):
            with Image.open(OUT/view/f'{label}_strip.png') as im: both.paste(im, (0, row*736))
        both.save(OUT/view/'comparison.png')
        if view == 'side_view':
            with mujoco.Renderer(m, 720, 960) as r:
                for label, (data, _, _) in results.items():
                    # Encode real simulator renders, with no optical flow or
                    # interpolated image content. Simulation states are 50 Hz;
                    # every other stored state is shown at 25 fps.
                    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '960x720', '-r', '25',
                               '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                               '-movflags', '+faststart', str(OUT/f'{label}_motion.mp4')]
                    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
                    for i in range(0, len(data['times_s']), 2):
                        restore(data, i); r.update_scene(d, camera=cam, scene_option=opt)
                        proc.stdin.write(r.render().tobytes())
                    proc.stdin.close(); assert proc.wait() == 0
        print('Exported', view, flush=True)


def main(snapshot=None):
    OUT.mkdir(parents=True, exist_ok=True)
    m, d, _ = scene(snapshot)
    m.vis.global_.offheight = max(m.vis.global_.offheight, 1440)
    initial = {key: getattr(d, key).copy() for key in ['qpos', 'qvel', 'mocap_pos', 'mocap_quat']}
    results = simulate(m, initial)
    export(m, results)
    # Reuse the existing editable six-frame/two-row PPTX layout.
    layout.OUT = OUT; layout.VIEWS = {name: settings['azimuth'] for name, settings in CAMERAS.items()}
    layout.deck(description='Motor-driven MuJoCo contact demonstration with scripted targets. Close-up and context views; collision top, avoidance bottom; six separate frames per row. Original scene assets, not a recorded policy trial.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3), layout='constrained')
    for label, (_, _, trace) in results.items():
        ax.plot([r['time_s'] for r in trace], [r['summed_normal_force_N'] for r in trace],
                label=label.capitalize(), linewidth=2)
    ax.set(xlabel='Time (s)', ylabel='Normal contact force (N)')
    ax.spines[['top', 'right']].set_visible(False); ax.legend(frameon=False)
    fig.savefig(OUT/'contact_force_trace.png', dpi=220); plt.close(fig)
    evidence = {}
    for label, (_, _, trace) in results.items():
        hits = [r for r in trace if r['contacts']]
        evidence[label] = dict(checked_motion_steps=len(trace), contact_steps=len(hits),
                               first_contact_time_s=hits[0]['time_s'] if hits else None,
                               peak_normal_force_N=max(r['summed_normal_force_N'] for r in trace),
                               deepest_penetration_m=min([c['distance_m'] for r in hits for c in r['contacts']], default=0),
                               exported_times_s=TIMES)
    manifest = dict(
        type='New physics-stepped simulator demonstration with scripted motor targets; not a recorded dataset or learned-policy rollout.',
        source_row=str(ROW), source_scene=str(SCENE), engine_version=mujoco.__version__,
        integration_timestep_s=float(m.opt.timestep), settling_steps=400,
        controller='Original model position actuators and torque limits. Gripper command 0. No external force, gravity compensation, manual pose assignment or contact correction during motion.',
        targets='Existing six IK waypoints, smoothstep interpolation over 4 s and 1 s hold. Collision final command extends the last joint-space increment by 3x, so the motor attempts to continue after touch; actual joints remain governed by dynamics.',
        objects='Original restored geometry and poses; all kitchen props and the hood panel remain static. Only robot joints evolve dynamically. No physical-hardware claim.',
        contact_evidence='mj_forward contact records and mj_contactForce at every 2 ms step. Only detected environment contact is panel versus link7. Forces are simulated, not measured hardware forces.',
        visual_evidence='Fixed close-up camera shows the gap closing, wrist contact and deflection under load. Clean PNGs and MP4s have no annotations. contact_debug.png projects the solver contact into the image and draws a ring/normal arrow above the mesh; contact_debug_depth_tested.png uses native 3D markers. Arrows indicate normal direction with fixed 0.055 m world length, not force magnitude.',
        caveat='Avoidance is a hand-designed detour, not proof of proximity-driven policy performance or a completed manipulation task.',
        cameras=CAMERAS, clean_frame_pixels=[1920, 1440], evidence=evidence)
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    with ZipFile(OUT/'kitchen_contact_dynamics_assets.zip', 'w', ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix != '.zip': z.write(path, path.relative_to(OUT))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, help='Optional cache of the original restored experiment scene.')
    main(parser.parse_args().snapshot)
