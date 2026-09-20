#!/usr/bin/env python3
"""Render the original scene with link 6 lowered 8 cm using joint-space IK.

All outputs are new simulation renders, including native 8x8 depth. Recorded
measurements are never attached to the adjusted pose. No physics is stepped.
"""
import json
import zipfile

from export_table_occlusion_pair import main as restore_scene, ROOT, ROW
from export_sensor_modalities import depth_rgb, NEAR, FAR
from sensor_grid_image import bordered_sensor_grid
import mujoco
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

OUT = ROOT / 'images/table_occlusion/lowered_robot'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m, d, table_pose, table_k, table_opt, robot_ids, scene = restore_scene(return_scene=True)
    joints = [m.joint(f'robot_0/fr3_joint{i+1}') for i in range(7)]
    addresses = np.array([j.qposadr[0] for j in joints])
    original_q = d.qpos[addresses].copy()
    original_body_positions = d.xpos.copy()
    cid = m.camera('robot_0/link6_sensor_4').id
    original_position = d.cam_xpos[cid].copy()
    original_rotation = d.cam_xmat[cid].reshape(3, 3).copy()
    target = original_position + [0, 0, -.08]

    def residual(q):
        d.qpos[addresses] = q
        mujoco.mj_forward(m, d)
        rotation_error = Rotation.from_matrix(
            d.cam_xmat[cid].reshape(3, 3) @ original_rotation.T).as_rotvec()
        return np.r_[10 * (d.cam_xpos[cid] - target), rotation_error,
                     .002 * (q - original_q)]

    fit = least_squares(residual, original_q,
                        bounds=(np.array([j.range[0] for j in joints]),
                                np.array([j.range[1] for j in joints])),
                        max_nfev=250, gtol=1e-10, ftol=1e-10, xtol=1e-10)
    residual(fit.x)
    position_error = float(np.linalg.norm(d.cam_xpos[cid] - target))
    assert fit.success and position_error < 1e-5
    nonrobot = [i for i in range(m.nbody) if not m.body(i).name.startswith('robot_0/')]
    assert np.array_equal(d.xpos[nonrobot], original_body_positions[nonrobot])
    contacts = []
    for contact in d.contact:
        names = [m.body(int(m.geom_bodyid[g])).name for g in contact.geom]
        if (any(n.startswith('robot_0/') for n in names)
                and not all(n.startswith('robot_0/') for n in names)):
            contacts.append(dict(bodies=names, distance_m=float(contact.dist)))
    assert not any(c['distance_m'] < -.001 for c in contacts)

    # Preserve the original table camera and renderer's headlight convention.
    m.vis.global_.fovy = np.rad2deg(2 * np.arctan(table_k[1, 2] / table_k[1, 1]))
    table_camera = mujoco.MjvCamera()
    table_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    original_groups = m.geom_group.copy()
    with mujoco.Renderer(m, 1408, 2496) as renderer:
        for hidden, name in [(False, 'table_with_robot'), (True, 'table_without_robot')]:
            m.geom_group[:] = original_groups
            if hidden:
                m.geom_group[robot_ids] = 5
            renderer.update_scene(d, camera=table_camera, scene_option=table_opt)
            for glcam in renderer.scene.camera:
                glcam.pos = table_pose[:3, 3]
                glcam.forward = table_pose[:3, 2]
                glcam.up = -table_pose[:3, 1]
            Image.fromarray(renderer.render().copy()).save(OUT / f'{name}.png')
    m.geom_group[:] = original_groups

    sensor_opt = mujoco.MjvOption()
    sensor_opt.geomgroup[2] = 0
    sensor_opt.geomgroup[4] = 1
    sensor_opt.sitegroup[:] = 0
    records = []
    with mujoco.Renderer(m, 1024, 1024) as renderer, mujoco.Renderer(m, 8, 8) as native:
        for i in range(6):
            name = f'link6_sensor_{i}'
            out = OUT / name
            out.mkdir(exist_ok=True)
            camera = 'robot_0/' + name
            renderer.update_scene(d, camera=camera, scene_option=sensor_opt)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            native.update_scene(d, camera=camera, scene_option=sensor_opt)
            native.enable_depth_rendering()
            readout = native.render().copy()
            native.disable_depth_rendering()
            # Segmentation identifies the actual surfaces hit by the 64 rays.
            native.enable_segmentation_rendering()
            segmentation = native.render().copy()
            native.disable_segmentation_rendering()
            hit_counts = {}
            for geom_id, object_type in segmentation.reshape(-1, 2):
                if object_type == int(mujoco.mjtObj.mjOBJ_GEOM) and geom_id >= 0:
                    body_name = m.body(int(m.geom_bodyid[geom_id])).name
                    hit_counts[body_name] = hit_counts.get(body_name, 0) + 1
            assert readout.shape == (8, 8) and np.all(np.isfinite(readout))
            assert np.all(readout > 0)
            Image.fromarray(rgb).save(out / 'rgb.png')
            Image.fromarray(depth_rgb(depth)).save(out / 'depth.png')
            small = Image.fromarray(depth_rgb(readout))
            small.save(out / 'readout_8x8_native.png')
            bordered_sensor_grid(small).save(out / 'readout_8x8.png')
            np.save(out / 'depth_render_metres.npy', depth)
            np.save(out / 'readout_8x8_metres.npy', readout)
            np.savetxt(out / 'readout_8x8_metres.csv', readout, delimiter=',', fmt='%.9g')
            sheet = Image.new('RGB', (3096, 1024), 'white')
            for j, filename in enumerate(['rgb.png', 'depth.png', 'readout_8x8.png']):
                with Image.open(out / filename) as panel:
                    sheet.paste(panel, (j * 1036, 0))
            sheet.save(out / 'rgb_depth_8x8.png')
            sensor_id = m.camera(camera).id
            records.append(dict(sensor=name, sensor_position_world=d.cam_xpos[sensor_id].tolist(),
                                sensor_rotation_gl=d.cam_xmat[sensor_id].reshape(3, 3).tolist(),
                                fovy_degrees=float(m.cam_fovy[sensor_id]),
                                minimum_depth_m=float(readout.min()), maximum_depth_m=float(readout.max()),
                                native_8x8_surface_hit_counts=hit_counts))
            print(name, hit_counts, flush=True)

    manifest = dict(
        source_dataset=str(ROW), source_scene_frame=0,
        provenance='New deterministic simulation renders of the original dataset geometry with an adjusted robot pose; not recorded experiment frames.',
        adjustment='Link 6 lowered 0.08 m through seven arm joint angles; sensor orientation preserved by IK. Robot base, scene objects and table camera unchanged.',
        original_arm_qpos=original_q.tolist(), adjusted_arm_qpos=fit.x.tolist(),
        original_sensor4_position_world=original_position.tolist(),
        adjusted_sensor4_position_world=d.cam_xpos[cid].tolist(), ik_position_error_m=position_error,
        robot_environment_contacts=contacts, recommended_sensor='link6_sensor_4',
        rgb_provenance='Diagnostic RGB render from the sensor viewpoint; the proximity sensor itself does not output RGB.',
        depth_provenance='Dense 1024x1024 simulated axial depth, measured in metres.',
        readout_provenance='Fresh native 8x8 simulated axial depth at the same adjusted pose, not downsampled dense depth and not reused HDF5 measurements.',
        geometry='Original geometry; proximity rendering hides cosmetic skin and includes sensor-only collision geometry, including original green collision surfaces.',
        display_range_metres=[NEAR, FAR], colour_map='turbo: near red, far blue; values outside the display range saturate, numerical arrays remain unclipped.',
        qualification='A static illustrative pose, not a policy rollout or a demonstration of collision-avoidance performance. Surface-hit counts do not alone establish table-camera occlusion.',
        grid_display='1024x1024 nearest-neighbor display with solid black 4 px cell boundaries and outer border. Native 8x8 PNG and numerical arrays unchanged.',
        processing='No labels, generative edits, object movement or physics stepping.', records=records)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (OUT / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Lowered arm sensor views</title>'
        '<style>body{font:16px Arial;max-width:1500px;margin:30px auto}img{width:100%}</style>'
        '<h1>Arm lowered 8 cm</h1><p>Adjusted simulation pose using original scene geometry. '
        'Each sensor row: RGB viewpoint / dense depth / native 8×8 simulated depth. '
        'Near red, far blue; 0.05–1.0 m. Exported images contain no text.</p>'
        '<h2>Table camera</h2><img src="table_with_robot.png"><img src="table_without_robot.png">'
        + ''.join(f'<h2>Sensor {i}</h2><img src="link6_sensor_{i}/rgb_depth_8x8.png">'
                  for i in [4, 3, 0, 1, 2, 5]))
    with zipfile.ZipFile(OUT / 'lowered_robot_images.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix != '.zip':
                archive.write(path, path.relative_to(OUT))
    print('Saved', OUT, flush=True)


if __name__ == '__main__':
    main()
