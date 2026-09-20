#!/usr/bin/env python3
"""Exact 8x8 axial-depth backprojection. No random points or fitted surfaces.

Also exports the original stored link-6 readings directly from a dataset HDF5.
"""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np


def backproject(depth, intrinsic_cv, camera_to_world_cv):
    """One XYZ per input pixel. depth is axial Z in metres, not ray length.

    Pixel centers are u+0.5, v+0.5. The camera frame is +X right, +Y down,
    +Z forward. Output has shape (height, width, 3); nothing is resampled.
    """
    depth = np.asarray(depth, dtype=np.float64)
    assert depth.ndim == 2
    k = np.asarray(intrinsic_cv, dtype=np.float64)
    pose = np.asarray(camera_to_world_cv, dtype=np.float64)
    v, u = np.indices(depth.shape)
    pixels = np.stack([u+.5, v+.5, np.ones_like(u)], axis=-1)
    rays = pixels @ np.linalg.inv(k).T
    local = rays * (depth/rays[..., 2])[..., None]
    return local @ pose[:3, :3].T + pose[:3, 3]


def native_link6(model, data, renderer, segmentation=False):
    """Use the project's native 8x8 proximity-camera rendering settings."""
    import mujoco
    opt = mujoco.MjvOption(); opt.geomgroup[2] = 0; opt.geomgroup[4] = 1
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    names = sorted((model.camera(i).name for i in range(model.ncam)
                    if re.fullmatch(r'robot_0/link6_sensor_\d+', model.camera(i).name)),
                   key=lambda name: int(name.rsplit('_', 1)[1]))
    if not names:
        raise ValueError('No robot_0/link6_sensor_* cameras found')
    depths, poses, intrinsics, points, ids = [], [], [], [], []
    for name in names:
        cid = model.camera(name).id
        renderer.update_scene(data, camera=cid, scene_option=opt)
        renderer.enable_depth_rendering()
        z = renderer.render().copy()
        renderer.disable_depth_rendering()
        assert z.shape == (8, 8)
        f = 4/np.tan(np.deg2rad(model.cam_fovy[cid])/2)
        k = np.array([[f, 0, 4], [0, f, 4], [0, 0, 1.]])
        pose = np.eye(4)
        # MuJoCo camera local axes are GL (+X right,+Y up,-Z forward).
        pose[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])
        pose[:3, 3] = data.cam_xpos[cid]
        depths.append(z); poses.append(pose); intrinsics.append(k)
        points.append(backproject(z, k, pose))
        if segmentation:
            renderer.enable_segmentation_rendering()
            seg = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            ids.append(np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1))
    result = dict(sensor_names=np.array(names), depth_m=np.array(depths),
                  intrinsic_cv=np.array(intrinsics), camera_to_world_cv=np.array(poses),
                  points_world_m=np.array(points))
    if segmentation: result['geom_ids'] = np.array(ids)
    return result


def recorded_link6(path, frame=0, substep=-1):
    import h5py
    depths, poses, intrinsics, points = [], [], [], []
    with h5py.File(path, 'r') as f:
        obs = f['traj_0/obs']
        names = sorted((name for name in obs['proximity'] if re.fullmatch(r'link6_sensor_\d+', name)),
                       key=lambda name: int(name.rsplit('_', 1)[1]))
        for name in names:
            z = obs[f'proximity/{name}'][frame, substep].copy()
            k = obs[f'sensor_param/{name}/intrinsic_cv'][frame].copy()
            # Despite this legacy key's suffix, the stored transform uses CV
            # axes, as confirmed by the dataset extrinsic/intrinsic matrices.
            pose = obs[f'sensor_param/{name}/cam2world_gl'][frame].copy()
            assert z.shape == (8, 8)
            depths.append(z); intrinsics.append(k); poses.append(pose)
            points.append(backproject(z, k, pose))
    return dict(sensor_names=np.array(names), depth_m=np.array(depths),
                intrinsic_cv=np.array(intrinsics), camera_to_world_cv=np.array(poses),
                points_world_m=np.array(points))


def save_cloud(folder, cloud):
    """Preserve every native sample, with exact sensor/pixel provenance."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'measurements.npz', **cloud)
    points = cloud['points_world_m'].reshape(-1, 3)
    with (folder/'points.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['sensor', 'pixel_u', 'pixel_v', 'axial_depth_m', 'world_x_m', 'world_y_m', 'world_z_m'])
        for s, name in enumerate(cloud['sensor_names']):
            for v in range(8):
                for u in range(8):
                    writer.writerow([name, u, v, cloud['depth_m'][s, v, u], *cloud['points_world_m'][s, v, u]])
    assert np.all(np.isfinite(points))
    header = f'ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n'
    with (folder/'points.ply').open('w') as f:
        f.write(header); np.savetxt(f, points, fmt='%.12g')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdf5', required=True, type=Path)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--substep', type=int, default=-1)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    cloud = recorded_link6(args.hdf5, args.frame, args.substep)
    save_cloud(args.out, cloud)
    (args.out/'source.json').write_text(json.dumps(dict(hdf5=str(args.hdf5), frame=args.frame,
        substep=args.substep, samples=int(cloud['depth_m'].size),
        note='Original recorded simulated depths, with recorded calibration. No interpolation, averaging, ray jitter or synthetic point filling.'), indent=2))
