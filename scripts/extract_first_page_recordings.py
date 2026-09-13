#!/usr/bin/env python3
"""Extract paper panels from recordings; never launch a simulator or image model."""
from pathlib import Path
import hashlib
import json
import cv2
import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'images/first_page/recorded_assets'
ROW = Path('/mnt/laptop/data/pact_pick_n_place_v2/data/v1011d/rows/000_187ba0ce76cb3011')
FRAME = 150


def frame(path, index):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'Cannot decode {path}, frame {index}')
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = []
    for cam in ('exo_camera_1', 'wrist_camera'):
        src = ROW / f'episode_00000000_{cam}.mp4'
        im = frame(src, FRAME)
        im.save(OUT / f'{cam}.png')
        sources.append(dict(path=str(src), sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                            frame_index=FRAME, output=f'{cam}.png', size=list(im.size)))
    # Same timestamp and calibration for every range sample. Use recorded sensor
    # intrinsics, not a visually chosen point pattern. Pool the four stored samples
    # by their mean, matching dataset_viz's default. Preserve the complete 8x8
    # patch, including ranges omitted by the earlier 60 cm display cutoff.
    points, distances, sensor_ids, pixels, origins = [], [], [], [], []
    with h5py.File(ROW / 'trajectory.h5', 'r') as f:
        g = f['traj_0']
        names = list(g['obs/proximity'])
        heat = []
        for name in names:
            raw = g[f'obs/proximity/{name}'][FRAME]
            depth = raw.mean(axis=0) if raw.ndim == 3 else raw
            heat.append(depth)
            prm = g[f'obs/sensor_param/{name}']
            k = prm['intrinsic_cv'][FRAME]
            c2w = prm['cam2world_gl'][FRAME]
            ext = prm['extrinsic_cv'][FRAME]
            # The recorded matrix maps CV coordinates despite its GL name.
            assert np.allclose(ext @ c2w, np.eye(4)[:3], atol=1e-5)
            v, u = np.indices(depth.shape)
            assert depth.shape == (8, 8)
            mask = np.isfinite(depth) & (depth >= .02)
            z = depth[mask]
            xyz = np.stack([(u[mask]-k[0,2])*z/k[0,0],
                            (v[mask]-k[1,2])*z/k[1,1], z, np.ones_like(z)], axis=1)
            points.extend((c2w @ xyz.T).T[:, :3])
            distances.extend(z.tolist())
            sensor_ids.extend([name] * len(z))
            pixels.extend(np.stack([u[mask], v[mask]], axis=1).tolist())
            origins.extend([c2w[:3,3].tolist()] * len(z))
        points = np.asarray(points)
        projected = {}
        for cam in ('exo_camera_1', 'wrist_camera'):
            prm = g[f'obs/sensor_param/{cam}']
            ext = prm['extrinsic_cv'][FRAME]
            k = prm['intrinsic_cv'][FRAME].copy()
            # MP4s preserve the square camera's vertical FOV at 624 x 352.
            k[0,0] *= 352 / (2*k[1,2])
            k[1,1] *= 352 / (2*k[1,2])
            k[0,2], k[1,2] = 312, 176
            pc = points @ ext[:,:3].T + ext[:,3]
            homogeneous = pc @ k.T
            uv = homogeneous[:,:2] / homogeneous[:,2:]
            origin_pc = np.array(origins) @ ext[:,:3].T + ext[:,3]
            origin_h = origin_pc @ k.T
            origin_uv = origin_h[:,:2] / origin_h[:,2:]
            projected[cam] = [dict(u=float(p[0]), v=float(p[1]), z=float(z),
                                   distance_m=float(d), sensor=s, pixel=ij,
                                   origin_u=float(o[0]), origin_v=float(o[1]), origin_z=float(oz))
                              for p,z,d,s,ij,o,oz in zip(uv,pc[:,2],distances,sensor_ids,pixels,origin_uv,origin_pc[:,2])
                              if z > .02]
        scene = json.loads(g['obs_scene'][()])
        np.savez_compressed(OUT / 'recorded_ranges.npz', depth_m=np.array(heat),
                            sensor_names=names, world_points=points,
                            distance_m=distances, pixel_uv=pixels, sensor_ids=sensor_ids,
                            sensor_origins=origins)
        # Standard 3D point-cloud file, retaining all 64 samples from every sensor.
        header = ['ply', 'format ascii 1.0', f'element vertex {len(points)}',
                  'property float x', 'property float y', 'property float z',
                  'property float range_m', 'property int sensor_id',
                  'property uchar pixel_u', 'property uchar pixel_v', 'end_header']
        rows = [f'{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {d:.7f} {names.index(s)} {ij[0]} {ij[1]}'
                for p,d,s,ij in zip(points, distances, sensor_ids, pixels)]
        (OUT/'pointcloud_40x8x8.ply').write_text('\n'.join(header+rows)+'\n')
        (OUT / 'projected_returns.json').write_text(json.dumps(projected, indent=2))
        # A text-free, un-smoothed 40 x 8 x 8 sensor contact sheet for reuse.
        from matplotlib import colormaps
        mosaic = Image.new('RGB', (10*64+9*5, 4*64+3*5), 'white')
        for i, depth in enumerate(heat):
            rgb = (colormaps['viridis'](np.clip((depth-.02)/.58, 0, 1))[:,:,:3]*255).astype('uint8')
            rgb[(depth < .02) | (depth > .60) | ~np.isfinite(depth)] = [239,241,242]
            mosaic.paste(Image.fromarray(rgb).resize((64,64), Image.Resampling.NEAREST),
                         ((i%10)*69, (i//10)*69))
        mosaic.save(OUT / 'skin_ranges_40_sensors.png')
        (OUT / 'source_manifest.json').write_text(json.dumps(dict(
            dataset='pact_pick_n_place_v2 / v1011d', row=ROW.name,
            recording_type='Recorded simulated scripted demonstration',
            trajectory='traj_0', frame_index=FRAME, elapsed_seconds=FRAME*.066,
            rgb_sources=sources, hdf5=str(ROW/'trajectory.h5'),
            hdf5_sha256=hashlib.sha256((ROW/'trajectory.h5').read_bytes()).hexdigest(),
            range_processing='Mean over four stored samples; finite ranges >= 0.02 m; all 64 pixels per sensor retained; no upper display cutoff',
            range_min_m=float(min(distances)), range_max_m=float(max(distances)),
            per_sensor_sample_counts={n:sensor_ids.count(n) for n in names},
            projection='CV calibration; MP4 height/FOV correction to 624 x 352; no RGB depth occlusion culling',
            measurement_count=len(points), projected_counts={k:len(v) for k,v in projected.items()},
            task=scene['task_description'], success=bool(g['success'][-1]),
            contact_audit=scene.get('pact_contact_audit', {})), indent=2))
    for t, name in [(0, 'task_start'), (250, 'task_transfer'), (380, 'task_place')]:
        frame(ROW/'episode_00000000_exo_camera_1.mp4', t).save(OUT/f'{name}.png')
    print(f'Extracted native recording panels and {len(points)} measured returns to {OUT}')


if __name__ == '__main__':
    main()
