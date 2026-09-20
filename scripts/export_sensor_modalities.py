#!/usr/bin/env python3
"""Export clean, pose-matched RGB/depth diagnostic renders and raw 8x8 data.

RGB and dense depth are deterministic simulator views, NOT recorded high-res
measurements. The 8x8 panel contains the original HDF5 values without smoothing.
"""
from pathlib import Path
import json, csv, zipfile
from export_table_occlusion_pair import main as restore_scene, ROW, ROOT
import mujoco
import h5py
import numpy as np
from PIL import Image
from matplotlib import colormaps
from sensor_grid_image import bordered_sensor_grid

OUT=ROOT/'images/table_occlusion/sensor_views'
NEAR,FAR=.05,1.0


def depth_rgb(depth):
    a=np.clip((np.asarray(depth)-NEAR)/(FAR-NEAR),0,1)
    return (colormaps['turbo'](1-a)[...,:3]*255).astype(np.uint8)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    m,d,table_pose,table_k,table_opt,robot_ids,scene=restore_scene(return_scene=True)
    opt=mujoco.MjvOption();opt.geomgroup[2]=0;opt.geomgroup[4]=1;opt.sitegroup[:]=0
    records=[]
    with h5py.File(ROW/'trajectory.h5') as h,mujoco.Renderer(m,1024,1024) as renderer,mujoco.Renderer(m,8,8) as native:
        for i in range(6):
            name=f'link6_sensor_{i}';out=OUT/name;out.mkdir(exist_ok=True)
            camera='robot_0/'+name;cid=m.camera(camera).id
            raw=h[f'traj_0/obs/proximity/{name}'][0,-1].copy()
            pose=h[f'traj_0/obs/sensor_param/{name}/cam2world_gl'][0]
            k=h[f'traj_0/obs/sensor_param/{name}/intrinsic_cv'][0]
            pos_err=float(np.max(abs(d.cam_xpos[cid]-pose[:3,3])))
            rot_err=float(np.max(abs(d.cam_xmat[cid].reshape(3,3)@np.diag([1,-1,-1])-pose[:3,:3])))
            assert pos_err<1e-6 and rot_err<1e-6
            renderer.update_scene(d,camera=camera,scene_option=opt)
            rgb=renderer.render().copy()
            renderer.enable_depth_rendering();depth=renderer.render().copy();renderer.disable_depth_rendering()
            native.update_scene(d,camera=camera,scene_option=opt)
            native.enable_depth_rendering();replayed=native.render().copy();native.disable_depth_rendering()
            Image.fromarray(rgb).save(out/'rgb.png')
            Image.fromarray(depth_rgb(depth)).save(out/'depth.png')
            small=Image.fromarray(depth_rgb(raw));small.save(out/'readout_8x8_native.png')
            bordered_sensor_grid(small).save(out/'readout_8x8.png')
            np.save(out/'depth_render_metres.npy',depth)
            np.save(out/'readout_8x8_metres.npy',raw)
            np.savetxt(out/'readout_8x8_metres.csv',raw,delimiter=',',fmt='%.9g')
            sheet=Image.new('RGB',(3096,1024),'white')
            for j,filename in enumerate(['rgb.png','depth.png','readout_8x8.png']):
                sheet.paste(Image.open(out/filename),(j*1036,0))
            sheet.save(out/'rgb_depth_8x8.png')
            records.append(dict(sensor=name,frame=0,substep=3,range_min_m=float(raw.min()),range_max_m=float(raw.max()),
                sensor_pose_cv=pose.tolist(),intrinsics_8x8=k.tolist(),
                restored_pose_max_error_m=pos_err,restored_rotation_max_error=rot_err,
                native_replay_depth_mae_m=float(np.mean(abs(replayed-raw))),
                native_replay_depth_max_error_m=float(np.max(abs(replayed-raw)))))
            print(name,'saved RGB / dense depth / recorded 8x8',flush=True)
    manifest=dict(source_hdf5=str(ROW/'trajectory.h5'),trajectory='traj_0',frame=0,substep=3,
        recommended_sensor='link6_sensor_4',records=records,depth_colour_map='turbo, near=red, far=blue',
        display_range_metres=[NEAR,FAR],range_display_clipping='Values outside colour range saturate; numerical files retain original values.',
        rgb_provenance='1024x1024 simulator diagnostic view from the original sensor pose and 45-degree FOV. The SPAD sensor does not output RGB.',
        depth_provenance='1024x1024 simulator depth diagnostic from the identical camera pose; not a high-resolution recorded sensor measurement.',
        readout_provenance='Unmodified HDF5 array traj_0/obs/proximity/{sensor}[0,3,:,:]; 8x8 PNG enlarged by nearest-neighbor only.',
        geometry='Original saved-scene reconstruction used for the table-camera pair; cosmetic skin hidden and sensor-only geometry enabled, matching proximity renderer settings.',
        qualification='The dense re-render and raw 8x8 values differ at some pixels; numerical replay errors are recorded per sensor. These views show sensor field of view and recorded range structure, not a validated claim that all displayed objects are occluded in table RGB.',
        grid_display='1024x1024 nearest-neighbor display with solid black 4 px cell boundaries and outer border. Native 8x8 PNG and numerical arrays unchanged.',
        processing='No generated images, inpainting, arrows, labels or invented range samples.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Sensor RGB, depth and 8×8</title><style>body{font:16px Arial;max-width:1500px;margin:30px auto}img{width:100%}p{line-height:1.6}</style><h1>Link 6: RGB / depth / recorded 8×8</h1><p>Frame 0, same scene as the table-camera pair. Left: RGB diagnostic render. Middle: dense depth diagnostic render. Right: recorded 8×8 sensor data. Display scale: 0.05–1.0 m, near red and far blue. PNGs contain no labels.</p>'+''.join(f'<h2>link6_sensor_{i}</h2><img src="link6_sensor_{i}/rgb_depth_8x8.png">' for i in [4,0,1,2,3,5]))
    with zipfile.ZipFile(OUT/'sensor_rgb_depth_8x8.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.write(OUT/'manifest.json','manifest.json')
        for p in sorted(OUT.glob('link6_sensor_*/*')):
            z.write(p,p.relative_to(OUT))


if __name__=='__main__':main()
