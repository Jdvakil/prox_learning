#!/usr/bin/env python3
"""Clean matched table-camera renders from a recorded episode's initial state.

No image generation or inpainting. Restore source MJCF meshes, saved object
poses, robot joints and camera calibration. Hide robot geoms for the second
render without stepping physics or changing objects, lighting or camera.
"""
from pathlib import Path
import os, sys, json, pickle, base64, hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'submodules/molmospaces'))
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('MPLCONFIGDIR','/tmp/prox-mpl')
import h5py
import numpy as np
import mujoco
import cv2
from PIL import Image
from molmo_spaces.robots.franka import FrankaRobot
from molmo_spaces.utils.lazy_loading_utils import locate_uid_package

ROW=Path('/mnt/laptop/data/pact_pick_n_place_v2/data/v1011d/rows/000_187ba0ce76cb3011')
OUT=ROOT/'images/table_occlusion'


def main(return_scene=False, row=ROW, scene_path=None, kitchen=False):
    OUT.mkdir(parents=True,exist_ok=True)
    with h5py.File(row/'trajectory.h5','r') as f:
        g=f['traj_0'];meta=json.loads(g['obs_scene'][()])
        saved=pickle.loads(base64.b64decode(meta['frozen_config']))
        c2w=g['obs/sensor_param/exo_camera_1/cam2world_gl'][0]
        k=g['obs/sensor_param/exo_camera_1/intrinsic_cv'][0]
        q=g['env_states/articulations/panda'][0,:9]
        sensor_pose=g['obs/sensor_param/link6_sensor_0/cam2world_gl'][0]
        sensor_points=[]
        for i in range(6):
            name=f'link6_sensor_{i}';depth=g[f'obs/proximity/{name}'][0,-1]
            prm=g[f'obs/sensor_param/{name}'];sk=prm['intrinsic_cv'][0];sp=prm['cam2world_gl'][0]
            v,u=np.indices((8,8))
            pc=np.stack([(u+.5-sk[0,2])*depth/sk[0,0],(v+.5-sk[1,2])*depth/sk[1,1],depth,np.ones_like(depth)],-1).reshape(64,4)
            sensor_points.extend((sp@pc.T).T[:,:3])
    scene=scene_path or ROOT/'submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v10_7_neg5.xml'
    spec=mujoco.MjSpec.from_file(str(scene))
    params=meta['scene_params']
    # The frozen clutter poses already contain the settled object-root offsets.
    poses=saved.task_config.object_poses
    palette={x['body_prefix']+x['uid']:x for x in params['pact_clutter_palette']}
    object_sources=[]
    for name,pose in poses.items():
        if pose[2] <= -1:
            continue  # recorded inactive props are parked outside the room
        item=palette.get(name,{})
        primitive=item.get('primitive')
        if primitive:
            body=spec.worldbody.add_body(name=name,pos=pose[:3],quat=pose[3:])
            if primitive['shape']=='cylinder':
                kind=mujoco.mjtGeom.mjGEOM_CYLINDER
                size=[primitive['radius_m'],primitive['height_m']/2,0]
            else:
                kind=mujoco.mjtGeom.mjGEOM_BOX
                size=(np.array(primitive['size_m'])/2).tolist()
            body.add_geom(name=name+'_visual',type=kind,size=size,rgba=primitive['rgba'])
            object_sources.append(dict(name=name,source='recorded primitive definition',pose=pose))
        else:
            uid=name.split('/')[-1]
            _,_,path=locate_uid_package(uid)
            if not path or not Path(path).is_file():raise FileNotFoundError(f'Missing original asset: {uid}')
            obj=mujoco.MjSpec.from_file(str(path));root=obj.worldbody.bodies[0]
            # The frozen root pose is applied by the attachment frame.
            root.pos=[0,0,0];root.quat=[1,0,0,0]
            for j in list(obj.joints):
                if j.type==mujoco.mjtJoint.mjJNT_FREE:obj.delete(j)
            frame=spec.worldbody.add_frame(pos=pose[:3],quat=pose[3:])
            frame.attach_body(root,name.rsplit('/',1)[0]+'/', '')
            object_sources.append(dict(name=name,source=str(path),pose=pose))
    base=saved.task_config.robot_base_pose
    robot=mujoco.MjSpec.from_file(str(ROOT/'assets/robots/franka_skin/model_hybrid.xml'))
    FrankaRobot.add_robot_to_scene(saved.robot_config,spec,robot,'robot_0/',base[:3],base[3:],False)
    cam_rot=c2w[:3,:3]@np.diag([1,-1,-1])
    quat=np.zeros(4);mujoco.mju_mat2Quat(quat,cam_rot.ravel())
    fovy=np.rad2deg(2*np.arctan(k[1,2]/k[1,1]))
    spec.worldbody.add_camera(name='recorded_table',pos=c2w[:3,3],quat=quat,fovy=fovy)
    spec.visual.global_.offwidth=2496;spec.visual.global_.offheight=1408
    if kitchen:
        import render_pact_place_v12_clutter as kitchen_overlay
        # All assets are already installed. Do not trigger downloads.
        def installed_asset(uid):
            _, _, path = locate_uid_package(uid)
            if not path or not Path(path).is_file():
                raise FileNotFoundError(uid)
            return Path(path)
        kitchen_overlay._install_uid = installed_asset
        kitchen_names = kitchen_overlay._attach_standing_kitchen(spec)
    model=spec.compile();data=mujoco.MjData(model)
    def mocap(name,pos):
        b=model.body(name);data.mocap_pos[b.mocapid[0]]=pos
    mocap('sash',[.58,0,.72+params['ap_h']+.025])
    mocap('jamb_l',[.58,params['ap_w']/2+.18,.92])
    mocap('jamb_r',[.58,-params['ap_w']/2-.18,.92])
    for name in ['pact_clutter_l0','pact_clutter_l1','pact_clutter_r0','pact_clutter_r1']:
        mocap(name,[0,0,-2])
    if params.get('pact_v9_legacy_panel_active'):
        mocap(params['protr_name'],params['protr_center'])
    for i in range(7):data.qpos[model.joint(f'robot_0/fr3_joint{i+1}').qposadr[0]]=q[i]
    for i,side in enumerate(['left','right']):
        for joint in ['driver','spring_link','follower']:
            suffix='follower' if side=='left' and joint=='follower' else f'{joint}_joint'
            j=model.joint(f'robot_0/gripper/{side}_{suffix}')
            data.qpos[j.qposadr[0]]=q[7+i] if joint!='follower' else -q[7+i]
    mujoco.mj_forward(model,data)
    if kitchen:
        kitchen_overlay._hide_primitive_colliders(model)
        placed = kitchen_overlay._place_standing_kitchen(model, data, kitchen_names)
        meta['figure_kitchen_overlay'] = dict(
            provenance='Reconstructed with the existing v12 kitchen-placement routine; these auxiliary poses are absent from frozen_config.',
            placed_bodies=placed)
        mujoco.mj_forward(model,data)
    cid=model.camera('robot_0/link6_sensor_0').id
    pose_error=float(np.linalg.norm(data.cam_xpos[cid]-sensor_pose[:3,3]))
    print('Recorded link6 sensor origin error (m):',pose_error,flush=True)
    if pose_error>1e-4:raise ValueError('Robot reconstruction does not match recorded link-6 pose')
    opt=mujoco.MjvOption();opt.geomgroup[3:]=0;opt.sitegroup[:]=0
    ids=[i for i in range(model.ngeom) if model.body(int(model.geom_bodyid[i])).name.startswith('robot_0/')
         and model.body(int(model.geom_bodyid[i])).name!='robot_0/base']
    before=data.qpos.copy();poses_before=data.xpos.copy()
    original_groups=model.geom_group.copy()
    if return_scene:
        return model,data,c2w,k,opt,ids,meta
    # Match CPUMujocoEnv._render_frame exactly: it updates a free-camera scene
    # and then sets the GL camera vectors. This also preserves the recording's
    # headlight behavior, which differs from rendering via a fixed MJCF camera.
    model.vis.global_.fovy=fovy
    recorded_cam=mujoco.MjvCamera();recorded_cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    depth_checks={}
    for width,height,label in [(624,352,'native'),(2496,1408,'highres')]:
        with mujoco.Renderer(model,height=height,width=width) as renderer:
            for hidden,stem in [(False,'table_with_robot'),(True,'table_without_robot')]:
                model.geom_group[:]=original_groups
                if hidden:model.geom_group[ids]=5
                renderer.update_scene(data,camera=recorded_cam,scene_option=opt)
                for glcam in renderer.scene.camera:
                    glcam.pos=c2w[:3,3]
                    glcam.forward=c2w[:3,2]
                    glcam.up=-c2w[:3,1]
                Image.fromarray(renderer.render().copy()).save(OUT/f'{stem}_{label}.png')
                if label=='native':
                    renderer.enable_depth_rendering()
                    depth_checks[stem]=renderer.render().copy()
                    renderer.disable_depth_rendering()
        model.geom_group[:]=original_groups
    assert np.array_equal(before,data.qpos) and np.array_equal(poses_before,data.xpos)
    cap=cv2.VideoCapture(str(ROW/'episode_00000000_exo_camera_1.mp4'))
    ok,frame=cap.read();cap.release();assert ok
    Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).save(OUT/'table_original_recording_frame000.png')
    original=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).astype(float)
    restored=np.array(Image.open(OUT/'table_with_robot_native.png')).astype(float)
    error=np.abs(original-restored)
    camera_points=(np.array(sensor_points)-c2w[:3,3])@c2w[:3,:3]
    fk=k[0,0]*352/(2*k[1,2])
    uv=np.stack([fk*camera_points[:,0]/camera_points[:,2]+312,
                 fk*camera_points[:,1]/camera_points[:,2]+176],1)
    blocked=0
    for (u,v),z in zip(uv,camera_points[:,2]):
        x,y=int(np.floor(u)),int(np.floor(v))
        if z<=0 or not (0<=x<624 and 0<=y<352):continue
        near=depth_checks['table_with_robot'][y,x]
        revealed=depth_checks['table_without_robot'][y,x]
        if z>near+.02 and abs(z-revealed)<.025 and revealed>near+.02:blocked+=1
    manifest=dict(source_hdf5=str(ROW/'trajectory.h5'),frame=0,scene=str(scene),
        hdf5_sha256=hashlib.sha256((ROW/'trajectory.h5').read_bytes()).hexdigest(),
        camera='exo_camera_1',camera_pose=c2w.tolist(),fovy=float(fovy),objects=object_sources,
        link6_origin_error_m=pose_error,hidden_robot_geoms=len(ids),
        original_rgb_mean_absolute_error=float(error.mean()),
        original_pixels_within_10_levels_percent=float(100*(error.max(2)<10).mean()),
        link6_returns_hidden_by_robot_depth_check=blocked,
        visibility_check='384 recorded link6 samples at frame 0; samples project behind robot-on depth by >2 cm and agree with robot-off depth within 2.5 cm. Approximate visibility check, not object identification.',
        gripper_replay='Recorded driver joints; passive spring/follower joints restored with mechanical coupling because their independent states are not stored.',
        exports='Matched deterministic simulator re-renders; the original decoded recording is supplied separately.',
        changes_between_pair='Only robot geometry visibility; pedestal retained. No physics step, object pose, light or camera change.',
        image_processing='None: no generative model, inpainting, labels, markers, arrows or compositing.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Saved clean matched camera pair:',OUT,flush=True)
    print('Recorded link6 returns behind robot and matching revealed surface:',blocked,flush=True)


if __name__=='__main__':main()
