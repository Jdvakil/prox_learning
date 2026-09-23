#!/usr/bin/env python3
"""MuJoCo setup figure: skin sensor locations and 40 native 8x8 projections."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse,json
from pathlib import Path
import numpy as np
import mujoco as mj
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'images/skin_projection_setup'
ROW=Path('/mnt/laptop/data/pact_pick_n_place_v2/data/v12/rows/000_2f77fef1863bdeb3')

def build():
    from export_table_occlusion_pair import main as restore
    m,d,_,_,_,_,meta=restore(return_scene=True,row=ROW,scene_path=ROOT/'custom_scenes/pact_place_corridor_v12.xml',kitchen=True)
    # A recorded task pose, with the actual cup and collection clutter.
    frame=0;rec=meta['trajectory'][frame]
    adjustment=json.loads((ROOT/'images/whole_body/kitchen_items/manifest.json').read_text())['provenance']['adjusted_simulation_pose']
    d.qpos[:13]=rec['qpos'][:13]
    d.qpos[:7]=adjustment['adjusted_arm_qpos']
    cup=m.body('cavity_obj_0/Cup_10').id
    m.body_pos[cup]=rec['object_position_m'];m.body_quat[cup]=rec['object_quat_xyzw']
    mj.mj_forward(m,d)
    m.vis.global_.offwidth=2600;m.vis.global_.offheight=1800;m.vis.quality.offsamples=4
    mj.mj_saveModel(m,str(OUT/'model.mjb'))
    np.savez_compressed(OUT/'state.npz',qpos=d.qpos,mocap_pos=d.mocap_pos,mocap_quat=d.mocap_quat)
    (OUT/'source.json').write_text(json.dumps(dict(source_row=str(ROW),source_frame=frame,pose_adjustment=adjustment,task=meta['task_description'],scene='custom_scenes/pact_place_corridor_v12.xml',overlay=meta.get('figure_kitchen_overlay'),note='Fresh sensor measurements from restored dataset geometry, frame 0 cup pose and an existing sensor visualization arm pose (link 6 lowered 16 cm); auxiliary kitchen poses rebuilt by existing collection loader.'),indent=2))
    return m,d

def load():
    m=mj.MjModel.from_binary_path(str(OUT/'model.mjb'));d=mj.MjData(m)
    with np.load(OUT/'state.npz') as z:
        for k in ['qpos','mocap_pos','mocap_quat']:getattr(d,k)[:]=z[k]
    mj.mj_forward(m,d)
    return m,d

def measure(m,d):
    names=sorted(m.camera(i).name for i in range(m.ncam) if '_sensor_' in m.camera(i).name)
    assert len(names)==40
    opt=mj.MjvOption();opt.geomgroup[2]=0;opt.geomgroup[4]=1;opt.sitegroup[:]=0
    origins=[];points=[];depths=[];gids=[];raster=[];directions=[]
    with mj.Renderer(m,8,8) as r:
        for name in names:
            ci=m.camera(name).id;origin=d.cam_xpos[ci].copy();rotation=d.cam_xmat[ci].reshape(3,3)
            f=4/np.tan(np.deg2rad(m.cam_fovy[ci])/2)
            v,u=np.indices((8,8));local=np.stack([(u+.5-4)/f,-(v+.5-4)/f,-np.ones((8,8))],-1).reshape(64,3)
            dirs=local@rotation.T;dirs/=np.linalg.norm(dirs,axis=1)[:,None]
            r.update_scene(d,camera=name,scene_option=opt);r.enable_depth_rendering();raster.append(r.render().copy());r.disable_depth_rendering()
            pp=[];gg=[];dd=[]
            for ray in dirs:
                gid=np.array([-1],dtype=np.int32)
                dist=mj.mj_ray(m,d,origin,ray,opt.geomgroup,1,-1,gid)
                assert dist>0 and gid[0]>=0,(name,dist)
                pp.append(origin+dist*ray);gg.append(int(gid[0]));dd.append(dist)
            origins.append(origin);points.append(pp);gids.append(gg);depths.append(dd);directions.append(dirs)
    origins=np.array(origins);points=np.array(points);gids=np.array(gids)
    labels=np.array([[(m.body(int(m.geom_bodyid[g])).name if m.geom_bodyid[g] else m.geom(g).name) for g in row] for row in gids])
    assert not any(n.startswith('robot_0/') and n!='robot_0/base' for n in labels.ravel()),'Sensor mask must exclude robot.'
    np.savez_compressed(OUT/'measurements.npz',sensor_names=names,origins=origins,points_world=points,ray_directions=directions,ray_distance_m=depths,native_depth_m=raster,hit_geom_ids=gids,hit_bodies=labels)
    counts={str(n):int(np.sum(labels==n)) for n in np.unique(labels)}
    (OUT/'measurements.json').write_text(json.dumps(dict(sensor_count=40,pixels_per_sensor=[8,8],ray_count=2560,hit_counts=counts,method='Exact MuJoCo ray intersection along all native 8x8 pixel-center directions. Same geometry mask as skin depth renderer (robot hidden, sensor-only geometry enabled). Native 8x8 raster depths also exported.',display='All 2560 rays submitted to the renderer, with normal camera cropping and 3D occlusion. Teal denotes structural surfaces; amber denotes cup and clutter. No invented or accumulated points.'),indent=2))
    return names,origins,points,labels

TEAL=np.array([.02,.58,.65]);AMBER=np.array([1.,.43,.06])
def sphere(scene,p,color,radius):
    g=scene.geoms[scene.ngeom];mj.mjv_initGeom(g,mj.mjtGeom.mjGEOM_SPHERE,np.array([radius,0,0]),np.array(p),np.eye(3).ravel(),np.array(color,dtype=np.float32));g.segid=scene.ngeom;g.category=int(mj.mjtCatBit.mjCAT_DECOR);g.objtype=int(mj.mjtObj.mjOBJ_GEOM);g.objid=0;g.emission=.45;scene.ngeom+=1

def line(scene,a,b,color,width):
    g=scene.geoms[scene.ngeom];mj.mjv_initGeom(g,mj.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,dtype=np.float32));mj.mjv_connector(g,mj.mjtGeom.mjGEOM_LINE,width,np.array(a),np.array(b));g.segid=scene.ngeom;g.category=int(mj.mjtCatBit.mjCAT_DECOR);g.objtype=int(mj.mjtObj.mjOBJ_GEOM);g.objid=0;g.emission=.6;scene.ngeom+=1

def is_object(name):
    return any(s in name for s in ['cavity_obj','pact_preview_extra','pact_clutter_0','pact_distractor'])

def render_rear(m,d,origins,points,labels):
    """Rear camera, original cached scene materials and lighting, no recolouring."""
    opt=mj.MjvOption();opt.geomgroup[3:]=0;opt.sitegroup[:]=0
    m.vis.global_.fovy=43
    cam=mj.MjvCamera();cam.lookat=[.64,0,.87];cam.distance=2.2
    cam.azimuth=0;cam.elevation=-30
    with mj.Renderer(m,1600,2400,max_geom=15000) as r:
        for rays,stem in [(False,'scene_clean_rear'),(True,'scene_projections_rear')]:
            r.update_scene(d,cam,scene_option=opt)
            if rays:
                for origin,pp,ll in zip(origins,points,labels):
                    for p,label in zip(pp,ll):
                        obj=is_object(label);col=AMBER if obj else TEAL
                        near=any(x in label for x in ['hood','bench','jamb','sash','pact_intrusion','pact_clutter'])
                        line(r.scene,origin,p,[*col,.65 if obj else (.10 if near else .004)],1.1 if obj else .45)
                        sphere(r.scene,p,[*col,1. if obj else (.9 if near else .13)],.0048 if obj else .0027)
                    sphere(r.scene,origin,[1.,.48,.07,1],.0045)
            im=Image.fromarray(r.render().copy())
            im.save(OUT/f'{stem}.png',dpi=(300,300))
            if rays:
                im.save(OUT/f'{stem}.pdf',resolution=300)
                im.resize((1500,1000),Image.Resampling.LANCZOS).save(OUT/f'{stem}_preview.png')
    (OUT/'rear_view.json').write_text(json.dumps(dict(
        camera=dict(lookat=cam.lookat.tolist(),distance=cam.distance,azimuth=cam.azimuth,elevation=cam.elevation,fovy=43),
        rendering='Direct MuJoCo render of cached source model and state. Original scene geometry, materials and lighting retained. Sensor rays and endpoints are visualization overlays.',
        scene_provenance='source.json',measurements='measurements.npz',rays=2560),indent=2))
    print('Saved',OUT/'scene_projections_rear.png',flush=True)

def render(m,d,names,origins,points,labels,azimuth=315):
    opt=mj.MjvOption();opt.geomgroup[3:]=0;opt.sitegroup[:]=0
    m.vis.global_.fovy=43;m.vis.headlight.ambient[:]=.45;m.vis.headlight.diffuse[:]=.5
    # Plain neutral room materials make the measured samples readable.
    for i in range(m.ngeom):
        name=m.geom(i).name or ''
        if name=='floor' or name.startswith('room_'):
            m.geom_matid[i]=-1;m.geom_rgba[i]=[.93,.94,.95,1]
    cam=mj.MjvCamera();cam.lookat=[.64,.02,.87];cam.distance=2.2;cam.azimuth=azimuth;cam.elevation=-23
    with mj.Renderer(m,1500,2300,max_geom=15000) as r:
        for rays,filename in [(False,'scene_clean.png'),(True,'scene_projections.png')]:
            r.update_scene(d,cam,scene_option=opt)
            if rays:
                for origin,pp,ll in zip(origins,points,labels):
                    for p,label in zip(pp,ll):
                        obj=is_object(label);col=AMBER if obj else TEAL
                        near=any(x in label for x in ['hood','bench','jamb','sash','pact_intrusion','pact_clutter'])
                        line(r.scene,origin,p,[*col,.65 if obj else (.10 if near else .004)],1.1 if obj else .45)
                        sphere(r.scene,p,[*col,1. if obj else (.9 if near else .13)],.0048 if obj else .0027)
                    sphere(r.scene,origin,[1.,.48,.07,1],.0045)
            Image.fromarray(r.render().copy()).save(OUT/filename,dpi=(300,300))
    # Isolate the same arm pose with all forty sensor origins and short optical axes.
    groups=m.geom_group.copy();robot=[]
    for i in range(m.ngeom):
        n=m.body(int(m.geom_bodyid[i])).name
        if not n.startswith('robot_0/') or n=='robot_0/base':m.geom_group[i]=5
        else:robot.append(i)
    cam.lookat=[.23,.0,.69];cam.distance=1.45;cam.azimuth=315;cam.elevation=-18
    # Transparent MuJoCo background is composited onto white using the renderer's segmentation mask.
    with mj.Renderer(m,1500,1000,max_geom=5000) as r:
        r.update_scene(d,cam,scene_option=opt)
        for name,origin in zip(names,origins):
            ci=m.camera(name).id;axis=-d.cam_xmat[ci].reshape(3,3)[:,2]
            sphere(r.scene,origin,[1.,.43,.06,1],.006)
            line(r.scene,origin,origin+.025*axis,[1.,.43,.06,.9],1.4)
        rgb=r.render().copy()
        # Background pixels have no geometry; using segmentation preserves antialiasing at the robot boundary.
        r.enable_segmentation_rendering();seg=r.render();r.disable_segmentation_rendering()
        rgb[seg[:,:,0]<0]=255
        Image.fromarray(rgb).save(OUT/'skin_sensors.png',dpi=(300,300))
    m.geom_group[:]=groups
    left=Image.open(OUT/'skin_sensors.png');right=Image.open(OUT/'scene_projections.png')
    fig=Image.new('RGB',(3330,1500),'white');fig.paste(left,(0,0));fig.paste(right,(1030,0))
    fig.save(OUT/'figure.png',dpi=(300,300));fig.save(OUT/'figure.pdf',resolution=300)
    fig.resize((1776,800),Image.Resampling.LANCZOS).save(OUT/'preview.png')
    print('Saved',OUT/'figure.png',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--rebuild',action='store_true');p.add_argument('--rear',action='store_true');p.add_argument('--azimuth',type=float,default=315);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    m,d=build() if args.rebuild or not (OUT/'model.mjb').exists() else load()
    if not (OUT/'measurements.npz').exists() or args.rebuild:n,o,pts,labels=measure(m,d)
    else:
        z=np.load(OUT/'measurements.npz');n,o,pts,labels=[z[k] for k in ['sensor_names','origins','points_world','hit_bodies']]
    if args.rear:render_rear(m,d,o,pts,labels)
    else:render(m,d,n,o,pts,labels,args.azimuth)
if __name__=='__main__':main()
