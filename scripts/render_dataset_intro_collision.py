#!/usr/bin/env python3
"""Text-free intro figure using a real v12 task, meshes, and recorded motion.

The relocated collection bottle is a staged counterfactual; original rollout
was successful and collision-free. The transfer wrist angle is adapted smoothly; pick and place use recorded poses.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import json
from pathlib import Path
import h5py
import mujoco as mj
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from export_kitchen_collision_pair import ROOT,ROW,SCENE
from export_table_occlusion_pair import main as restore
import render_pact_place_v12_clutter as kitchen

OUT=ROOT/'images/intro_dataset_collision'
WINE='pact_preview_extra_Wine_Bottle_1/Wine_Bottle_1'
CUP='cavity_obj_0/Cup_10'
BOTTLE_POSITION=np.array([.55,.25,.883896])  # +Y is the robot's left.

def build():
    captured={}
    original=kitchen._attach_standing_kitchen
    def capture(spec):
        names=original(spec);captured.update(spec=spec,names=names);return names
    kitchen._attach_standing_kitchen=capture
    m,d,cam,k,opt,_,meta=restore(return_scene=True,row=ROW,scene_path=SCENE,kitchen=True)
    kitchen._attach_standing_kitchen=original
    s=captured['spec']
    # Freeze the collection overlay at the same supported poses as its loader.
    poses={m.body(i).name:(d.mocap_pos[m.body_mocapid[i]].copy(),d.mocap_quat[m.body_mocapid[i]].copy())
           for i in range(m.nbody) if m.body_mocapid[i]>=0}
    bottle=s.body(WINE)
    # Keep the actual bottle at its original scale, resting on the bench.
    bottle.pos=BOTTLE_POSITION-np.array(bottle.frame.pos)
    bottle.quat=poses[WINE][1]
    bottle.mocap=False
    bottle.add_freejoint(name='falling_wine_joint')
    s.option.timestep=.002
    s.visual.global_.offwidth=1800;s.visual.global_.offheight=1600
    s.visual.quality.offsamples=4
    m=s.compile();d=mj.MjData(m)
    for name,(p,q) in poses.items():
        idx=m.body(name).mocapid[0]
        if idx>=0:d.mocap_pos[idx]=p;d.mocap_quat[idx]=q
    kitchen._hide_primitive_colliders(m)
    mj.mj_forward(m,d)
    return m,d,cam,meta

def restore_frame(m,d,record):
    d.qpos[:13]=record['qpos'][:13]
    b=m.body(CUP).id
    m.body_pos[b]=record['object_position_m']
    m.body_quat[b]=record['object_quat_xyzw']
    mj.mj_forward(m,d)

def adapt_transfer(m,d,rows):
    """Preserve the recorded grasp center and endpoints; relax wrist pitch in transit."""
    rows=[dict(row,qpos=list(row['qpos'])) for row in rows]
    site=m.site('robot_0/gripper/grasp_site').id
    bounds=np.array([m.joint(f'robot_0/fr3_joint{i}').range for i in range(1,8)])
    for k in range(160,481):
        row=rows[k];d.qpos[:13]=row['qpos'][:13];mj.mj_forward(m,d)
        pos=d.site_xpos[site].copy();rot=d.site_xmat[site].reshape(3,3).copy()
        phase=np.clip(min((k-175)/45,(335-k)/40),0,1)
        phase=phase*phase*(3-2*phase)
        delta=Rotation.from_euler('y',-35*phase,degrees=True).as_matrix()
        target=delta@rot;seed=d.qpos[:7].copy()
        def error(q):
            d.qpos[:7]=q;mj.mj_kinematics(m,d)
            return np.r_[20*(d.site_xpos[site]-pos),
                         Rotation.from_matrix(d.site_xmat[site].reshape(3,3)@target.T).as_rotvec(),
                         .001*(q-seed)]
        fit=least_squares(error,seed,bounds=(bounds[:,0],bounds[:,1]),max_nfev=70)
        error(fit.x)
        assert np.linalg.norm(d.site_xpos[site]-pos)<.0002
        cp=np.array(row['object_position_m']);cq=np.array(row['object_quat_xyzw'])
        cr=Rotation.from_quat(cq[[1,2,3,0]]).as_matrix()
        nq=Rotation.from_matrix(delta@cr).as_quat()
        row['qpos'][:7]=fit.x.tolist()
        row['object_position_m']=(pos+delta@(cp-pos)).tolist()
        row['object_quat_xyzw']=nq[[3,0,1,2]].tolist()
    return rows


def simulate(m,d,rows):
    cup=m.body(CUP).id;wine=m.body(WINE).id
    ba=m.joint('falling_wine_joint').qposadr[0]
    d.qpos[ba:ba+7]=[*BOTTLE_POSITION,.70710678,.70710678,0,0]
    restore_frame(m,d,rows[160]);d.qvel[:]=0
    def state(frame):
        tilt=float(np.degrees(np.arccos(np.clip(d.xmat[wine].reshape(3,3)[2,1],-1,1))))
        return dict(frame=frame,qpos=d.qpos.copy(),cup_pos=m.body_pos[cup].copy(),cup_quat=m.body_quat[cup].copy(),tilt=tilt)
    first=None;frames=[];logs=[];link5_frames=[]
    for frame in range(160,480):
        a,c=rows[frame],rows[frame+1]
        qa,qc=np.array(a['qpos'][:13]),np.array(c['qpos'][:13])
        for substep in range(33):
            u=substep/33
            d.qpos[:13]=(1-u)*qa+u*qc;d.qvel[:13]=(qc-qa)/.066
            d.ctrl[:7]=d.qpos[:7]
            m.body_pos[cup]=(1-u)*np.array(a['object_position_m'])+u*np.array(c['object_position_m'])
            q=(1-u)*np.array(a['object_quat_xyzw'])+u*np.array(c['object_quat_xyzw'])
            m.body_quat[cup]=q/np.linalg.norm(q)
            mj.mj_step(m,d);mj.mj_forward(m,d)
            hits=[]
            for ci,contact in enumerate(d.contact):
                bodies=[m.body(m.geom_bodyid[g]).name for g in contact.geom]
                if any('Wine_Bottle' in n for n in bodies) and any('robot_0/' in n for n in bodies):
                    force=np.zeros(6);mj.mj_contactForce(m,d,ci,force)
                    if force[0]>.001:
                        hits.append(dict(bodies=bodies,force_N=float(force[0]),position=contact.pos.tolist()))
            if hits:logs.append(dict(source_frame=frame+u,contacts=hits))
        frames.append(state(frame))
        if any('robot_0/fr3_link5' in h['bodies'] for h in hits):
            link5_frames.append(frames[-1])
        if frame==175:first=frames[-1]
    assert logs
    contacted={n for row in logs for c in row['contacts'] for n in c['bodies'] if n.startswith('robot_0/')}
    assert not any('gripper' in name for name in contacted),contacted
    assert any('fr3_link5' in name for name in contacted),contacted
    assert all('robot_0/fr3_link5' in c['bodies'] for c in logs[0]['contacts'])
    assert frames[-1]['tilt']>60
    assert np.linalg.norm(frames[-1]['cup_pos']-rows[479]['object_position_m'])<1e-5
    np.savez_compressed(OUT/'replay.npz',**{k:np.array([r[k] for r in frames]) for k in frames[0]},
                        mocap_pos=d.mocap_pos,mocap_quat=d.mocap_quat)
    (OUT/'contact_log.json').write_text(json.dumps(logs,indent=2))
    middle=min(link5_frames,key=lambda st:abs(st['tilt']-25))
    assert middle['tilt']>15, 'Selected frame must show link 5 touching a tilted bottle.'
    return [first,middle,frames[-1]],logs


def render(m,d,selected):
    opt=mj.MjvOption();opt.geomgroup[3:]=0;opt.sitegroup[:]=0
    cam=mj.MjvCamera();cam.lookat=[.68,.04,.91];cam.distance=1.95;cam.azimuth=320;cam.elevation=-23
    m.vis.global_.fovy=45
    m.vis.headlight.ambient[:]=.35;m.vis.headlight.diffuse[:]=.45
    width,height,gap=1600,1143,12
    cup=m.body(CUP).id
    strip=Image.new('RGB',(width*3+gap*2,height),'white')
    with mj.Renderer(m,height,width) as renderer:
        for i,st in enumerate(selected):
            d.qpos[:]=st['qpos'];m.body_pos[cup]=st['cup_pos'];m.body_quat[cup]=st['cup_quat']
            mj.mj_forward(m,d);renderer.update_scene(d,cam,scene_option=opt)
            im=Image.fromarray(renderer.render().copy())
            im.save(OUT/f'frame_{i+1:02}.png',dpi=(300,300));strip.paste(im,((width+gap)*i,0))
    strip.save(OUT/'figure.png',dpi=(300,300))
    strip.save(OUT/'figure.pdf',resolution=300)
    strip.resize((1890,448),Image.Resampling.LANCZOS).save(OUT/'preview.png')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    m,d,cam,meta=build()
    adapted=adapt_transfer(m,d,meta['trajectory'])
    selected,logs=simulate(m,d,adapted)
    render(m,d,selected)
    (OUT/'provenance.json').write_text(json.dumps(dict(
        source_row=str(ROW),source_scene=str(SCENE),task=meta['task_description'],
        actual_cup_asset='Cup_10',actual_struck_asset='Wine_Bottle_1',source_task_success=meta['place_phase_success'],
        source_frames=[st['frame'] for st in selected],bottle_tilt_degrees=[st['tilt'] for st in selected],
        provenance='Staged counterfactual of a real dataset task, not an unmodified dataset collision episode. Recorded task adapted with a smooth wrist-pitch change during transfer, preserving grasp-center motion and recorded pick/place endpoints; bottle moves under MuJoCo contact dynamics.',
        changes=['Collection wine bottle placed on the robot left (+Y), among clutter on the bench at original scale and made dynamic.',
                 'Original hood opening, blue tray, cup and household meshes retained.',
                 'Wrist-pitch adjustment limited to 35 degrees, with smooth transitions and recorded task endpoints.',
                 'Wider fixed camera and reduced headlight intensity; mid-contact frame selected at a modest bottle tilt.',
                 'Three renders from one fixed camera; no labels or overlays.'],
        bottle_initial_position_m=BOTTLE_POSITION.tolist(),
        contacted_robot_bodies=sorted({n for row in logs for c in row['contacts'] for n in c['bodies'] if n.startswith('robot_0/')}),
        validation='First bottle contact is link 5. Middle frame has force-bearing link 5 contact with a tilted bottle. Links 6 and 7 also contact the bottle during the fall; no gripper contact. Bottle settles horizontally on the bench. Cup reaches recorded successful tray pose.',
        collection_overlay='Existing v12 loader rebuilds auxiliary kitchen poses, which are absent from frozen_config.'),indent=2))
    mj.mj_saveModel(m,str(OUT/'model.mjb'))
    print('Saved',OUT/'figure.png',flush=True)

if __name__=='__main__':main()
