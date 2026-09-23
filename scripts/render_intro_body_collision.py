#!/usr/bin/env python3
"""Staged MuJoCo intro illustration; scripted arm/cup, free dynamic bottle.

Run with /opt/conda/envs/prox/bin/python scripts/render_intro_body_collision.py.
This is an illustrative reconstruction, not an evaluated policy rollout.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import json
from pathlib import Path
import mujoco as mj
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'images/intro_body_collision'
SCENE = ROOT/'submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/fumehood_clutter.xml'
ROBOT = ROOT/'assets/robots/franka_skin/model_hybrid.xml'

def build():
    s = mj.MjSpec.from_file(str(SCENE))
    r = mj.MjSpec.from_file(str(ROBOT))
    s.attach(r, prefix='robot/', frame=s.worldbody.add_frame(pos=[.08, 0, .35]))
    s.worldbody.add_geom(name='pedestal', type=mj.mjtGeom.mjGEOM_BOX,
                         pos=[.08, 0, .175], size=[.13,.13,.175], rgba=[.22,.25,.29,1])
    # Open the existing adjustable jambs for a clear view into the hood.
    s.body('jamb_l').pos = [.58,.57,.92]
    s.body('jamb_r').pos = [.58,-.57,.92]
    s.body('sash').pos = [.58,0,1.44]
    clutter = {'sh0': [.12,.61,.50], 'sh1':[.34,.66,.50],
               'sh2':[.13,.64,.90], 'sh3':[.35,.65,.90], 'sh4':[.22,.63,1.30],
               'hd0':[1.12,.29,.805], 'hd1':[1.18,.05,.805],
               'hd2':[1.10,-.30,.805], 'hd3':[.90,.31,.805],
               'cb0':[-.57,.19,.96], 'cb1':[-.55,-.02,.96], 'cb2':[-.55,-.22,.96]}
    for name, p in clutter.items(): s.body('clut_'+name).pos = p
    # A hollow cup built from actual MuJoCo geometry, with a visible loop handle.
    cup = s.worldbody.add_body(name='task_cup', mocap=True, pos=[.7,.04,.78])
    teal = [.03,.58,.53,1]
    cup.add_geom(type=mj.mjtGeom.mjGEOM_CYLINDER, size=[.035,.004,0], pos=[0,0,-.047], rgba=teal, contype=0, conaffinity=0)
    for i in range(48):
        a=2*np.pi*i/48
        cup.add_geom(type=mj.mjtGeom.mjGEOM_BOX, size=[.003,.0027,.05],
                     pos=[.033*np.cos(a),.033*np.sin(a),0],
                     quat=[np.cos(a/2),0,0,np.sin(a/2)], rgba=teal, contype=0, conaffinity=0)
    for i in range(18):
        a=2*np.pi*i/18; b=2*np.pi*(i+1)/18
        cup.add_geom(type=mj.mjtGeom.mjGEOM_CAPSULE, size=[.004,0,0],
                     fromto=[0,-.046-.021*np.cos(a),.028*np.sin(a),0,-.046-.021*np.cos(b),.028*np.sin(b)],
                     rgba=teal, contype=0, conaffinity=0)
    bottle=s.worldbody.add_body(name='hazard_bottle', pos=[.37,-.16,.94])
    bottle.add_freejoint(name='bottle_free')
    bottle.add_geom(name='bottle_body', type=mj.mjtGeom.mjGEOM_CYLINDER, size=[.048,.22,0],
                    rgba=[.77,.32,.055,1], mass=.20, friction=[.75,.01,.001])
    bottle.add_geom(name='bottle_label', type=mj.mjtGeom.mjGEOM_CYLINDER, size=[.0485,.065,0],
                    rgba=[.98,.87,.59,1], contype=0, conaffinity=0, mass=.001)
    bottle.add_geom(name='bottle_neck', type=mj.mjtGeom.mjGEOM_CYLINDER, pos=[0,0,.236],
                    size=[.024,.02,0], rgba=[.65,.26,.045,1], mass=.005)
    bottle.add_geom(name='bottle_cap', type=mj.mjtGeom.mjGEOM_CYLINDER, pos=[0,0,.26],
                    size=[.029,.010,0], rgba=[.19,.21,.23,1], mass=.005)
    s.worldbody.add_geom(name='bench_extension', type=mj.mjtGeom.mjGEOM_BOX, pos=[.44,-.18,.70], size=[.16,.16,.02], rgba=[.85,.86,.88,1])
    for name in ['cabinet_body','cabinet_top']:
        s.geom(name).pos[0] -= .45
    for name in ['clut_cb0','clut_cb1','clut_cb2']:
        s.body(name).pos[0] -= .45
    s.add_texture(name='background',type=mj.mjtTexture.mjTEXTURE_SKYBOX,builtin=mj.mjtBuiltin.mjBUILTIN_FLAT,rgb1=[.9,.92,.94],rgb2=[.9,.92,.94],width=128,height=768)
    # Goal marker is a thin disk on the existing cart.
    s.worldbody.add_geom(name='place_marker', type=mj.mjtGeom.mjGEOM_CYLINDER,
                         pos=[.32,-.56,.621], size=[.085,.001,0], rgba=[.64,.84,.77,1], contype=0, conaffinity=0)
    s.visual.global_.offwidth=1600; s.visual.global_.offheight=1400
    s.visual.quality.offsamples=4
    s.visual.headlight.ambient=[.45]*3; s.visual.headlight.diffuse=[.6]*3
    s.option.timestep=.002
    m=s.compile(); d=mj.MjData(m)
    for g in range(m.ngeom):
        if m.geom(g).name.startswith('room_'): m.geom_group[g]=5
    return s,m,d

def setup(m,d):
    ids=[m.joint(f'robot/fr3_joint{i}').id for i in range(1,8)]
    qa=m.jnt_qposadr[ids]; va=m.jnt_dofadr[ids]
    site=m.site('robot/gripper/grasp_site').id
    q0=np.array([0,-.2,0,-1.8,0,1.6,.785])
    bounds=m.jnt_range[ids]
    def ik(p, seed):
        def error(q):
            d.qpos[qa]=q; mj.mj_forward(m,d)
            rot=d.site_xmat[site].reshape(3,3)
            dr=Rotation.from_matrix(rot@np.diag([1,-1,-1])).as_rotvec()
            return np.r_[15*(d.site_xpos[site]-p),dr,.005*(q-seed)]
        fit=least_squares(error,seed,bounds=(bounds[:,0]+.001,bounds[:,1]-.001),max_nfev=180)
        error(fit.x)
        assert np.linalg.norm(d.site_xpos[site]-p)<.003, (p,d.site_xpos[site])
        return fit.x
    ps=np.array([[.70,.04,.80],[.70,.04,.87],[.55,-.13,.89],[.37,-.39,.85],[.32,-.56,.674],[.32,-.56,.83]])
    qs=[]
    for p in ps:
        q0=ik(p,q0); qs.append(q0.copy())
        print('pose',p,'forearm',d.xpos[m.body('robot/fr3_link5').id],flush=True)
    return qa,va,site,ps,np.array(qs)

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    s,m,d=build(); qa,va,site,ps,qs=setup(m,d)
    cid=m.body('task_cup').mocapid[0]
    badr=m.joint('bottle_free').qposadr[0]
    times=np.array([0, .65,1.6,2.55,3.4,4.1])
    states=[]; logs=[]
    armact=[m.actuator(f'robot/fr3_joint{i}').id for i in range(1,8)]
    d.qpos[qa]=qs[0]; d.qvel[:]=0
    for step in range(2401):
        t=step*m.opt.timestep
        seg=min(np.searchsorted(times,t,side='right')-1,4)
        a=np.clip((t-times[seg])/(times[seg+1]-times[seg]),0,1); a=a*a*(3-2*a)
        q=(1-a)*qs[seg]+a*qs[seg+1]
        old=d.qpos[qa].copy(); d.qpos[qa]=q; d.qvel[va]=(q-old)/m.opt.timestep
        d.ctrl[armact]=q
        d.ctrl[m.actuator('robot/gripper/fingers_actuator').id]=110 if t<3.4 else 0
        mj.mj_forward(m,d)
        d.mocap_pos[cid]=d.site_xpos[site] if t<3.4 else ps[4]
        d.mocap_quat[cid]=[1,0,0,0]
        mj.mj_step(m,d); mj.mj_forward(m,d)
        contacts=[]
        for i,c in enumerate(d.contact):
            bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom]
            if 'hazard_bottle' in bodies and any(b.startswith('robot/') for b in bodies):
                force=np.zeros(6); mj.mj_contactForce(m,d,i,force)
                contacts.append(dict(bodies=bodies,position=c.pos.tolist(),force_N=float(force[0])))
        tilt=float(np.degrees(np.arccos(np.clip(d.xmat[m.body('hazard_bottle').id].reshape(3,3)[2,2],-1,1))))
        if step%5==0:
            states.append(dict(t=t,qpos=d.qpos.copy(),qvel=d.qvel.copy(),mocap_pos=d.mocap_pos.copy(),mocap_quat=d.mocap_quat.copy(),tilt=tilt,contacts=contacts))
        if contacts: logs.append(dict(t=t,tilt=tilt,contacts=contacts))
    (OUT/'contact_log.json').write_text(json.dumps(logs,indent=2))
    np.savez_compressed(OUT/'states.npz',**{k:np.array([st[k] for st in states]) for k in ['t','qpos','qvel','mocap_pos','mocap_quat','tilt']})
    print('contacts',len(logs),'max tilt',max(st['tilt'] for st in states),'final',states[-1]['tilt'],flush=True)
    # Verify that the knockdown is from the body, never the end effector.
    assert logs and max(c['force_N'] for row in logs for c in row['contacts']) > .1
    assert all(all('gripper' not in b for b in c['bodies']) for row in logs for c in row['contacts'])
    assert all(any(b in ['robot/fr3_link5','robot/fr3_link6'] for b in c['bodies']) for row in logs for c in row['contacts'])
    assert 85 < states[-1]['tilt'] < 95
    assert np.linalg.norm(states[-1]['mocap_pos'][cid]-ps[4]) < 1e-6
    # Middle frame is just after the strike, while the bottle is toppling.
    selected=[min(states,key=lambda st:abs(st['t']-t)) for t in [.25,2.30,4.8]]
    opt=mj.MjvOption(); opt.geomgroup[3:]=0; opt.sitegroup[:]=0
    cam=mj.MjvCamera(); cam.lookat=[.38,-.10,.64]; cam.distance=2.72; cam.azimuth=65; cam.elevation=-26
    width,height,gap=1500,1375,16
    with mj.Renderer(m,height,width) as ren:
        strip=Image.new('RGB',(3*width+2*gap,height),'white')
        for i,st in enumerate(selected):
            for key in ['qpos','qvel','mocap_pos','mocap_quat']: getattr(d,key)[:]=st[key]
            mj.mj_forward(m,d); ren.update_scene(d,cam,scene_option=opt)
            im=Image.fromarray(ren.render().copy())
            im.save(OUT/f'frame_{i+1:02}.png',dpi=(300,300))
            strip.paste(im,((width+gap)*i,0))
        strip.save(OUT/'figure.png',dpi=(300,300))
        strip.save(OUT/'figure.pdf',resolution=300)
        strip.resize((1813,550),Image.Resampling.LANCZOS).save(OUT/'preview.png')
    (OUT/'provenance.json').write_text(json.dumps(dict(source_scene=str(SCENE),robot=str(ROBOT),
        kind='Staged MuJoCo illustration: prescribed arm joint path and cup attachment; bottle moves under mj_step contact dynamics. Not a policy evaluation.',
        frame_times=[st['t'] for st in selected],frame_bottle_tilt_degrees=[st['tilt'] for st in selected],
        first_arm_contact=logs[0],last_arm_contact=logs[-1],
        contacted_robot_bodies=sorted({b for row in logs for c in row['contacts'] for b in c['bodies'] if b.startswith('robot/')}),
        middle_frame='Bottle toppling immediately after arm contact; no image overlays.',
        modifications=['Existing clutter populated; cabinet shifted back 0.45 m; adjustable hood jambs widened and sash raised.',
                       'Bench extension, free dynamic bottle, geometric hollow cup and cart goal disk added.',
                       'Room shell hidden for external camera; all three frames use identical scene and camera.'],
        validation='Positive-force arm contact; no bottle/gripper contacts; final bottle horizontal; cup at cart destination.'),indent=2))
    mj.mj_saveModel(m,str(OUT/'scene.mjb'))

if __name__=='__main__': run()
