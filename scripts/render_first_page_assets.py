"""Render source-based scene assets for the conceptual first-page figure."""
import json,os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import mujoco,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'images/first_page/assets'
SCENE=ROOT/'submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v2.xml'
ROBOT=ROOT/'assets/robots/franka_skin/model_hybrid.xml'
s=mujoco.MjSpec.from_file(str(SCENE))
r=mujoco.MjSpec.from_file(str(ROBOT))
f=s.worldbody.add_frame(pos=[0,0,.35])
s.attach(r,prefix='robot/',frame=f)
s.worldbody.add_geom(name='paper_pedestal',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[0,0,.175],size=[.13,.13,.175],rgba=[.31,.34,.39,1])
s.worldbody.add_geom(name='paper_target',type=mujoco.mjtGeom.mjGEOM_CYLINDER,pos=[.82,.05,.79],size=[.032,.07,0],rgba=[.13,.56,.44,1])
s.visual.global_.offwidth=2200;s.visual.global_.offheight=1700
s.visual.quality.offsamples=8
s.visual.headlight.ambient=[.45]*3;s.visual.headlight.diffuse=[.55]*3
s.add_texture(name='paper_background',type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,rgb1=[1,1,1],rgb2=[1,1,1],width=128,height=768)
m=s.compile();d=mujoco.MjData(m)
for i in range(m.ngeom):
 name=m.geom(i).name
 if name.startswith('room_') or name=='floor':m.geom_group[i]=5
 if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH and m.mesh(m.geom_dataid[i]).name.rsplit('/',1)[-1] in ['link0_4','link0_6']:
  m.geom_matid[i]=-1;m.geom_rgba[i]=[0,0,0,1]
for i,v in enumerate([0,-.1,0,-1.9,0,1.8,.79],1):d.qpos[m.joint(f'robot/fr3_joint{i}').qposadr[0]]=v
mujoco.mj_forward(m,d)
opt=mujoco.MjvOption();opt.geomgroup[3:]=0;opt.sitegroup[:]=0
for az in [300]:
 cam=mujoco.MjvCamera();cam.lookat=[.62,0,.81];cam.distance=2.72;cam.azimuth=az;cam.elevation=-18
 with mujoco.Renderer(m,1700,2200) as ren:
  ren.update_scene(d,cam,scene_option=opt)
  ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
  arr=ren.render().copy();Image.fromarray(arr).save(OUT/f'fumehood_{az}.png',dpi=(300,300))
(OUT/'scene_provenance.json').write_text(json.dumps(dict(scene=str(SCENE),robot=str(ROBOT),kind='Conceptual scene rendered from repository meshes; not an evaluated rollout.',presentation_changes=['Robot on 0.35 m pedestal','Green cylinder represents task target','Room shell and floor hidden','Uniform black base collar suppresses branding','No hazard or glass added to this source render']),indent=2))
print('Saved scene assets',flush=True)
