"""Read-only geometric measurements; no environment creation or policy execution."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import collections
import hashlib
import itertools
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT=Path('/root/prox_learning_pact_remediation')
WORK=ROOT/'diagnostics_output/pact_place_v1010_wrist288_s3_v1'
OUT=Path(__file__).parent
INPUT=OUT/(sys.argv[1] if len(sys.argv)>1 else 'features_234.json')
tracked={}
def checked(p):
    p=Path(p);s=p.stat();tracked.setdefault(p,(s.st_size,s.st_mtime_ns));return p
def read(p):return json.loads(checked(p).read_text())
def sha(p):return hashlib.sha256(checked(p).read_bytes()).hexdigest()
def first(a):return int(np.flatnonzero(a)[0]) if np.any(a) else None
def decode(x):return json.loads(bytes(x).split(b'\0')[0])
def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.generic):return safe(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

asset=Path('/root/.cache/molmo-spaces-resources/objects/thor/20251117/Kitchen Objects/Cup/Prefabs/Cup_10/Cup_10_prim.xml')
tree=ET.parse(checked(asset));body=tree.find('.//worldbody/body/body')
body_offset=np.fromstring(body.get('pos'),sep=' ')
signs=np.array(list(itertools.product((-1,1),repeat=3)))
boxes=[]
for geom in body.findall('geom'):
    if geom.get('type')!='box':continue
    pos=np.fromstring(geom.get('pos'),sep=' ');quat=np.fromstring(geom.get('quat'),sep=' ')
    size=np.fromstring(geom.get('size'),sep=' ')
    vertices=R.from_quat(quat[[1,2,3,0]]).apply(signs*size)+pos+body_offset
    boxes.append(dict(name=geom.get('name'),pos=pos,quat=quat,size=size,vertices=vertices))
local_vertices=np.concatenate([b['vertices'] for b in boxes])
assert len(boxes)==5
manifests={s:read(WORK/f'manifests/final_{s}.json') for s in (3103,3104,3105)}
layouts={r['episode_id']:r for m in manifests.values() for r in m['rows']}
source=read(INPUT);rows=source['rows'];enriched=[]
for i,r in enumerate(rows):
    d=ROOT/r['directory'];layout=layouts[r['episode_id']]
    with np.load(checked(OUT/'episodes'/(r['rollout_id']+'.npz'))) as a:
        a={k:a[k] for k in a.files}
    with h5py.File(checked(d/'initial_observation.h5'),'r') as h:
        q=h['physics/qpos'][()]
        target0=np.array(r['initial_target_xyz'])
        starts=[j for j in range(len(q)-6) if np.allclose(q[j:j+3],target0,atol=1e-7,rtol=0)]
        assert len(starts)==1
        quat=q[starts[0]+3:starts[0]+7];rot=R.from_quat(quat[[1,2,3,0]])
        gs=h['model/geom_size'][()];gp=h['model/geom_pos'][()];gq=h['model/geom_quat'][()]
        matches=[]
        for b in boxes:
            ids=np.flatnonzero(np.all(np.isclose(gs,b['size'],atol=1e-10,rtol=0),axis=1))
            assert len(ids)==1
            j=int(ids[0]);assert np.allclose(gp[j],b['pos'],atol=1e-9)
            assert abs(np.dot(gq[j],b['quat'])/np.linalg.norm(b['quat']))>1-1e-10
            matches.append(j)
    target_vertices=rot.apply(local_vertices)+target0
    top=float(target_vertices[:,2].max());bottom=float(target_vertices[:,2].min())
    with h5py.File(checked(d/'trajectory.h5'),'r') as h:
        tcp=h['traj_0/obs/extra/tcp_pose'][()].astype(float)
        base=np.array(r['initial_base_pose']);br=R.from_quat(base[[4,5,6,3]])
        tr=br*R.from_quat(tcp[:,[4,5,6,3]])
        scene=json.loads(h['traj_0/obs_scene'][()])
        video=bytes(h['traj_0/obs/sensor_data/wrist_camera'][()]).split(b'\0')[0].decode()
        assert not (d/video).is_file()
    w=a['tcp_world'];target=a['target_world'];xy=np.linalg.norm(w[:,:2]-target[:,:2],axis=1)
    c=r['first_close_command_step'];clutter0=a['clutter_positions'][0]
    diffs=[]
    for slot,center in zip(a['clutter_slots'],clutter0):
        item=next(x for x in layout['pact_clutter_palette'] if str(x['slot'])==str(slot))
        half=np.array(item['half_m'])
        gap=np.maximum(np.abs(target0-center)-half,0)
        diffs.append(dict(slot=str(slot),uid=item['uid'],center=center,half=half,
                          target_origin_to_clutter_aabb_m=float(np.linalg.norm(gap)),
                          target_origin_to_clutter_xy_aabb_m=float(np.linalg.norm(gap[:2]))))
    e=dict(r,role_index=layout['role_index'],initial_target_quat_wxyz=quat,
           target_primitive_geom_ids=matches,initial_target_collision_top_z_m=top,
           initial_target_collision_bottom_z_m=bottom,target_collision_vertices_world=target_vertices,
           target_collision_top_above_origin_m=top-target0[2],clutter_initial_geometry=diffs,
           initial_nearest_clutter_xy_aabb_m=min(x['target_origin_to_clutter_xy_aabb_m'] for x in diffs),
           initial_target_base_xyz_m=br.inv().apply(target0-base[:3]),
           initial_base_yaw_deg=float(br.as_euler('xyz',degrees=True)[2]),
           initial_target_yaw_deg=float(np.degrees(np.arctan2(rot.as_matrix()[1,0],rot.as_matrix()[0,0]))),
           initial_target_upright_tilt_deg=float(np.degrees(np.arccos(np.clip(rot.as_matrix()[2,1],-1,1)))),
           video_retained=False)
    if c is not None:
        end=min(c+60,900)
        near=(xy[:end+1]<.06)&(a['lift'][:end+1]<.01)
        # Rim reference follows measured translation, retaining the initial cup orientation.
        rim=target[:,2]+(top-target0[2]);height=w[:,2]-rim
        approaching=np.flatnonzero(near)
        zaxis=tr[c].apply([0,0,1])
        jaw_axis=tr[c].apply([0,1,0])
        jaw_xy=jaw_axis[:2]/np.linalg.norm(jaw_axis[:2])
        # Four side-wall top faces, excluding the bottom collider. Project the
        # initial rim footprint onto the gripper's horizontal closing direction.
        rim_vertices=np.concatenate([rot.apply(b['vertices'][b['vertices'][:,1]>.06])+target0
                                     for b in boxes if int(b['name'].rsplit('_',1)[1])!=2])
        projected=(rim_vertices[:,:2]-target0[:2])@jaw_xy
        rim_span=float(np.ptp(projected))
        wall_normals=np.array([rot.apply(R.from_quat(b['quat'][[1,2,3,0]]).apply([0,0,1])) for b in boxes if int(b['name'].rsplit('_',1)[1])!=2])
        normal_xy=wall_normals[:,:2]/np.linalg.norm(wall_normals[:,:2],axis=1,keepdims=True)
        wall_angle=float(np.degrees(np.arccos(np.clip(np.max(np.abs(normal_xy@jaw_xy)),-1,1))))
        tmin=int(approaching[np.argmin(height[approaching])]) if len(approaching) else None
        e.update(first_close_tcp_above_initial_rim_m=float(w[c,2]-top),
                 initial_rim_span_along_closing_xy_m=rim_span,
                 initial_rim_span_exceeds_nominal_open_gap=bool(rim_span>.087),
                 first_close_closing_axis_to_initial_wall_normal_deg=wall_angle,
                 first_close_hand_offset_along_closing_xy_m=float((w[c,:2]-target[c,:2])@jaw_xy),
                 first_close_tcp_above_translated_rim_m=float(height[c]),
                 first_close_tcp_z_axis_from_vertical_down_deg=float(np.degrees(np.arccos(np.clip(-zaxis[2],-1,1)))),
                 first_close_tcp_world_z_m=float(w[c,2]),first_close_target_world_z_m=float(target[c,2]),
                 first_close_raw_score_prev=float(a['model_output'][max(0,c-1),7]),
                 first_close_gripper_qpos=a['q_gripper'][c],
                 minimum_rim_relative_tcp_early_grasp_m=float(height[tmin]) if tmin is not None else None,
                 minimum_rim_relative_step_early_grasp=tmin,
                 tcp_world_z_change_10_steps_after_close_m=float(w[min(c+10,900),2]-w[c,2]),
                 tcp_world_z_change_20_steps_after_close_m=float(w[min(c+20,900),2]-w[c,2]),
                 target_world_z_change_20_steps_after_close_m=float(target[min(c+20,900),2]-target[c,2]),
                 below_rim_early_grasp_steps=int(np.count_nonzero(near&(height[:end+1]<0))),
                 below_rim_5mm_early_grasp_steps=int(np.count_nonzero(near&(height[:end+1]<-.005))),
                 below_rim_10mm_early_grasp_steps=int(np.count_nonzero(near&(height[:end+1]<-.01))),
                 close_to_tcp_rise_1cm_step=first(w[c:,2]>w[c,2]+.01),
                 contact_before_close_any_environment=any(r['collision_before_close'].values()))
    enriched.append(safe(e))
    if i%50==0:print('enriched',i+1,len(rows),flush=True)
for p,before in tracked.items():
    st=p.stat();assert before==(st.st_size,st.st_mtime_ns),str(p)
doc={'source_features':str(INPUT),'source_sha256':sha(INPUT),'augment_sha256':sha(Path(__file__)),
     'target_primitive_asset':str(asset),'target_primitive_asset_sha256':sha(asset),
     'input_files_unchanged_size_mtime':len(tracked),
     'geometry_note':'Exact initial primitive-box corners verified against saved model arrays. Rim at closure follows target translation with initial orientation; dynamic target orientation and passive finger joints are not retained, so this is a TCP-height reference, not exact pad penetration.',
     'rows':enriched}
dest=OUT/f'geometry_{len(rows)}.json';dest.write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')
print('COMPLETE',dest,flush=True)
