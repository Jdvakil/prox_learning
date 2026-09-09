"""Forward kinematics only: compare commanded and recorded hand trajectories."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import hashlib,json,sys
from pathlib import Path
import h5py,mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
ROOT=Path('/root/prox_learning_pact_remediation');OUT=Path(__file__).parent
src=OUT/(sys.argv[1] if len(sys.argv)>1 else 'geometry_234.json')
rows=json.loads(src.read_text())['rows']
robot=Path('/root/.cache/molmo-spaces-resources/robots/franka_droid/20260127/model.xml')
m=mujoco.MjModel.from_xml_path(str(robot));data=mujoco.MjData(m)
site=m.site('gripper/grasp_site').id
def fk(q):
    data.qpos[:7]=q;mujoco.mj_kinematics(m,data)
    return data.site_xpos[site].copy()
destdir=OUT/'commands';destdir.mkdir(exist_ok=True);out=[]
for i,r in enumerate(rows):
    with np.load(OUT/'episodes'/(r['rollout_id']+'.npz')) as z:
        a={k:z[k] for k in z.files}
    path=ROOT/r['directory']/'trajectory.h5';before=path.stat()
    with h5py.File(path,'r') as h:
        base=h['traj_0/obs/extra/robot_base_pose'][()].astype(float)
        tcp=h['traj_0/obs/extra/tcp_pose'][()].astype(float)
    assert (before.st_size,before.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
    br=R.from_quat(base[:,[4,5,6,3]])
    actual_fk=np.array([fk(q) for q in a['q_arm']])
    # The experiment mounts this asset 35 cm above its robot-base frame.
    # Verify the inferred fixed transform on EVERY recorded state, rather than
    # assuming that a standalone robot XML exactly matches the injected model.
    offset=tcp[0,:3]-actual_fk[0]
    assert np.allclose(offset,[0,0,.35],atol=1e-6)
    actual_world=br.apply(actual_fk+offset)+base[:,:3]
    residual=np.linalg.norm(actual_world-a['tcp_world'],axis=1)
    assert residual.max()<.01,(r['rollout_id'],residual.max())
    predicted_local=np.array([fk(q) for q in a['model_output'][:,:7]])+offset
    predicted=br[:-1].apply(predicted_local)+base[:-1,:3]
    # Command k is applied after observation k. Report both same-observation
    # demand and next-observation tracking, explicitly named.
    gap=a['tcp_world'][:-1]-predicted
    nextgap=a['tcp_world'][1:]-predicted
    np.savez_compressed(destdir/(r['rollout_id']+'.npz'),command_tcp_world=predicted,
                        actual_minus_command_world=gap,next_actual_minus_command_world=nextgap)
    row=dict(rollout_id=r['rollout_id'],seed=r['seed'],arm=r['arm'],episode_id=r['episode_id'],
             success=r['success'],failure_stage=r['failure_stage'],fk_offset_m=offset.tolist(),
             fk_verification_max_error_m=float(residual.max()),
             fk_verification_max_error_step=int(np.argmax(residual)),
             fk_verification_p99_error_m=float(np.quantile(residual,.99)))
    c=r['first_close_command_step']
    if c is not None:
        rim=a['target_world'][:-1,2]+r['target_collision_top_above_origin_m']
        xy=np.linalg.norm(predicted[:,:2]-a['target_world'][:-1,:2],axis=1)
        window=slice(max(c-10,0),min(c+11,900))
        heights=predicted[:,2]-rim
        row.update(command_at_close_tcp_above_rim_m=float(heights[c]),
                   fk_verification_max_error_around_close_m=float(residual[window].max()),
                   actual_minus_command_at_close_xyz_m=gap[c].tolist(),
                   command_at_close_xy_error_m=float(xy[c]),
                   command_z_change_2_steps_after_close_m=float(predicted[min(c+2,899),2]-predicted[c,2]),
                   actual_z_change_2_steps_after_close_m=float(a['tcp_world'][min(c+2,900),2]-a['tcp_world'][c,2]),
                   command_min_height_10steps_around_close_m=float(heights[window].min()),
                   max_next_tracking_z_gap_10steps_around_close_m=float(nextgap[window,2].max()),
                   max_next_tracking_xyz_gap_10steps_around_close_m=float(np.linalg.norm(nextgap[window],axis=1).max()),
                   rim_blocked_signature=bool(r['touch'] and not r['lift'] and
                     heights[c]<-.005 and gap[c,2]>.005 and
                     r.get('minimum_rim_relative_tcp_early_grasp_m',-1) is not None and
                     r['minimum_rim_relative_tcp_early_grasp_m']>=0))
    out.append(row)
    if i%50==0:print(i+1,len(rows),flush=True)
doc={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
     'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'robot_xml':str(robot),'robot_xml_sha256':hashlib.sha256(robot.read_bytes()).hexdigest(),
     'method':'mj_kinematics only; no mj_step, no policy inference. Fixed mount offset verified on every recorded state. TCP kinematics uses seven arm joints; passive finger joints do not affect the grasp site.',
     'rows':out}
(OUT/f'command_geometry_{len(rows)}.json').write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')
print('COMPLETE',len(rows),flush=True)
