"""Reconstruct the instantaneous held heuristic from recorded solver contacts."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from pact_v1010b_contract import *
from pact_v1010b_metrics import decode
import h5py,numpy as np
scene=next(s for s in read(B/'scene_manifests/A1.json')['scenes'] if s['scene_id'].startswith('0f25bb'))
path=B/'rollouts/A1/A1_PACT_3105_frozen60000_0f25bb0b1a72d962/attempt_00'
with h5py.File(path/'initial_observation.h5') as h:parent=h['model/body_parentid'][()]
with h5py.File(path/'trajectory.h5') as h:grasp=[decode(v)['gripper'] for v in h['traj_0/obs/extra/grasp_state_pickup_obj'][()]]
with h5py.File(path/'grasp_diagnostics.h5') as h:
 meta=json.loads(h.attrs['metadata']);contacts=h['contacts/solver_samples'][()];times=h['physics/time'][()];control_time=h['control/time'][()]
 cup=set(meta['cup_geom_ids']);gripper_root=meta['body_all_names'].index('robot_0/gripper/base')
 def is_gripper(geom):
  body=int(meta['geom_body_ids'][geom])
  while body:
   if body==gripper_root:return True
   body=int(parent[body])
  return False
 rows=[]
 for obs in range(130,180):
  ix=int(np.argmin(abs(times-control_time[obs])));assert abs(times[ix]-control_time[obs])<1e-7
  current=contacts[contacts[:,0]==ix];other=[]
  for row in current:
   g1,g2=int(row[2]),int(row[3])
   if (g1 in cup)==(g2 in cup):continue
   g=g2 if g1 in cup else g1;other.append({'geom_id':g,'name':meta['geom_names'][g],'is_gripper':is_gripper(g),'distance_m':float(row[6])})
  touching=any(g['is_gripper'] for g in other);held=touching and all(g['is_gripper'] for g in other)
  rows.append({'observation':obs,'time_s':float(control_time[obs]),'physics_frame':ix,'recorded':grasp[obs],'reconstructed':{'touching':touching,'held':held},'other_contact_geoms':other})
report={'schema':SCHEMA,'utc':now(),'analysis_source_sha256':sha(__file__),'sensor_source_sha256':sha(ROOT/'submodules/molmospaces/molmo_spaces/env/sensors.py'),
 'sources':{str(path/n):sha(path/n) for n in ('initial_observation.h5','trajectory.h5','grasp_diagnostics.h5')},
 'definition':'held = at least one gripper contact and no non-gripper cup contact; no duration or lift requirement',
 'reconstructed_observations':len(rows),'all_exact':all(r['recorded']==r['reconstructed'] for r in rows),'rows':rows}
freeze(B/'category_shift_0f25bb_contact_reconstruction.json',report)
print(json.dumps({'all_exact':report['all_exact'],'held_rows':[r for r in rows if r['recorded']['held']],'mismatches':[r for r in rows if r['recorded']!=r['reconstructed']]},indent=2))
