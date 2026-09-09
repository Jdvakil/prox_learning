"""Reconstruct the original/new first query from their actual saved inputs."""
import sys,argparse,collections
from pathlib import Path
sys.path.insert(0,'/root/prox_learning_pact_remediation/scripts')
from pact_v1010b_contract import *
os.environ.update(environment());pin_torch()
import numpy as np,h5py,cv2,pickle,torch
from policy import ACTPolicy
import eval_pact_collision_row as legacy
import eval_pact_frontend_screen_row as frontend
from surface_proximity_encoder import load_frozen_surface_embedding_encoder

def reconstruct(arm,scene_id,training_seed,new_directory):
 scene=next(r for r in read(B/'scene_manifests/A1.json')['scenes'] if r['scene_id']==scene_id)
 paths={'original':ROOT/scene['historical_directories'][arm],'new':Path(new_directory)}
 model_dir=W/f'checkpoints/{arm.lower()}_seed{training_seed}';manifest=read(model_dir/'run_manifest.json')
 with legacy._detr_argv(str(model_dir),training_seed):policy=ACTPolicy(manifest['policy_config'])
 policy.load_state_dict(torch.load(model_dir/'policy_update_60000.ckpt',map_location='cpu',weights_only=True),strict=True)
 policy.cuda().eval();stats=pickle.loads((W/'dataset_stats.pkl').read_bytes())
 encoder=load_frozen_surface_embedding_encoder(str(ENCODER_PATH),map_location='cuda')[0].cuda().eval() if arm=='PACT' else None
 outputs={};inputs={};raws={};images={}
 for name,path in paths.items():
  with h5py.File(path/'initial_observation.h5') as h:
   q=np.r_[h['observation/qpos/arm'][()][:7].astype(np.float32),h['observation/qpos/gripper'][()][:2].astype(np.float32)]
   q=(q-stats['qpos_mean'])/stats['qpos_std'];image=h['observation/wrist_camera'][()]
   if image.shape[:2]!=(240,320):image=cv2.resize(image,(320,240),interpolation=cv2.INTER_AREA)
   raw=np.stack([h['observation/'+sensor][()].astype(np.float32) for sensor in CANONICAL_SENSOR_NAMES]) if arm=='PACT' else None
  images[name]=image
  image_tensor=torch.from_numpy(np.transpose(image.astype(np.float32)/255.,(2,0,1))[None,None]).cuda()
  q_tensor=torch.from_numpy(q).float().cuda().unsqueeze(0);prox=None
  with torch.inference_mode():
   if arm=='PACT':
    stub=type('ReadonlyEncoderContext',(),{})();stub._proximity_history=collections.deque(maxlen=8);stub._surface_encoder=encoder
    prox=frontend.PactFrontendScreenInferencePolicy._surface_positions(stub,raw)
   prediction=policy(q_tensor,image_tensor,proximity_positions=prox).squeeze(0).cpu().numpy()
   repeated=policy(q_tensor,image_tensor,proximity_positions=prox).squeeze(0).cpu().numpy()
  assert np.array_equal(prediction,repeated)
  outputs[name]=prediction*stats['action_std']+stats['action_mean']
  with np.load(path/'actions.npz') as z:raws[name]=z['model_output'][0].copy()
  inputs[name]={'qpos_sha256':hashlib.sha256(q.tobytes()).hexdigest(),
   'image_tensor_sha256':hashlib.sha256(image_tensor.cpu().numpy().tobytes()).hexdigest(),
   'proximity_sha256':hashlib.sha256(prox.cpu().numpy().tobytes()).hexdigest() if prox is not None else None}
 delta=np.abs(images['original'].astype(int)-images['new'].astype(int))
 return {'schema':SCHEMA,'analysis_code_sha256':sha(__file__),'arm':arm,'scene_id':scene_id,'training_seed':training_seed,
  'checkpoint_sha256':sha(model_dir/'policy_update_60000.ckpt'),'inputs':inputs,'preprocessed_rgb_changed_channels':int(np.count_nonzero(delta)),
  'preprocessed_rgb_max_delta':int(delta.max()),'repeated_inference_exact':True,
  'comparisons':{name:{'first_vector_exact':bool(np.array_equal(outputs[name][0],raws[name])),
   'first_vector_max_abs':float(np.max(np.abs(outputs[name][0]-raws[name]))),
   'reconstructed_first_vector':outputs[name][0].tolist(),'retained_first_vector':raws[name].tolist()} for name in paths},
  'predicted_delta':(outputs['new'][0]-outputs['original'][0]).tolist(),'retained_delta':(raws['new']-raws['original']).tolist()}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--scene-id',required=True);p.add_argument('--training-seed',type=int,required=True);p.add_argument('--new-directory',type=Path,required=True);a=p.parse_args()
 result=reconstruct(a.arm,a.scene_id,a.training_seed,a.new_directory)
 freeze(B/f'replay_query_reconstruction/{a.arm}_{a.scene_id}.json',result);print(json.dumps(result,indent=2))
