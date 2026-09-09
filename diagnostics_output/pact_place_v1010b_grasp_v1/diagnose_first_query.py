"""Read-only first-observation model replay; no simulator rollout or updates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path('/root/prox_learning_pact_remediation/scripts')))
from pact_v1010b_contract import *
os.environ.update(environment());pin_torch()
import numpy as np,h5py,cv2,pickle,torch
from policy import ACTPolicy
import eval_pact_collision_row as legacy
scene=read(B/'scene_manifests/A1.json')['scenes'][0];job=read(B/'schedules/A1.json')[0]
paths={'original':ROOT/scene['historical_directories']['ACT'],'new':Path(job['output_dir'])}
model_dir=W/'checkpoints/act_seed3104';manifest=read(model_dir/'run_manifest.json')
with legacy._detr_argv(str(model_dir),3104):policy=ACTPolicy(manifest['policy_config'])
policy.load_state_dict(torch.load(model_dir/'policy_update_60000.ckpt',map_location='cpu',weights_only=True),strict=True)
policy.cuda().eval();stats=pickle.loads((W/'dataset_stats.pkl').read_bytes())
outputs={};inputs={};raws={}
for name,path in paths.items():
 with h5py.File(path/'initial_observation.h5') as h:
  arm=h['observation/qpos/arm'][()][:7].astype(np.float32);grip=h['observation/qpos/gripper'][()][:2].astype(np.float32)
  q=(np.r_[arm,grip]-stats['qpos_mean'])/stats['qpos_std']
  image=h['observation/wrist_camera'][()]
  if image.shape[:2]!=(240,320):image=cv2.resize(image,(320,240),interpolation=cv2.INTER_AREA)
 image_tensor=torch.from_numpy(np.transpose(image.astype(np.float32)/255.,(2,0,1))[None,None]).cuda()
 q_tensor=torch.from_numpy(q).float().cuda().unsqueeze(0)
 with torch.inference_mode():
  prediction=policy(q_tensor,image_tensor).squeeze(0).cpu().numpy()
  repeated=policy(q_tensor,image_tensor).squeeze(0).cpu().numpy()
 assert np.array_equal(prediction,repeated)
 output=prediction*stats['action_std']+stats['action_mean'];outputs[name]=output
 with np.load(path/'actions.npz') as z:raws[name]=z['model_output'][0].copy()
 inputs[name]={'qpos_sha256':hashlib.sha256(q.tobytes()).hexdigest(),
  'image_tensor_sha256':hashlib.sha256(image_tensor.cpu().numpy().tobytes()).hexdigest()}
result={'schema':SCHEMA,'checkpoint_sha256':sha(model_dir/'policy_update_60000.ckpt'),'inputs':inputs,
 'repeated_inference_exact':True,'runtime_torch':torch.__version__,'historical_torch':manifest['runtime']['torch'],
 'comparisons':{name:{'first_vector_exact':bool(np.array_equal(outputs[name][0],raws[name])),
  'first_vector_max_abs':float(np.max(np.abs(outputs[name][0]-raws[name]))),
  'reconstructed_first_vector':outputs[name][0].tolist(),'retained_first_vector':raws[name].tolist()} for name in paths},
 'original_to_new_predicted_first_delta':(outputs['new'][0]-outputs['original'][0]).tolist(),
 'original_to_new_retained_first_delta':(raws['new']-raws['original']).tolist()}
freeze(B/'first_query_reconstruction.json',result);print(json.dumps(result,indent=2))
