"""Collection-identical cameras/sampler; two-RGB learned policy, original TE100."""
from pact_place_v1011c_dualcam import *
os.environ.update(environment())
import pickle
import cv2
import h5py
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import pact_blur
from policy import ACTPolicy
from surface_proximity_encoder import load_frozen_surface_embedding_encoder
from molmo_spaces.configs.camera_configs import FrankaSkinHybridCameraSystem, FrankaSkinHybridWristOnlyCameraSystem
import eval_pact_place_v1011c_row as base


class DualCameraInferencePolicy(base.V1011CInferencePolicy):
    def prepare_model(self, model_name=None):
        directory = Path(self.pc.checkpoint_dir)
        config = read(directory/'run_manifest.json')['policy_config']
        assert config['camera_names'] == CAMERAS and config['num_queries'] == 100
        assert config['n_proximity_sensors'] == (40 if self.pc.arm == 'PACT' else 0)
        self._pilot_checkpoint_step = read(directory/'checkpoint_step.json')['global_step']
        assert self._pilot_checkpoint_step in MILESTONES
        with base.legacy._detr_argv(self.pc.checkpoint_dir,self.pc.checkpoint_seed):
            policy = ACTPolicy(config)
        policy.load_state_dict(torch.load(self.pc.checkpoint_path,map_location='cpu',weights_only=True),strict=True)
        self._policy = policy.cuda().eval()
        projection = getattr(policy.model,'input_proj_proximity',None)
        self._input_proj_proximity_shape = list(projection.weight.shape) if projection is not None else None
        assert self._input_proj_proximity_shape == ([512,32] if self.pc.arm=='PACT' else None)
        with open(self.pc.stats_path,'rb') as stream:
            self._stats = pickle.load(stream)
        self._proximity_feature_dim = int(config['proximity_feature_dim'])
        if self.pc.arm == 'PACT':
            assert sha256_file(Path(self.pc.surface_encoder_path)) == ENCODER_SHA256
            encoder, payload = load_frozen_surface_embedding_encoder(self.pc.surface_encoder_path,map_location='cuda')
            assert int(payload['policy_feature_dim']) == self._proximity_feature_dim == 32
            self._surface_encoder = encoder.cuda().eval()

    def inference_model(self, observation):
        # Retain exactly the same complete initial-observation evidence as the
        # original evaluator, now including the table camera.
        if not self._initial_written:
            with h5py.File(base.OUTPUT/'initial_observation.h5','x') as handle:
                def write(prefix,value):
                    if isinstance(value,dict):
                        for key,item in sorted(value.items()):
                            write(prefix+'/'+str(key),item)
                    else:
                        array = np.asarray(value)
                        if array.dtype.kind in 'OU':
                            array = np.bytes_(json.dumps(value,sort_keys=True,default=str))
                        options = {'compression':'gzip'} if array.ndim and array.size else {}
                        handle.create_dataset(prefix,data=array,**options)
                write('observation',observation)
            self._initial_written = True
        if self._policy is None:
            self.prepare_model()
        stats = self._stats
        arm = np.asarray(observation['qpos']['arm'][:7],dtype=np.float32)
        gripper = np.asarray((observation['qpos'].get('gripper') or [0.0,0.0])[:2],dtype=np.float32)
        qpos = (np.concatenate((arm,gripper))-stats['qpos_mean'])/stats['qpos_std']
        qpos_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
        images = []
        for camera in CAMERAS:
            image = np.asarray(observation[camera])
            if image.dtype != np.uint8:
                image = (image*255.0 if float(np.max(image))<=1.0 else image).astype(np.uint8)
            if image.shape[:2] != (240,320):
                image = cv2.resize(image,(320,240),interpolation=cv2.INTER_AREA)
            images.append(image)
        image_tensor = torch.from_numpy(np.stack(images).astype(np.float32).transpose(0,3,1,2)[None]/255.0).cuda()
        assert image_tensor.shape == (1,2,3,240,320)
        sharp = image_tensor
        image_tensor = pact_blur.mean_fill_images(image_tensor) if self.pc.blind_rgb else pact_blur.blur_images(image_tensor,self.pc.blur_sigma)
        self._record_blur_diagnostic(sharp,image_tensor)
        raw = self._raw_proximity(observation)
        proximity_positions = self._surface_positions(raw) if self.pc.arm == 'PACT' else None
        self._record_sensor_diagnostic(raw,proximity_positions)
        with torch.inference_mode():
            predicted = self._policy(qpos_tensor,image_tensor,proximity_positions=proximity_positions)
        chunk = predicted.squeeze(0).cpu().numpy()
        assert chunk.shape == (100,8) and np.isfinite(chunk).all()
        chunk = chunk*stats['action_std']+stats['action_mean']
        self._pending_chunks.append((self._step,chunk))
        self._pending_chunks = [(start,value) for start,value in self._pending_chunks if self._step-start<len(value)]
        values,weights = [],[]
        for start,value in self._pending_chunks:
            age = self._step-start
            if 0 <= age < len(value):
                values.append(value[age])
                weights.append(np.exp(-0.01*age))
        weights_array = np.asarray(weights,dtype=np.float64)
        weights_array /= weights_array.sum()
        return (np.stack(values)*weights_array[:,None]).sum(axis=0).astype(np.float32)

    def get_info(self):
        info = super().get_info()
        info.update(policy_camera_names=CAMERAS, action_alignment=ALIGNMENT,
                    training_global_step=self._pilot_checkpoint_step, temporal_ensemble_history=100,
                    camera_system='FrankaSkinHybridCameraSystem')
        return info


class DualCameraPolicyConfig(base.V1011CPolicyConfig):
    policy_cls: type = DualCameraInferencePolicy


class DualCameraEvalConfig(base.V1011CEvalConfig):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        cameras = FrankaSkinHybridCameraSystem()
        wrist = FrankaSkinHybridWristOnlyCameraSystem()
        old_names = [c.name for c in wrist.cameras]
        new_names = [c.name for c in cameras.cameras]
        assert new_names[:2] == CAMERAS and new_names[2:] == old_names[1:]
        self.camera_config = cameras


def main():
    base.WORK = WORK
    base.EVAL = EVAL
    base.load_eval_manifest = load_eval_manifest
    base.retry_seed = retry_seed
    base.V1011CInferencePolicy = DualCameraInferencePolicy
    base.V1011CPolicyConfig = DualCameraPolicyConfig
    base.V1011CEvalConfig = DualCameraEvalConfig
    return base.main()


if __name__ == '__main__':
    raise SystemExit(main())
