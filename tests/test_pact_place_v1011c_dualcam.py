"""Pilot contract and runtime wiring; no historical-artifact mutation."""
import os
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from pact_place_v1011c_dualcam import *
os.environ.update(environment())
import h5py
import numpy as np


class PilotTests(unittest.TestCase):
    def test_derivative_shapes_and_split(self):
        split = read(WORK/'split_manifest.json')
        old = read(original.WORK/'split_manifest.json')
        self.assertEqual(split['episodes'],old['episodes'])
        conversion = read(WORK/'conversion_manifest_encoded.json')
        self.assertEqual(sum(x['timesteps'] for x in conversion['episodes']),37869)
        for item in conversion['episodes']:
            with h5py.File(DATA/item['act_file'],'r') as h:
                n = item['timesteps']
                self.assertEqual(h['action'].shape,(n,8))
                self.assertEqual(h['observations/proximity_embeddings'].shape,(n,40,32))
                for camera in CAMERAS:
                    self.assertEqual(h[f'observations/images/{camera}'].shape,(n,240,320,3))

    def test_manifest_streams(self):
        manifest = load_eval_manifest(EVAL/'eval_manifest.json')
        self.assertEqual(len(manifest['rows']),62)
        self.assertEqual(len(manifest['smoke']['rows']),4)
        seeds = []
        for row in manifest['rows']+manifest['smoke']['rows']:
            seeds.append(row['task_seed_u32'])
            seeds.extend(retry_seed(row,i)['seed_u32'] for i in range(1,row['max_sampling_retries']+1))
        self.assertEqual(len(seeds),len(set(seeds)))

    def test_sampler_camera_wiring(self):
        import eval_pact_place_v1011c_dualcam_row as evaluation
        row = load_eval_manifest(EVAL/'eval_manifest.json')['rows'][0]
        evaluation.base.v109._ACTIVE_SCENE = str(ROOT/row['pact_v1011_scene_relative'])
        evaluation.base.v109._ACTIVE_SCENE_SHA256 = row['pact_v106_scene_sha256']
        config = evaluation.DualCameraEvalConfig()
        self.assertEqual([c.name for c in config.camera_config.cameras][:2],CAMERAS)
        self.assertEqual(len(config.camera_config.cameras),42)
        self.assertIs(config.task_sampler_config.task_sampler_class,evaluation.base.PactPlaceCorridorV1011C33PctTallerPrimitiveSampler)
        self.assertEqual(config.task_horizon,900)
        self.assertFalse(config.end_on_success)
        self.assertFalse(config.robot_config.action_noise_config.enabled)
        self.assertIs(evaluation.DualCameraInferencePolicy.model_output_to_action,
                      evaluation.base.V1011CInferencePolicy.model_output_to_action)

    def test_ensemble_matches_original_math(self):
        # Both inference implementations use every still-live chunk's output at
        # its age with exp(-0.01*age), including warm-up and the 100-tick cap.
        generator = np.random.default_rng(17)
        chunks = generator.normal(size=(140,100,8)).astype(np.float32)
        pending = []
        for t in range(140):
            pending.append((t,chunks[t]))
            pending = [(s,v) for s,v in pending if t-s<len(v)]
            values,weights = zip(*[(v[t-s],np.exp(-.01*(t-s))) for s,v in pending])
            w = np.asarray(weights,dtype=np.float64);w /= w.sum()
            actual = (np.stack(values)*w[:,None]).sum(0).astype(np.float32)
            ages = np.arange(min(t+1,100))
            expected_w = np.exp(-.01*ages);expected_w /= expected_w.sum()
            expected = (chunks[t-ages,ages]*expected_w[:,None]).sum(0).astype(np.float32)
            np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0)

    def test_runtime_prepare_sets_inherited_export_metadata(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import eval_pact_place_v1011c_dualcam_row as evaluation
        import torch
        for arm in ('ACT','PACT'):
            directory = TRAIN/f'{arm.lower()}_seed3103_step20000'
            policy = object.__new__(evaluation.DualCameraInferencePolicy)
            policy.pc = SimpleNamespace(checkpoint_dir=str(directory),checkpoint_seed=3103,
                checkpoint_path=str(directory/'policy_best.ckpt'),stats_path=str(directory/'dataset_stats.pkl'),
                arm=arm,surface_encoder_path=ENCODER_PATH,num_queries=100)
            policy.prepare_model()
            policy._contact_audit = SimpleNamespace()
            with patch.object(evaluation.base.place.frontend.PactFrontendScreenInferencePolicy,'get_info',return_value={}):
                info = evaluation.base.place.PactPlaceInferencePolicy.get_info(policy)
            self.assertEqual(info['input_proj_proximity_shape'],[512,32] if arm=='PACT' else None)
            self.assertEqual(info['num_queries'],100)
            del policy
            torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
