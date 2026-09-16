"""Portable inference adapter for the exported seed-3103 ACT / PACT models.

This adapter also ships in each model repository as inference.py. When running
this GitHub copy, pass --model-dir to the downloaded model bundle. The bundled
model/encoder definitions are unchanged copies of the evaluated source files.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.transforms import Normalize


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class RobotPolicy:
    """One control step per call; reset before starting each new episode."""

    def __init__(self, model_dir=None, device='cuda', verify=True):
        self.root = Path(model_dir or Path(__file__).resolve().parent).resolve()
        if verify:
            hashes = json.loads((self.root / 'checksums.json').read_text())
            for name, expected in hashes.items():
                path = (self.root / name).resolve()
                if not path.is_relative_to(self.root) or sha256(path) != expected:
                    raise ValueError(f'Model bundle checksum mismatch: {name}')
        self.config = json.loads((self.root / 'model_config.json').read_text())
        self.device = torch.device(device)
        runtime = self.root / 'runtime/prox_learning'
        for path in (runtime, runtime / 'submodules/act'):
            sys.path.insert(0, str(path))
        from detr.main import get_args_parser
        from detr.models import build_ACT_model

        args = get_args_parser().parse_args([
            '--ckpt_dir', str(self.root), '--policy_class', 'ACT',
            '--task_name', 'v1010', '--seed', '3103', '--num_epochs', '2000'])
        for key, value in self.config['policy_config'].items():
            setattr(args, key, value)
        # The checkpoint contains the complete backbone. Disable only the
        # constructor's unnecessary ImageNet download, then load every tensor.
        with patch('detr.models.backbone.is_main_process', return_value=False):
            self.policy = nn.Module()
            self.policy.model = build_ACT_model(args)
        state = torch.load(self.root / 'policy_last.ckpt', map_location='cpu', weights_only=True)
        self.policy.load_state_dict(state, strict=True)
        self.policy.to(self.device).eval().requires_grad_(False)
        del state
        with np.load(self.root / 'normalization.npz', allow_pickle=False) as stats:
            self.stats = {k: stats[k].copy() for k in stats.files}
        self.normalize = Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
        self.sensor_order = self.config['observation']['sensor_order']
        self.encoder = None
        if self.config['policy_config']['n_proximity_sensors']:
            from encoders.pact import build_pact_encoder
            pair = json.loads((self.root / 'checkpoint_pairs.json').read_text())['policy_last.ckpt']
            if pair['policy_sha256'] != sha256(self.root / 'policy_last.ckpt') or \
                    pair['encoder_sha256'] != sha256(self.root / 'prox_encoder.pt'):
                raise ValueError('Policy/encoder checkpoint pairing mismatch')
            self.encoder = build_pact_encoder('surface_embedding',
                checkpoint=self.root / 'prox_encoder.pt', device=str(self.device),
                layout='per_sensor', tokens_per_sensor=1, frozen=False, policy_tap='readout')
            self.encoder.sensor_order = list(self.sensor_order)
            self.encoder.eval().requires_grad_(False)
            if (self.encoder.n_act_sensors, self.encoder.act_feat_dim) != (40, 128):
                raise ValueError('Expected the jointly finetuned 40 x 128 readout')
        self.reset()

    def reset(self):
        self.history = deque(maxlen=8)
        self.pending_chunks = []
        self.control_step = 0
        self.last_tokens = None
        self.last_chunk = None
        self.last_model_output = None

    @torch.inference_mode()
    def step(self, qpos, wrist_rgb, proximity=None):
        """Return absolute arm targets and the original 0/255 gripper command.

        qpos: arm7 + two gripper joint positions, shape (9,).
        wrist_rgb: RGB HWC array; uint8 or the original float image convention.
        proximity (PACT): metres, (40,4,8,8) or pre-min-pooled (40,8,8),
        in self.sensor_order, or a dictionary keyed by those 40 sensor names.
        """
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.shape != (9,) or not np.isfinite(qpos).all():
            raise ValueError('qpos must contain 7 arm and 2 gripper joint positions')
        normalized_qpos = (qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        state = torch.from_numpy(normalized_qpos).float().to(self.device).unsqueeze(0)
        rgb = np.asarray(wrist_rgb)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError('wrist_rgb must be an RGB HWC image')
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255. if float(np.max(rgb)) <= 1. else rgb).astype(np.uint8)
        if rgb.shape[:2] != (240, 320):
            rgb = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(np.transpose(rgb.astype(np.float32) / 255., (2, 0, 1))[None, None]).to(self.device)
        tokens = None
        if self.encoder is not None:
            from encoders.pact import causal_pooled_window, encode_for_act
            from encoders.peak_closeness import stack_obs_proximity
            if isinstance(proximity, dict):
                raw = {name: np.asarray(proximity[name], dtype=np.float32) for name in self.sensor_order}
            else:
                raw_array = np.asarray(proximity, dtype=np.float32)
                if raw_array.shape not in ((40, 4, 8, 8), (40, 8, 8)):
                    raise ValueError('PACT needs proximity in metres, shape (40,4,8,8) or (40,8,8)')
                raw = dict(zip(self.sensor_order, raw_array))
            pooled = stack_obs_proximity(raw, self.sensor_order, pool='min')
            self.history.append(pooled.copy())
            window = causal_pooled_window(np.stack(self.history), len(self.history) - 1)
            tokens = encode_for_act(self.encoder, torch.from_numpy(window).to(self.device).unsqueeze(0))
            self.last_tokens = tokens.detach().cpu().numpy().copy()
        predicted, _, _ = self.policy.model(state, self.normalize(image), None, proximity_positions=tokens)
        chunk = predicted.squeeze(0).cpu().numpy()
        chunk = chunk * self.stats['action_std'] + self.stats['action_mean']
        if chunk.shape != (100, 8) or not np.isfinite(chunk).all():
            raise ValueError('Invalid action chunk')
        self.last_chunk = chunk.copy()
        self.pending_chunks = self.pending_chunks[-99:]
        self.pending_chunks.append((self.control_step, chunk))
        self.pending_chunks = [(s, x) for s, x in self.pending_chunks if self.control_step - s < len(x)]
        values, weights = [], []
        for start, value in self.pending_chunks:
            elapsed = self.control_step - start
            values.append(value[elapsed])
            weights.append(np.exp(-.01 * elapsed))
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()
        action = (np.stack(values) * weights[:, None]).sum(axis=0).astype(np.float32)
        self.last_model_output = action.copy()
        self.control_step += 1
        return {'arm': action[:7].copy(),
                'gripper': np.asarray([0. if float(action[7]) < 127.5 else 255.], dtype=np.float32)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path,
                        help='Downloaded Hugging Face bundle; defaults to this script directory')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    policy = RobotPolicy(model_dir=args.model_dir, device=args.device)
    with np.load(policy.root / 'example_observation.npz', allow_pickle=False) as sample:
        action = policy.step(sample['qpos'], sample['wrist_rgb'], sample.get('proximity'))
        error = float(np.max(np.abs(policy.last_model_output - sample['expected_model_output'])))
        normalized_error = float(np.max(np.abs(
            (policy.last_model_output - sample['expected_model_output']) / policy.stats['action_std'])))
        # CPU/GPU kernels can differ slightly; report the difference in both
        # command units and standardized action units rather than claim identity.
        reference_check_enforced = policy.device.type == 'cuda'
        matches = np.allclose(policy.last_model_output, sample['expected_model_output'], rtol=1e-5, atol=1e-5)
        if reference_check_enforced and not matches:
            raise ValueError(f'Published example disagrees with its historical action; max error {error}')
    print(json.dumps({'model': policy.config['model_name'], 'device': str(policy.device),
        'example_max_absolute_error': error, 'example_max_normalized_error': normalized_error,
        'reference_check_enforced': reference_check_enforced,
        'reference_matches_cuda_tolerance': bool(matches), 'arm': action['arm'].tolist(),
        'gripper': action['gripper'].tolist(), 'checksums_verified': True}))


if __name__ == '__main__':
    main()
