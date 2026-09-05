"""Keep finetuned encoder weights paired with the selected ACT policy weights."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re

def file_digest(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def encoder_filename(policy_name: str) -> str:
    if policy_name == 'policy_best.ckpt':
        return 'prox_encoder_best.pt'
    if policy_name == 'policy_last.ckpt':
        return 'prox_encoder.pt'
    if re.fullmatch(r'policy_epoch_\d+_seed_-?\d+\.ckpt', policy_name):
        return policy_name.replace('policy_', 'prox_encoder_', 1).replace('.ckpt', '.pt')
    raise ValueError(f'No encoder pairing convention for {policy_name}')


def paired_encoder_checkpoint(directory, config, policy_name='policy_best.ckpt'):
    """Never substitute pretrained/best weights for a missing finetuned pair.

    Historical best/last pairs without an index remain loadable. New training
    writes hashes so modifications or half-written pairs fail before inference.
    """
    if not config.get('finetune_prox_encoder'):
        return None
    directory = Path(directory)
    encoder = directory / encoder_filename(policy_name)
    if not encoder.is_file():
        raise ValueError(f'Missing matching finetuned encoder for {policy_name}: {encoder}')
    index = directory / 'checkpoint_pairs.json'
    if index.exists():
        pair = json.loads(index.read_text()).get(policy_name)
        if pair is None or pair['encoder'] != encoder.name:
            raise ValueError(f'Missing checkpoint pair record for {policy_name}')
        if (file_digest(directory / policy_name) != pair['policy_sha256'] or
                file_digest(encoder) != pair['encoder_sha256']):
            raise ValueError(f'Checkpoint pair hash mismatch for {policy_name}')
    return encoder
