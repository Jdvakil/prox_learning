"""Isolated, one-seed V10.11c dual-camera/forward-action pilot contract."""
from pact_place_v1011c_experiment import *
import pact_place_v1011c_experiment as original

WORK = ROOT / 'diagnostics_output/pact_place_v1011c_dualcam_aligned_s3103'
DATA = ROOT / 'assets/act_style_data/pact_place_v1011c_dualcam_aligned_99'
TRAIN = WORK / 'checkpoints'
EVAL = WORK / 'evaluation'
SEEDS = (3103,)
CAMERAS = ['wrist_camera', 'exo_camera_1']
MILESTONES = (20000, 30000)
EVAL_MASTER_SEED = 2026090501
DEV_MASTER_SEED = 2026090502
SMOKE_MASTER_SEED = 2026090503
RETRY_MASTER_SEED = 2026090504
DEADLINE_UTC = '2026-09-06T11:03:02+00:00'
ALIGNMENT = 'observation[t] -> raw commanded_action[t+1]; terminal status-only transition excluded'


def retry_seed(row, retry_index):
    raw = hashlib.sha256(f"{RETRY_MASTER_SEED}:{row['episode_id']}:{retry_index}".encode()).digest()
    value = int.from_bytes(raw[:8], 'big')
    return {'seed_u32': value % 2**32, 'seed_u64': value}


def load_eval_manifest(path):
    doc = read(path)
    assert doc['schema_version'] == 'pact_place_v1011c_paired_eval_v1'
    assert doc['manifest_sha256'] == digest({k:v for k,v in doc.items() if k != 'manifest_sha256'})
    for row in doc['rows'] + doc['smoke']['rows']:
        assert row['task_sampler_class'] == collection.SAMPLER_CLASS
        assert row['row_sha256'] == digest({k:v for k,v in row.items() if k != 'row_sha256'})
    return doc
