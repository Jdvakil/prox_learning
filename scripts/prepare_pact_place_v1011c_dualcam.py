"""Create-only derivative: reuse verified sensor embeddings, add recorded table RGB.

The source corpus, old conversion and original split are never opened for writing.
"""
from pact_place_v1011c_dualcam import *
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import collections
import subprocess
import time
import shutil
import cv2
import h5py
import numpy as np


def decode(value):
    return json.loads(bytes(value).split(b'\0', 1)[0])


def check_hash(item):
    path, expected = item
    actual = sha256_file(ROOT / path)
    assert actual == expected, (path, actual, expected)
    return path


def convert(job):
    row, old = job
    idx = row['act_episode_index']
    previous = original.DATA / old['act_file']
    assert sha256_file(previous) == old['act_file_sha256']
    raw_path = ROOT / row['trajectory_h5']
    assert sha256_file(raw_path) == row['trajectory_h5_sha256']
    video = raw_path.parent / 'episode_00000000_exo_camera_1.mp4'
    video_hash = sha256_file(video)
    assert video_hash == row['coordinator_validation']['table_camera']['detail']['rgb_video_sha256']
    with h5py.File(raw_path, 'r') as raw:
        group = raw['traj_0']
        commands = [decode(x) for x in group['actions/commanded_action'][()]]
        joint = [decode(x) for x in group['actions/joint_pos'][()]]
        assert commands[0] == {} and joint[-1] == {}
        raw_t = len(commands)
        assert len(joint) == raw_t
        status_only = 'arm' not in commands[-1]
        if status_only:
            assert set(commands[-1]) == {'success'}
        valid = commands[1:-1] if status_only else commands[1:]
        assert all(len(x['arm']) == 7 and len(x['gripper']) == 1 for x in valid)
        action = np.asarray([x['arm'] + x['gripper'] for x in valid], dtype=np.float32)
        n = len(action)
        assert n == raw_t - 1 - int(status_only)
        assert np.array_equal(np.asarray([x['arm'] for x in joint[1:-1]], dtype=np.float32),
                              np.asarray([x['arm'] for x in commands[1:-1]], dtype=np.float32))
        qpos = np.asarray([decode(x)['arm'] + decode(x)['gripper']
                           for x in group['obs/agent/qpos'][:n]], dtype=np.float32)
    target = DATA / old['act_file']
    assert not target.exists(), target
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_N_THREADS, 1) if hasattr(cv2, 'CAP_PROP_N_THREADS') else None
    assert capture.isOpened(), video
    with h5py.File(previous, 'r') as src, h5py.File(target, 'x') as dst:
        assert src.attrs['pact_surface_encoder_sha256'] == ENCODER_SHA256
        assert np.array_equal(qpos, src['observations/qpos'][:n])
        assert np.array_equal(src['action'][:, :7], np.asarray([x['arm'] for x in joint[:-1]], dtype=np.float32))
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs['derivative_recipe'] = 'v1011c_dualcam_aligned_v1'
        dst.attrs['action_alignment'] = ALIGNMENT
        dst.attrs['parent_converted_sha256'] = old['act_file_sha256']
        dst.attrs['exo_rgb_video_sha256'] = video_hash
        dst.create_dataset('action', data=action)
        src.copy('observations', dst)
        src.copy('pact_provenance', dst)
        # Two terminal-status episodes lose one extra observation. Copy all other
        # sensor values bit-for-bit; never recompute the frozen embeddings.
        if n != len(src['action']):
            names = []
            dst['observations'].visititems(lambda name, item: names.append('observations/' + name)
                if isinstance(item, h5py.Dataset) and item.shape and item.shape[0] == len(src['action']) else None)
            for name in names:
                source = src[name]
                kwargs = {'compression': source.compression, 'compression_opts': source.compression_opts} if source.compression else {}
                values = source[:n]
                del dst[name]
                dst.create_dataset(name, data=values, **kwargs)
        images = dst.create_dataset('observations/images/exo_camera_1', (n,240,320,3), dtype='u1',
                                    chunks=(1,240,320,3), compression='gzip', compression_opts=1)
        frames = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame.shape == (352,624,3), frame.shape
            if frames < n:
                images[frames] = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320,240), interpolation=cv2.INTER_AREA)
            frames += 1
        capture.release()
        assert frames == raw_t, (idx, frames, raw_t)
        assert np.array_equal(dst['observations/proximity_embeddings'][()], src['observations/proximity_embeddings'][:n])
        assert np.isfinite(action).all()
    result = dict(old)
    result.update(act_file_sha256=sha256_file(target), converted_timesteps=n, timesteps=n,
                  terminal_status_transitions_excluded=int(status_only), raw_timesteps=raw_t,
                  action_shape=list(action.shape), qpos_shape=[n,9], image_shape=[n,240,320,3],
                  proximity_shape=[n,40,4,8,8], extrinsic_shape=[n,40,3,4],
                  intrinsic_shape=[n,40,3,3], camera_names=CAMERAS,
                  exo_frames_decoded=frames, exo_rgb_video_sha256=video_hash,
                  parent_converted_sha256=old['act_file_sha256'], action_alignment=ALIGNMENT)
    result['act_h5_sha256'] = result['act_file_sha256']
    result.pop('pre_embedding_act_file_sha256', None)
    result.pop('pre_embedding_act_semantic_sha256', None)
    print(f'converted {idx}: {n} aligned transitions, two RGB cameras', flush=True)
    return result


def main():
    started = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    source = read(original.WORK / 'source_manifest.json')
    old = read(original.WORK / 'conversion_manifest_encoded.json')
    split = read(original.WORK / 'split_manifest.json')
    assert source['verified'] and len(source['rows']) == 99
    assert source['encoder_sha256'] == ENCODER_SHA256 == sha256_file(Path(ENCODER_PATH))
    assert subprocess.check_output(['git','-C',str(ROOT/'submodules/molmospaces'),'branch','--show-current'],text=True).strip() == 'experiment/pact-vs-act-remediation-v2'
    protected = source['protected_source_files'] | source['implementation_hashes']
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check_hash, protected.items()))
    freeze(WORK / 'source_manifest.json', source)
    # Re-derive validation IDs from actual accepted ledger rows, then verify that
    # the original 75/24 split used precisely this rule.
    ledger = [json.loads(x) for x in (ROOT/source['ledger_path']).read_text().splitlines() if x.strip()]
    accepted = [r for r in ledger if r['accepted']]
    assert {r['attempt_id'] for r in accepted} == {r['attempt_id'] for r in source['rows']}
    cells = collections.defaultdict(list)
    for row in accepted:
        cells[row['cell']].append(row['attempt_id'])
    validation = {min(ids, key=lambda x: hashlib.sha256(f'{SPLIT_SEED}:{x}'.encode()).hexdigest()) for ids in cells.values()}
    assert len(validation) == 24
    assert {e['episode_id'] for e in split['episodes'] if e['split'] == 'validation'} == validation
    freeze(WORK / 'pilot_plan.json', {**empty_authorization(), 'seed':3103,
        'arms':['ACT','PACT'], 'cameras':CAMERAS, 'action_alignment':ALIGNMENT,
        'train_episodes':75, 'validation_episodes':24, 'training_chunk':100,
        'temporal_ensemble_history':100, 'ensemble_age_decay':0.01, 'gripper_threshold':127.5,
        'updates_per_arm':30000, 'checkpoint_update_counts':list(MILESTONES), 'batch_size':8,
        'learning_rate':1e-5, 'encoder_sha256':ENCODER_SHA256, 'no_new_collection':True,
        'development_paired_instances':12, 'final_paired_instances':50, 'initial_eval_workers':12,
        'checkpoint_selection':'Shared update count: highest mean two-arm development task success; ties by collision-free success, fewer hazard frames, then earlier checkpoint.',
        'development_and_final_disjoint':True, 'deadline_utc':DEADLINE_UTC,
        'hardware':{'cpu_quota_cores':15.36,'memory_limit_bytes':183497654272,'gpu':'NVIDIA A10',
                    'gpu_memory_mib':23028,'pids_max':3840,'disk_free_bytes':shutil.disk_usage(ROOT).free},
        'thread_environment':THREAD_ENV, 'historical_artifacts_read_only':True})
    by_index = {r['act_episode_index']:r for r in old['episodes']}
    with ProcessPoolExecutor(max_workers=6) as pool:
        episodes = list(pool.map(convert, [(r,by_index[r['act_episode_index']]) for r in source['rows']]))
    episodes.sort(key=lambda e:e['act_episode_index'])
    tree = hashlib.sha256()
    for episode in episodes:
        tree.update(f"{episode['act_file']}\x1f{episode['act_file_sha256']}\n".encode())
    result = dict(old)
    result.update(schema_version='pact_place_v1011c_dualcam_aligned_conversion_v1', episodes=episodes,
        dataset_root=str(DATA.relative_to(ROOT)), converted_tree_file_sha256=tree.hexdigest(),
        parent_conversion_manifest_sha256=sha256_file(original.WORK/'conversion_manifest_encoded.json'),
        converter_module_sha256=sha256_file(Path(__file__)), camera_names=CAMERAS, action_alignment=ALIGNMENT,
        elapsed_seconds=time.time()-started,
        timesteps={'converted_t_min':min(e['timesteps'] for e in episodes),
                   'converted_t_max':max(e['timesteps'] for e in episodes),
                   'converted_t_sum':sum(e['timesteps'] for e in episodes)},
        terminal_status_transitions_excluded=sum(e['terminal_status_transitions_excluded'] for e in episodes))
    assert result['timesteps']['converted_t_sum'] == 37869
    assert result['terminal_status_transitions_excluded'] == 2
    result.pop('payload_sha256', None)
    result['payload_sha256'] = digest(result)
    freeze(WORK/'conversion_manifest_encoded.json', result)
    split.update(experiment='pact_place_v1011c_dualcam_aligned_99', source_collection_tree_sha256=tree.hexdigest(),
                 parent_split_manifest_sha256=split['split_manifest_sha256'], action_alignment=ALIGNMENT, camera_names=CAMERAS)
    split.pop('split_manifest_sha256')
    split['split_manifest_sha256'] = digest(split)
    freeze(WORK/'split_manifest.json', split)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check_hash, protected.items()))
    freeze(WORK/'conversion_verification.json', {**empty_authorization(), 'verified':True,
        'episodes':99,'transitions':37869,'source_files_unchanged':len(protected),
        'encoder_reused_bitwise':True,'camera_names':CAMERAS,'elapsed_seconds':time.time()-started,
        'dataset_tree_sha256':tree.hexdigest()})
    print('CONVERSION VERIFIED', json.dumps(result['timesteps']), flush=True)


if __name__ == '__main__':
    main()
