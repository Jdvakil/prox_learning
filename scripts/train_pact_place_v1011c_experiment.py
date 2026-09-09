"""Six matched chunk-100 models, independent receipts and bounded disk use."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import time
from pact_place_v1011c_experiment import *
from pact_place_v109_contract import TRAIN_PARAMS, training_command, command_diff, ACT_TRAIN_COMMIT_V5


def preflight():
    split = read(WORK / 'split_manifest.json')
    conv = read(WORK / 'conversion_manifest_encoded.json')
    source = read(WORK / 'source_manifest.json')
    assert split['counts']['train']['total'] == source['counts']['accepted'] - 24
    assert split['counts']['validation']['total'] == 24
    assert read(WORK / 'embedding_report.json')['verified']
    assert sha256_file(Path(ENCODER_PATH)) == ENCODER_SHA256
    act = ROOT / 'submodules/act'
    changed = subprocess.check_output(['git','-C',str(act),'diff',ACT_TRAIN_COMMIT_V5,'HEAD','--numstat'],text=True)
    protected = {'imitate_episodes.py','policy.py','utils.py','fixed_split_data.py','surface_proximity_encoder.py'}
    for line in changed.splitlines():
        added,removed,path = line.split('\t')
        assert path not in protected and int(removed) == 0, line
    assert not subprocess.check_output(['git','-C',str(act),'status','--porcelain'],text=True).strip()
    horizon = max(635,conv['timesteps']['converted_t_max']+8)
    commands, differences = {}, {}
    for seed in SEEDS:
        pair = {}
        for arm in ('act','pact'):
            directory = TRAIN / f'{arm}_seed{seed}'
            assert not directory.exists() or not any(directory.iterdir()), directory
            command = training_command(arm=arm,ckpt_dir=str(directory),dataset_dir=str(DATA),
                split_manifest=str(WORK / 'split_manifest.json'),dataset_manifest=str(WORK / 'conversion_manifest_encoded.json'),
                expect_split_sha256=split['split_manifest_sha256'],expect_dataset_tree_sha256=conv['converted_tree_file_sha256'])
            command[0:2] = [sys.executable,str(ROOT / 'scripts/pact_place_v1011c_train_worker.py')]
            command[command.index('--seed')+1] = str(seed)
            command[command.index('--episode_horizon')+1] = str(horizon)
            pair[arm] = command
            commands[f'{arm}_seed{seed}'] = command
        differences[str(seed)] = command_diff(pair['act'],pair['pact'])
        assert differences[str(seed)]['identical_except_allowance']
    free = shutil.disk_usage(ROOT).free / 2**30
    # Five retained best checkpoints plus best/last for the final active model.
    assert free >= 3.2, f'{free:.2f} GiB free; bounded training needs 3.2 GiB'
    doc = {**empty_authorization(), 'ready':True,'train_params':{**TRAIN_PARAMS,'seed':list(SEEDS),'episode_horizon':horizon},
           'commands':commands,'command_diff':differences,'split_counts':split['counts'],
           'split_manifest_sha256':split['split_manifest_sha256'],'dataset_tree_sha256':conv['converted_tree_file_sha256'],
           'disk_free_gib':free,'checkpoint_storage':'No periodic snapshots/resume bundles; best and last verified, then last pruned. Best retained for all six models.',
           'training_source_drift_check':changed,'thread_environment':THREAD_ENV}
    doc['payload_sha256'] = digest(doc)
    freeze(WORK / 'training_preflight.json',doc)
    print(json.dumps({k:doc[k] for k in ('ready','split_counts','disk_free_gib')}),flush=True)


def train():
    from verify_pact_place_v1011c_models import verify_arm
    pre = read(WORK / 'training_preflight.json')
    split = read(WORK / 'split_manifest.json')
    assert pre['ready']
    results = {}
    for seed in SEEDS:
        for arm in ('act','pact'):
            key = f'{arm}_seed{seed}'
            directory = TRAIN / key
            receipt = WORK / 'models' / (key+'.json')
            if receipt.exists():
                old = read(receipt)
                assert old['verified'] and sha256_file(directory / 'policy_best.ckpt') == old['verification']['hashes']['policy_best.ckpt']
                results[key] = old
                continue
            assert not directory.exists() or not any(directory.iterdir()), f'Unclosed training attempt: {directory}'
            log = WORK / 'logs' / (key+'.log')
            started = time.time()
            print(f'{key}: starting 2000 epochs',flush=True)
            with log.open('x') as stream:
                process = subprocess.run(pre['commands'][key],cwd=ROOT / 'submodules/act',env=environment(),
                    stdout=stream,stderr=subprocess.STDOUT)
            doc = {**empty_authorization(),'arm':arm,'seed':seed,'returncode':process.returncode,
                   'elapsed_minutes':(time.time()-started)/60,'log':str(log.relative_to(ROOT)),'verified':False}
            if process.returncode:
                doc['output_tail'] = log.read_text()[-5000:]
                freeze(receipt,doc)
                print(doc['output_tail'],flush=True)
                raise SystemExit(process.returncode)
            verification = verify_arm(arm,split,seed)
            doc['verification'] = verification
            doc['verified'] = not verification['problems']
            if not doc['verified']:
                freeze(receipt,doc)
                raise RuntimeError(verification['problems'])
            last = directory / 'policy_last.ckpt'
            doc['pruned_last'] = {'path':str(last.relative_to(ROOT)),'sha256':sha256_file(last),'bytes':last.stat().st_size,
                                  'reason':'Verified final checkpoint; retain validation-selected best under disk constraint.'}
            freeze(receipt,doc)
            last.unlink()
            print(f"{key}: verified 2000 epochs, {doc['elapsed_minutes']:.1f} min; pruned {doc['pruned_last']['bytes']/2**20:.0f} MiB last checkpoint, retained best",flush=True)
            results[key] = doc
    freeze(WORK / 'training_timing.json',{**empty_authorization(),'complete':True,'models':results})


def verify():
    pre = read(WORK / 'training_preflight.json')
    models = read(WORK / 'training_timing.json')['models']
    assert set(models) == {f'{a}_seed{s}' for s in SEEDS for a in ('act','pact')}
    for key, result in models.items():
        assert result['verified'] and result['returncode'] == 0
        ver = result['verification']
        assert ver['epochs_recorded'] == 2000 and ver['strict_reload_ok']
        assert ver['split_manifest_sha256'] == pre['split_manifest_sha256']
        assert ver['dataset_tree_sha256'] == pre['dataset_tree_sha256']
        for name in ('policy_best.ckpt','dataset_stats.pkl','run_manifest.json'):
            assert sha256_file(TRAIN / key / name) == ver['hashes'][name]
    freeze(WORK / 'training_verification.json',{**empty_authorization(),'verified':True,'models':models,
        'authorization_fields_remain_false':True,'split_manifest_sha256':pre['split_manifest_sha256']})
    print('All six chunk-100 models verified',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage',choices=('preflight','train','verify'),required=True)
    a = p.parse_args()
    {'preflight':preflight,'train':train,'verify':verify}[a.stage]()
