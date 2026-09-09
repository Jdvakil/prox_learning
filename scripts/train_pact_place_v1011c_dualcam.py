"""Two fresh one-seed runs; actual process exits and dense update counts recorded."""
from pact_place_v1011c_dualcam import *
from pact_place_v109_contract import training_command, command_diff
import subprocess
import time


def main():
    assert read(WORK/'conversion_verification.json')['verified']
    split = read(WORK/'split_manifest.json')
    conv = read(WORK/'conversion_manifest_encoded.json')
    assert split['counts']['train']['total'] == 75 and split['counts']['validation']['total'] == 24
    commands = {}
    for arm in ('act','pact'):
        directory = TRAIN/f'{arm}_seed3103'
        command = training_command(arm=arm,ckpt_dir=str(directory),dataset_dir=str(DATA),
            split_manifest=str(WORK/'split_manifest.json'),dataset_manifest=str(WORK/'conversion_manifest_encoded.json'),
            expect_split_sha256=split['split_manifest_sha256'],expect_dataset_tree_sha256=conv['converted_tree_file_sha256'])
        command[:2] = [sys.executable,str(ROOT/'scripts/train_pact_place_v1011c_dualcam_worker.py')]
        for flag, value in {'--seed':'3103','--num_epochs':'3000','--ckpt_every':'100','--num_workers':'4'}.items():
            command[command.index(flag)+1] = value
        i = command.index('--camera_names') + 1
        command[i:i+1] = CAMERAS
        commands[arm] = command
    diff = command_diff(commands['act'],commands['pact'])
    assert diff['identical_except_allowance'], diff
    freeze(WORK/'training_preflight.json', {**empty_authorization(),'commands':commands,'command_diff':diff,
        'epochs_per_arm':3000,'expected_updates_per_arm':30000,'camera_names':CAMERAS,
        'dataset_tree_sha256':conv['converted_tree_file_sha256'], 'split_manifest_sha256':split['split_manifest_sha256'],
        'worker_script_sha256':sha256_file(ROOT/'scripts/train_pact_place_v1011c_dualcam_worker.py')})
    for arm, command in commands.items():
        receipt = WORK/'models'/f'{arm}_seed3103.json'
        if receipt.exists():
            assert read(receipt)['verified']
            continue
        directory = TRAIN/f'{arm}_seed3103'
        assert not directory.exists() or not any(directory.iterdir()), directory
        log = WORK/'logs'/f'{arm}_seed3103.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        with log.open('x') as out:
            process = subprocess.Popen(command,cwd=ROOT/'submodules/act',env=environment(),
                stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            freeze(WORK/'launches'/f'{arm}_seed3103.json',{'pid':process.pid,'started_unix':start,'command':command,'log':str(log)})
            code = process.wait()
        result = {**empty_authorization(),'arm':arm,'seed':3103,'returncode':code,
            'elapsed_seconds':time.time()-start,'log':str(log.relative_to(ROOT)),'verified':False}
        if code:
            result['output_tail'] = log.read_text()[-6000:]
            freeze(receipt,result)
            print(json.dumps(result),flush=True)
            raise SystemExit(code)
        epochs = [json.loads(x) for x in (directory/'epoch_log.jsonl').read_text().splitlines() if x.strip()]
        assert [x['epoch'] for x in epochs] == list(range(3000))
        assert [x['global_step'] for x in epochs] == list(range(10,30001,10))
        manifest = read(directory/'run_manifest.json')
        assert manifest['policy_config']['camera_names'] == CAMERAS
        assert manifest['policy_config']['num_queries'] == 100
        milestones = {}
        for step in MILESTONES:
            row = read(directory/f'policy_step_{step}.json')
            assert row['global_step'] == step and row['checkpoint_sha256'] == sha256_file(directory/f'policy_step_{step}.ckpt')
            view = TRAIN/f'{arm}_seed3103_step{step}'
            view.mkdir(exist_ok=False)
            for name, target in {'policy_best.ckpt':f'policy_step_{step}.ckpt',
                                 'dataset_stats.pkl':'dataset_stats.pkl','run_manifest.json':'run_manifest.json'}.items():
                os.link(directory/target, view/name)
            freeze(view/'checkpoint_step.json',row)
            milestones[str(step)] = row
        result.update(verified=True,epochs_recorded=3000,updates_recorded=30000,milestones=milestones,
                      dataset_stats_sha256=sha256_file(directory/'dataset_stats.pkl'))
        freeze(receipt,result)
        print(f'{arm} training complete: actual exit 0, 30000 updates, {result["elapsed_seconds"]/60:.1f} min',flush=True)
    act = read(WORK/'models/act_seed3103.json')
    pact = read(WORK/'models/pact_seed3103.json')
    assert act['dataset_stats_sha256'] == pact['dataset_stats_sha256']
    freeze(WORK/'training_completion.json',{**empty_authorization(),'complete':True,'models':{'act':act,'pact':pact}})


if __name__ == '__main__':
    main()
