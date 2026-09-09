"""Continue the already-started training pipeline through evaluation/reporting."""
import argparse
import fcntl
import os
import subprocess
import sys
import time
import traceback
from pact_place_v1011c_experiment import *
from run_pact_place_v1011c_training_pipeline import run_stage


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--training-pid',type=int,required=True)
    p.add_argument('--adopt-training',action='store_true',
                   help='Original outer coordinator was lost; verify surviving trainer artifacts on completion.')
    p.add_argument('--detach',action='store_true')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--resume-full',action='store_true')
    a=p.parse_args()
    if a.detach:
        directory=WORK/'coordinator_recovery'
        directory.mkdir(parents=True,exist_ok=True)
        attempt=len(list(directory.glob('launch_*.json')))+1
        log=directory/f'coordinator_{attempt:03d}.log'
        command=[sys.executable,str(Path(__file__).resolve()),'--training-pid',str(a.training_pid)]
        if a.adopt_training:
            command.append('--adopt-training')
        command += ['--workers',str(a.workers)]
        if a.resume_full:
            command.append('--resume-full')
        with log.open('x') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=environment(),stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        freeze(directory/f'launch_{attempt:03d}.json',{**empty_authorization(),'pid':child.pid,
            'training_pid':a.training_pid,'adopt_training':a.adopt_training,'command':command,
            'log':str(log.relative_to(ROOT)),'started_unix_s':time.time()})
        print(f'Detached coordinator PID {child.pid}; log {log}',flush=True)
        return
    lock=(WORK/'coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    print('Waiting for six-model training verification; all evaluation settings frozen.',flush=True)
    while not (WORK/'training_verification.json').exists():
        for file in (WORK/'stages').glob('*.json'):
            stage=read(file)
            if stage['returncode']:
                subprocess.run([sys.executable,'scripts/finalize_pact_place_v1011c_eval.py','--failure-only'],cwd=ROOT,env=environment())
                raise SystemExit(f"Upstream failed: {stage['stage']}")
        if a.adopt_training and (WORK/'training_timing.json').exists():
            timing=read(WORK/'training_timing.json')
            assert timing['complete']
            expected={f'{arm}_seed{seed}' for arm in ('act','pact') for seed in SEEDS}
            assert set(timing['models'])==expected
            assert all(m['returncode']==0 and m['verified'] for m in timing['models'].values())
            run_stage('08_train_verify',[sys.executable,'scripts/train_pact_place_v1011c_experiment.py',
                '--stage','verify'],WORK/'training_verification.json')
            freeze(WORK/'stages/07_train.json',{**empty_authorization(),'stage':'07_train',
                'returncode':None,'artifact_completion_verified':True,
                'model_returncodes':{k:m['returncode'] for k,m in timing['models'].items()},
                'log':'diagnostics_output/pact_place_v1011c_train_eval/logs/07_train.log',
                'output_tail':'Original outer coordinator was lost during Codex crash. Trainer survived; all six models returned 0 and completed independent verification. Original coordinator exit code is unavailable.',
                'recovery_verification_receipt':'08_train_verify.json'})
            break
        try:
            os.kill(a.training_pid,0)
        except ProcessLookupError:
            freeze(WORK/'stages/07_train_unexpected_exit.json',{**empty_authorization(),
                'stage':'07_train_unexpected_exit','returncode':None,'failed':True,
                'log':'diagnostics_output/pact_place_v1011c_train_eval/logs/07_train.log',
                'output_tail':(WORK/'logs/07_train.log').read_text()[-6000:]+'\nTrainer process vanished without completion receipt. Exit code unavailable.'})
            subprocess.run([sys.executable,'scripts/finalize_pact_place_v1011c_eval.py','--failure-only'],cwd=ROOT,env=environment())
            raise SystemExit('Training process exited without a verified completion artifact.')
        time.sleep(60)
    py=sys.executable
    try:
        # The manifest was frozen while training ran, before any evaluation.
        manifest=load_eval_manifest(EVAL/'eval_manifest.json')
        freeze(WORK/'stages/09_eval_manifest.json',{**empty_authorization(),'stage':'09_eval_manifest',
            'returncode':0,'instances':len(manifest['rows']),'manifest_sha256':manifest['manifest_sha256'],
            'output_tail':'150 frozen instances, 50 per seed; disjoint initial/retry seeds; reserved smoke rows recorded.'})
        if (EVAL/'smoke_reconciliation.json').exists():
            reconciliation=read(EVAL/'smoke_reconciliation.json')
            assert reconciliation['infrastructure_healthy'] and reconciliation['rollouts_complete']==8
            print('10_eval_smoke: retained original exit 1; all 8 raw results passed separately recorded RGB-tolerance reconciliation.',flush=True)
        else:
            run_stage('10_eval_smoke',[py,'scripts/run_pact_place_v1011c_eval.py','--stage','smoke','--workers','4','--h5-only'],EVAL/'smoke_run.json')
        command=[py,'scripts/run_pact_place_v1011c_eval.py','--stage','full','--workers',str(a.workers),'--h5-only']
        if a.resume_full:
            command.append('--resume')
        run_stage('11_eval_full_resumed_01' if a.resume_full else '11_eval_full',command,EVAL/'full_run.json')
        run_stage('12_analysis',[py,'scripts/finalize_pact_place_v1011c_eval.py'],EVAL/'analysis.json')
    except BaseException:
        subprocess.run([py,'scripts/finalize_pact_place_v1011c_eval.py','--failure-only'],cwd=ROOT,env=environment())
        raise


if __name__ == '__main__':
    main()
