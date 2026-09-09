"""V10.11c pipeline, independently logged and fail-closed at every stage."""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pact_place_v1011c_experiment import *


def run_stage(name, command, marker):
    log = WORK / 'logs' / (name + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    receipt = WORK / 'stages' / (name + '.json')
    if receipt.exists():
        prior = read(receipt)
        assert (prior['returncode'] == 0 or prior.get('artifact_completion_verified') is True) and Path(marker).is_file(), prior
        print(name + ': prior verified stage receipt', flush=True)
        return
    if log.exists():
        raise RuntimeError(f'prior unclosed stage log exists: {log}')
    started = time.time()
    print(name + ': starting', flush=True)
    with log.open('x') as stream:
        proc = subprocess.run(command, cwd=ROOT,env=environment(),stdout=stream,stderr=subprocess.STDOUT)
    result = {**empty_authorization(), 'stage':name,'command':command,
              'returncode':proc.returncode,'elapsed_s':time.time()-started,
              'log':str(log.relative_to(ROOT)),'output_tail':log.read_text()[-5000:]}
    freeze(receipt,result)
    print(f"{name}: exit {proc.returncode}; {result['elapsed_s']/60:.1f} min",flush=True)
    if proc.returncode:
        print(result['output_tail'],flush=True)
        raise SystemExit(proc.returncode)
    assert Path(marker).is_file(), f'stage did not produce {marker}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--through',choices=('split','train'),default='split')
    args = parser.parse_args()
    py = sys.executable
    source = WORK / 'source_manifest.json'
    if not source.exists():
        run_stage('01_source',[py,'scripts/prepare_pact_place_v1011c_experiment.py','--stage','source'],source)
    else:
        doc = read(source)
        assert doc['verified'] and doc['counts']['accepted'] == 99
        print('01_source: existing verified source manifest, 99 episodes',flush=True)
    n = str(read(source)['counts']['accepted'])
    conv = WORK / 'conversion_manifest.json'
    run_stage('02_convert',[py,'scripts/convert_pact_place_v1011c_to_act.py','--source-manifest',str(source),
        '--dst',str(DATA),'--manifest-out',str(conv),'--expected-episodes',n,'--workers','4'],conv)
    raw = WORK / 'conversion_manifest_encoded_raw.json'
    report = WORK / 'embedding_report_raw.json'
    run_stage('03_encode',[py,'scripts/encode_pact_embedding_tokens.py','--dataset-dir',str(DATA),
        '--checkpoint',ENCODER_PATH,'--conversion-manifest',str(conv),
        '--updated-conversion-manifest-out',str(raw),'--report-out',str(report),'--batch-size','640'],raw)
    encoded = WORK / 'conversion_manifest_encoded.json'
    run_stage('04_embeddings',[py,'scripts/verify_pact_place_v109_embeddings.py','--dataset-dir',str(DATA),
        '--conversion-manifest',str(conv),'--raw-encoded-manifest',str(raw),'--raw-encoding-report',str(report),
        '--manifest-out',str(encoded),'--report-out',str(WORK / 'embedding_report.json'),
        '--expected-episodes',n,'--workers','4'],encoded)
    run_stage('05_split',[py,'scripts/prepare_pact_place_v1011c_experiment.py','--stage','split'],WORK / 'split_manifest.json')
    if args.through == 'train':
        run_stage('06_train_preflight',[py,'scripts/train_pact_place_v1011c_experiment.py','--stage','preflight'],WORK / 'training_preflight.json')
        run_stage('07_train',[py,'scripts/train_pact_place_v1011c_experiment.py','--stage','train'],WORK / 'training_timing.json')
        run_stage('08_train_verify',[py,'scripts/train_pact_place_v1011c_experiment.py','--stage','verify'],WORK / 'training_verification.json')


if __name__ == '__main__':
    main()
