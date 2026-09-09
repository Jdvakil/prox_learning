"""Update60000 ->63000 continuation using unchanged ACT training operations."""
from __future__ import annotations
import argparse
import ast
import copy
import inspect
import os
import pickle
import signal
import sys
import time
from pathlib import Path
from pact_v1010b_contract import *
os.environ.update(environment());pin_torch()
import numpy as np
import torch
import pact_wrist288_train as original_training
from pact_v1010b_dataset import AcquisitionWindowDataset,stream_summary


def validate_fork(bundle,model_state,expected=60000):
    assert int(bundle['global_step'])==expected and int(bundle['epoch'])+1==expected//30
    steps={int(v['step']) for v in bundle['optimizer_state']['state'].values() if 'step' in v}
    assert steps=={expected},steps
    assert set(bundle['model_state'])==set(model_state)
    for name,value in model_state.items():assert torch.equal(bundle['model_state'][name].cpu(),value.cpu()),name
    assert set(bundle['rng_state'])=={'python','numpy','torch_cpu','torch_cuda'}
    assert all(torch.isfinite(v).all() for v in model_state.values() if v.dtype.is_floating_point)
    return {'model_tensors_exact':len(model_state),'optimizer_steps':sorted(steps),
        'epoch':int(bundle['epoch']),'global_step':expected,'rng_keys':sorted(bundle['rng_state'])}


def safe_save(value,path):
    path=inside(path);temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temp)
    with temp.open('rb') as stream:os.fsync(stream.fileno())
    os.replace(temp,path)


def assert_nested_exact(left,right,path='state'):
    assert type(left)==type(right),path
    if isinstance(left,torch.Tensor):assert left.dtype==right.dtype and torch.equal(left.cpu(),right.cpu()),path
    elif isinstance(left,np.ndarray):assert left.dtype==right.dtype and np.array_equal(left,right),path
    elif isinstance(left,dict):
        assert left.keys()==right.keys(),path
        for key in left:assert_nested_exact(left[key],right[key],path+'/'+str(key))
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right),path
        for index,(a,b) in enumerate(zip(left,right)):assert_nested_exact(a,b,path+'/'+str(index))
    else:assert left==right,path


def recover_startup_bundle(bundle,resume,frozen,contract,p,directory):
    recovery=read(B/'training_recovery.json')
    assert sha(B/'training_recovery.json')==contract['input_hashes'][str(B/'training_recovery.json')]
    assert recovery['resume_path']==str(resume) and recovery['resume_sha256']==sha(resume)
    assert recovery['allowed_attempt']==1 and recovery['new_optimizer_updates_attempted']==0
    assert bundle['v1010b_contract_sha256']==recovery['prior_contract_sha256'] and bundle['v1010b_sampling_p']==p
    original=torch.load(frozen['resume_path'],map_location='cpu',weights_only=False)
    assert int(bundle['global_step'])==recovery['expected_step']==60000
    for key in ('model_state','optimizer_state','rng_state','provenance'):
        assert_nested_exact(bundle[key],original[key],key)
    del original
    freeze(directory/'resume_recovery_verification.json',{'utc':now(),'prior_resume_sha256':sha(resume),
        'source_resume_sha256':frozen['resume_sha256'],'model_optimizer_rng_provenance_exact':True,
        'step':60000,'new_updates_before_retry':0,'recovery_manifest_sha256':sha(B/'training_recovery.json'),
        'prior_contract_sha256':bundle['v1010b_contract_sha256'],'contract_sha256':contract['contract_sha256']})
    freeze(directory/'fork_verification.json',dict(validate_fork(bundle,bundle['model_state']),
        source_model_sha256=frozen['sha256'],source_bundle_sha256=frozen['resume_sha256'],
        contract_sha256=contract['contract_sha256'],sampling_p=p,recovered_startup_attempt=True,
        original_failed_fork_receipt_sha256=recovery['archive_hashes'][str(Path(recovery['prior_archive'])/'fork_verification.json')]))
    bundle['v1010b_contract_sha256']=contract['contract_sha256'];safe_save(bundle,resume)


def run(args):
    require_stage('training')
    assert args.max_updates==63000
    contract=read(B/'contract.json');directory=inside(args.output_dir)
    binding=training_config_binding();assert binding==contract['training']['source_config']
    config_sha=binding['config_sha256']
    expected=B/f'checkpoints/{args.arm.lower()}_seed{args.training_seed}_{args.variant}'
    assert directory==expected.resolve();directory.mkdir(parents=True,exist_ok=True)
    frozen=contract['models'][f'{args.arm.lower()}_seed{args.training_seed}']
    assert sha(frozen['path'])==frozen['sha256'] and sha(frozen['resume_path'])==frozen['resume_sha256']
    p=0. if args.variant=='uniform63000' else .25
    resume=directory/'resume_bundle.ckpt'
    if not resume.exists():
        assert not (directory/'completed.json').exists()
        bundle=torch.load(frozen['resume_path'],map_location='cpu',weights_only=False)
        model=torch.load(frozen['path'],map_location='cpu',weights_only=False)
        validation=validate_fork(bundle,model)
        freeze(directory/'fork_verification.json',dict(validation,source_model_sha256=frozen['sha256'],
            source_bundle_sha256=frozen['resume_sha256'],contract_sha256=contract['contract_sha256'],sampling_p=p,
            temporary_bundle_note='Best-model copy and plotting histories omitted; model/optimizer/RNG and loss remain exact.'))
        # Unused best checkpoints and plot histories do not influence training operations.
        bundle['best_model_state']={};bundle['train_history']=[];bundle['train_epoch_history']=[];bundle['validation_history']=[]
        bundle['v1010b_contract_sha256']=contract['contract_sha256'];bundle['v1010b_sampling_p']=p
        safe_save(bundle,resume);del bundle,model
    bundle=torch.load(resume,map_location='cpu',weights_only=False)
    if bundle.get('v1010b_contract_sha256')!=contract['contract_sha256']:
        recover_startup_bundle(bundle,resume,frozen,contract,p,directory)
    start=original_training.validate_bundle(bundle,config_sha)
    assert 60000<=start<63000 and bundle['v1010b_contract_sha256']==contract['contract_sha256'] and bundle['v1010b_sampling_p']==p
    # Post-checkpoint records from a killed process remain available in a unique archive.
    if start>60000 or (directory/'sample_stream').exists():
        excess=[path for path in (directory/'sample_stream').glob('epoch_*_worker_*.jsonl') if int(path.name.split('_')[1])>=start//30]
        if excess:
            archive=directory/f'uncommitted_stream_{time.time_ns()}';archive.mkdir()
            for path in excess:path.rename(archive/path.name)
        log=directory/'epoch_log.jsonl';oldlogs=lines(log)
        if any(r['global_step']>start for r in oldlogs):
            freeze(directory/f'uncommitted_epoch_log_{time.time_ns()}.json',oldlogs)
            log.write_text(''.join(json.dumps(r)+'\n' for r in oldlogs if r['global_step']<=start))
    del bundle
    import imitate_episodes as trainer
    calls=0;started=time.monotonic();live_policy=None
    old_optimizer=trainer.make_optimizer
    def optimizer_factory(*a,**kw):
        nonlocal calls
        optimizer=old_optimizer(*a,**kw);step=optimizer.step
        def counted(*aa,**kk):
            nonlocal calls
            result=step(*aa,**kk);calls+=1;return result
        optimizer.step=counted;return optimizer
    trainer.make_optimizer=optimizer_factory
    old_policy=trainer.make_policy
    def policy_factory(*a,**kw):
        nonlocal live_policy
        live_policy=old_policy(*a,**kw)
        projection=getattr(live_policy.model,'input_proj_proximity',None)
        assert (projection is not None)==(args.arm=='PACT')
        if projection is not None:assert tuple(projection.weight.shape)==(512,32)
        return live_policy
    trainer.make_policy=policy_factory
    labels=read(B/'training_labels.json')['rows']
    for name in ('load_fixed_split_data','load_pact_fixed_split_data'):
        old_loader=getattr(trainer,name)
        def loader(*a,_original=old_loader,**kw):
            train,val,stats,meta=_original(*a,**kw)
            assert train.batch_size==8 and len(train)==30 and len(train.dataset)==240 and len(val.dataset)==40
            assert train.dataset.num_queries==100 and val.dataset.num_queries==100
            wrapped=AcquisitionWindowDataset.wrap(train.dataset,labels,args.training_seed,p,directory/'sample_stream',8,train.num_workers)
            object.__setattr__(train,'dataset',wrapped)
            return train,val,stats,meta
        setattr(trainer,name,loader)
    old_seed_epoch=trainer.seed_epoch
    def seed_epoch(train,val,seed,epoch):
        old_seed_epoch(train,val,seed,epoch);train.dataset.set_epoch(epoch)
    trainer.seed_epoch=seed_epoch
    trainer.V1010B_STOP_REQUESTED=False
    def stop(*_):trainer.V1010B_STOP_REQUESTED=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    old_save=trainer._atomic_torch_save
    def save(value,path):
        path=inside(path);assert path.parent==directory
        if path.name!='resume_bundle.ckpt':return
        value['wrist288_config_sha256']=config_sha
        step=original_training.validate_bundle(value,value['wrist288_config_sha256'])
        assert step==start+calls and (step%300==0 or trainer.V1010B_STOP_REQUESTED)
        value['best_model_state']={};value['train_history']=[]
        value['v1010b_contract_sha256']=contract['contract_sha256'];value['v1010b_sampling_p']=p
        live_policy.load_state_dict(value['model_state'],strict=True)
        safe_save(value,path)
        if step==63000:safe_save(value['model_state'],directory/'policy_update_63000.ckpt')
        atomic(directory/'progress.json',{'utc':now(),'updates':step,'started_updates':start,'calls':calls,
            'resume_sha256':sha(path),'elapsed_s':time.monotonic()-started})
    trainer._atomic_torch_save=save;trainer.plot_history=lambda *a,**kw:None
    source=inspect.getsource(trainer.train_bc)
    old='if epoch % ckpt_every == 0 or epoch == num_epochs - 1 or stop:'
    assert source.count(old)==1
    source=source.replace(old,'stop = stop or V1010B_STOP_REQUESTED\n        if global_step % 300 == 0 or epoch == num_epochs - 1 or stop:')
    source=source.replace('            loss.backward()',"            assert torch.isfinite(loss).all(), 'nonfinite training loss'\n            loss.backward()")
    source=source.replace('best_state_dict = deepcopy(policy.state_dict())','best_state_dict = {}')
    exec(compile(source,trainer.__file__,'exec'),trainer.__dict__)
    cmd=original_training.command(args.arm.lower(),args.training_seed,60000)
    for flag,value in [('--ckpt_dir',directory),('--num_epochs',2100),('--max_steps',63000),('--ckpt_every',10)]:
        cmd[cmd.index(flag)+1]=str(value)
    if '--resume' not in cmd:cmd.append('--resume')
    sys.argv=[str(Path(__file__).resolve()),*cmd[2:]]
    tree=ast.parse(Path(trainer.__file__).read_text())
    entry=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'")
    exec(compile(ast.Module(body=entry.body,type_ignores=[]),trainer.__file__,'exec'),trainer.__dict__)
    completed=torch.load(resume,map_location='cpu',weights_only=False)
    end=original_training.validate_bundle(completed,config_sha)
    assert end==start+calls
    if trainer.V1010B_STOP_REQUESTED:raise RuntimeError(f'training interrupted at{end}; resumable bundle preserved')
    assert end==63000
    strict=torch.load(directory/'policy_update_63000.ckpt',map_location='cpu',weights_only=False)
    verification=validate_fork(completed,strict,63000);live_policy.load_state_dict(strict,strict=True)
    shared=pickle.loads((W/'dataset_stats.pkl').read_bytes());actual=pickle.loads((directory/'dataset_stats.pkl').read_bytes())
    assert all(np.array_equal(shared[k],actual[k]) for k in shared)
    assert sha(directory/'dataset_stats.pkl')==sha(W/'dataset_stats.pkl')
    stream=stream_summary(directory/'sample_stream')
    freeze(directory/'sample_stream_hash.json',stream)
    freeze(directory/'completed.json',{'schema':SCHEMA,'arm':args.arm,'training_seed':args.training_seed,'variant':args.variant,
        'start_updates':60000,'completed_updates':63000,'new_updates':3000,'calls_this_process':calls,'started_this_process':start,
        'checkpoint_sha256':sha(directory/'policy_update_63000.ckpt'),'optimizer_rng_verification':verification,
        'resume_sha256':sha(resume),'stats_sha256':sha(directory/'dataset_stats.pkl'),'strict_reload':True,
        'sampling':stream,'elapsed_s':time.monotonic()-started,'temporary_bundle':str(resume)})
    print(json.dumps(read(directory/'completed.json')),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('ACT','PACT'),required=True)
    p.add_argument('--training-seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--variant',choices=VARIANTS[1:],required=True);p.add_argument('--max-updates',type=int,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args())

if __name__=='__main__':main()
