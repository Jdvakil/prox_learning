"""V10.10 training math, with atomic update-counted resume and fixed snapshots."""
from __future__ import annotations
import ast
import inspect
import signal
import time
from pact_wrist288_common import *

def command(arm,seed,stop):
    from pact_place_v109_contract import training_command
    split=read(WORK/'split_manifest.json');data=read(WORK/'conversion_manifest.json')
    cmd=training_command(arm=arm,ckpt_dir=str(WORK/f'checkpoints/{arm}_seed{seed}'),
        dataset_dir=str(WORK/'converted'),split_manifest=str(WORK/'split_manifest.json'),
        dataset_manifest=str(WORK/'conversion_manifest.json'),expect_split_sha256=split['split_manifest_sha256'],
        expect_dataset_tree_sha256=data['converted_tree_file_sha256'])
    cmd[:2]=[sys.executable,str(CODE/'pact_wrist288_train.py')]
    for flag,value in [('--seed',seed),('--ckpt_every',100),('--episode_horizon',max(635,data['timesteps']['converted_t_max']+8))]:
        cmd[cmd.index(flag)+1]=str(value)
    cmd+=['--max_steps',str(stop)]
    if (WORK/f'checkpoints/{arm}_seed{seed}/resume_bundle.ckpt').exists(): cmd+=['--resume']
    return cmd

def validate_bundle(bundle,config_sha,expected=None):
    assert bundle['wrist288_config_sha256']==config_sha
    step=int(bundle['global_step'])
    assert step>0 and step%30==0 and bundle['epoch']+1==step//30
    if expected is not None: assert step==expected,(step,expected)
    optimizer_steps={int(v['step']) for v in bundle['optimizer_state']['state'].values() if 'step' in v}
    assert optimizer_steps=={step},optimizer_steps
    import torch
    assert all(torch.isfinite(v).all() for v in bundle['model_state'].values() if v.dtype.is_floating_point)
    return step

def main():
    os.environ.update(environment());pin_torch();frozen=check_bindings()
    import imitate_episodes as trainer
    import torch
    import numpy as np
    directory=Path(sys.argv[sys.argv.index('--ckpt_dir')+1])
    stop=int(sys.argv[sys.argv.index('--max_steps')+1])
    assert stop in (900,1800,30000,60000) and directory.is_relative_to(WORK/'checkpoints')
    directory.mkdir(parents=True,exist_ok=True)
    start_step=0
    if '--resume' in sys.argv:
        bundle=torch.load(directory/'resume_bundle.ckpt',map_location='cpu',weights_only=False)
        start_step=validate_bundle(bundle,frozen['config_sha256'])
        assert start_step<stop
        previous_logs=lines(directory/'epoch_log.jsonl')
        # Preserve post-checkpoint observations separately when a process died.
        excess=[r for r in previous_logs if r['global_step']>start_step]
        if excess:
            freeze(directory/f'uncommitted_epoch_log_{time.time_ns()}.json',excess)
            (directory/'epoch_log.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in previous_logs if r['global_step']<=start_step))
        del bundle
    else:
        assert not (directory/'epoch_log.jsonl').exists(),'existing training requires a verified resume bundle'
    started=time.time();calls=0;prox_calls=0
    original_optimizer=trainer.make_optimizer
    def optimizer_factory(*args,**kwargs):
        opt=original_optimizer(*args,**kwargs);step=opt.step
        def counted(*a,**kw):
            nonlocal calls
            result=step(*a,**kw);calls+=1;return result
        opt.step=counted;return opt
    trainer.make_optimizer=optimizer_factory
    original_policy=trainer.make_policy
    live_policy=None
    def policy_factory(*args,**kwargs):
        nonlocal live_policy
        policy=original_policy(*args,**kwargs);live_policy=policy
        projection=getattr(policy.model,'input_proj_proximity',None)
        if '--use_proximity' in sys.argv:
            assert projection is not None and tuple(projection.weight.shape)==(512,32)
            def hook(module,inputs,output):
                nonlocal prox_calls
                assert inputs[0].shape[-1]==32 and torch.isfinite(inputs[0]).all();prox_calls+=1
            projection.register_forward_hook(hook)
        else: assert projection is None
        return policy
    trainer.make_policy=policy_factory
    original_save=trainer._atomic_torch_save
    def save(value,path):
        target=Path(path)
        assert target.parent==directory
        if target.name.startswith('policy_epoch_'): return
        if target.name=='resume_bundle.ckpt':
            value['wrist288_config_sha256']=frozen['config_sha256']
            step=validate_bundle(value,frozen['config_sha256'])
            assert step==start_step+calls
            assert step%3000==0 or step==stop or trainer.WRIST_STOP_REQUESTED
            live_policy.load_state_dict(value['model_state'],strict=True)
            original_save(value,path)
            if step in (30000,60000): original_save(value['model_state'],directory/f'policy_update_{step}.ckpt')
            atomic(directory/'progress.json',{'utc':now(),'updates':step,'optimizer_calls_this_process':calls,
                'start_updates':start_step,'proximity_projection_calls':prox_calls,'elapsed_s':time.time()-started,
                'resume_sha256':sha(path),'config_sha256':frozen['config_sha256']})
            return
        original_save(value,path)
    trainer._atomic_torch_save=save
    trainer.WRIST_STOP_REQUESTED=False
    def request_stop(*args): trainer.WRIST_STOP_REQUESTED=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    source=inspect.getsource(trainer.train_bc)
    old='if epoch % ckpt_every == 0 or epoch == num_epochs - 1 or stop:'
    assert source.count(old)==1
    source=source.replace(old,'stop = stop or WRIST_STOP_REQUESTED\n        if global_step % 3000 == 0 or epoch == num_epochs - 1 or stop:')
    source=source.replace("            loss.backward()","            assert torch.isfinite(loss).all(), 'nonfinite training loss'\n            loss.backward()")
    exec(compile(source,trainer.__file__,'exec'),trainer.__dict__)
    # Plotting every bundle adds memory and I/O without changing training math.
    trainer.plot_history=lambda *args,**kwargs:None
    tree=ast.parse(Path(trainer.__file__).read_text())
    entry=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'")
    exec(compile(ast.Module(body=entry.body,type_ignores=[]),trainer.__file__,'exec'),trainer.__dict__)
    bundle=torch.load(directory/'resume_bundle.ckpt',map_location='cpu',weights_only=False)
    completed=validate_bundle(bundle,frozen['config_sha256'])
    assert completed==start_step+calls
    if not trainer.WRIST_STOP_REQUESTED: assert completed==stop
    import pickle
    shared=pickle.loads((WORK/'dataset_stats.pkl').read_bytes())
    actual=pickle.loads((directory/'dataset_stats.pkl').read_bytes())
    assert all(np.array_equal(actual[k],shared[k]) for k in shared)
    assert '--use_proximity' not in sys.argv or prox_calls>calls
    receipt={'utc':now(),'completed_updates':completed,'started_updates':start_step,'optimizer_calls':calls,
        'elapsed_s':time.time()-started,'update_per_second':calls/(time.time()-started),
        'stats_sha256':sha(directory/'dataset_stats.pkl'),'config_sha256':frozen['config_sha256'],
        'strict_reload':True,'proximity_projection_calls':prox_calls,'stopped_by_signal':trainer.WRIST_STOP_REQUESTED}
    freeze(directory/f'process_{start_step}_{completed}.json',receipt)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__': main()
