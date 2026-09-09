"""Actual trainer startup and backward smoke, intercepted before every optimizer update."""
import sys,ast,inspect
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
from pact_v1010b_contract import *
os.environ.update(environment());pin_torch()
import torch
import pact_wrist288_train as original
import imitate_episodes as trainer
from pact_v1010b_dataset import AcquisitionWindowDataset
arm=sys.argv[1];root=B/'amendments/04_effective280_training_binding'
directory=root/f'bootstrap_{arm}';directory.mkdir()
trainer.RESUME_BUNDLE=str(W/f'checkpoints/{arm}_seed3103/resume_bundle.ckpt')
labels=read(B/'training_labels.json')['rows']
for name in ('load_fixed_split_data','load_pact_fixed_split_data'):
 old=getattr(trainer,name)
 def loader(*a,_old=old,**kw):
  train,val,stats,meta=_old(*a,**kw)
  assert train.batch_size==8 and len(train)==30 and len(train.dataset)==240 and len(val.dataset)==40
  assert train.dataset.num_queries==val.dataset.num_queries==100
  object.__setattr__(train,'dataset',AcquisitionWindowDataset.wrap(train.dataset,labels,3103,0. if arm=='act' else .25,directory/'sample_stream',8,train.num_workers))
  return train,val,stats,meta
 setattr(trainer,name,loader)
trainer.V1010B_BOOTSTRAP_DIRECTORY=directory
trainer.V1010B_BOOTSTRAP_SOURCE=sha(__file__)
trainer.V1010B_FREEZE=freeze
source=inspect.getsource(trainer.train_bc)
needle='    stop = False\n    for epoch in tqdm(range(start_epoch, num_epochs)):'
assert source.count(needle)==1
block='''    assert start_epoch==2000 and global_step==60000
    def no_update(*a,**kw):
        raise AssertionError('Bootstrap must never execute an optimizer update')
    optimizer.step=no_update
    seed_epoch(train_dataloader,val_dataloader,seed,2000)
    train_dataloader.dataset.set_epoch(2000)
    policy.eval()
    with torch.inference_mode():
        validation=forward_pass(next(iter(val_dataloader)),policy)
    policy.train();optimizer.zero_grad()
    data=next(iter(train_dataloader));losses=forward_pass(data,policy)
    assert torch.isfinite(losses['loss']).all()
    losses['loss'].backward()
    grads=[p.grad for p in policy.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert {int(v['step']) for v in optimizer.state_dict()['state'].values() if 'step' in v}=={60000}
    V1010B_FREEZE(V1010B_BOOTSTRAP_DIRECTORY/'receipt.json',{
        'source_sha256':V1010B_BOOTSTRAP_SOURCE,'original_resume_path':resume_path,
        'optimizer_updates':0,'optimizer_step':60000,'actual_resume_provenance_checks_passed':True,
        'loss':float(losses['loss']),'validation_loss':float(validation['loss']),
        'finite_gradient_tensors':len(grads),'batch_shapes':[list(v.shape) for v in data],
        'sampling_p':train_dataloader.dataset.p})
    return (best_epoch,min_val_loss,{})
    stop = False
    for epoch in tqdm(range(start_epoch, num_epochs)):'''
source=source.replace(needle,block)
exec(compile(source,trainer.__file__,'exec'),trainer.__dict__)
cmd=original.command(arm,3103,60000)
for flag,value in [('--ckpt_dir',directory),('--num_epochs',2100),('--max_steps',63000)]:cmd[cmd.index(flag)+1]=str(value)
sys.argv=[__file__,*cmd[2:]]
tree=ast.parse(Path(trainer.__file__).read_text())
entry=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'")
exec(compile(ast.Module(body=entry.body,type_ignores=[]),trainer.__file__,'exec'),trainer.__dict__)
print(json.dumps(read(directory/'receipt.json')),flush=True)
