import sys,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
from pact_v1010b_train import *
from pact_v1010b_run import resources,storage_gate,remaining_time_gate
am=B/'amendments/05_thread_resource_recovery';manifest=read(am/'recovery_manifest.json')
verify_contract(full=True)
path=Path(manifest['resume_path']);assert sha(path)==manifest['resume_sha256']
bundle=torch.load(path,map_location='cpu',weights_only=False)
assert original_training.validate_bundle(bundle,training_config_binding()['config_sha256'],61500)==61500
assert bundle['v1010b_contract_sha256']==manifest['contract_sha256'] and bundle['v1010b_sampling_p']==.25
fork=validate_fork(bundle,bundle['model_state'],61500)
assert set(bundle['rng_state'])=={'python','numpy','torch_cpu','torch_cuda'}
del bundle
act=stream_summary(B/'checkpoints/act_seed3105_acquisition63000/sample_stream',2000,2050)
pact=stream_summary(path.parent/'sample_stream',2000,2050);assert act==pact
check={'utc':now(),'resume_sha256':manifest['resume_sha256'],'validation':fork,'committed_prefix_samples':pact,
 'committed_prefix_matches_paired_ACT':True,'remaining_time_at10':remaining_time_gate('thread_recovery02',10),
 'storage':storage_gate('thread_recovery02'),'code_unchanged':True,'protected_hashes_unchanged':True}
samples=[]
for index in range(3):
 value=resources();value['pids_events_max']=int(Path('/sys/fs/cgroup/pids.events').read_text().split()[1]);samples.append(value)
 assert value['pid_fraction']<.75 and value['ram_fraction']<.8 and value['vram_fraction']<.85
 assert value['oom_kill']==3
 print(json.dumps({'settled_sample':index+1,'pid_fraction':value['pid_fraction'],'ram_fraction':value['ram_fraction'],'vram_fraction':value['vram_fraction']}),flush=True)
 if index<2:time.sleep(60)
assert len({r['pids_events_max'] for r in samples})==1
check['settled_resource_samples']=samples
freeze(am/'preflight.json',check)
