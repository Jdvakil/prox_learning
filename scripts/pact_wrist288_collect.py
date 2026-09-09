"""Isolated collection workers; historical recorder and physics are unchanged."""
from __future__ import annotations
import argparse
import ast
import inspect
import traceback
from pact_wrist288_common import *

def seed_for(row,index):
    s=row['sampling_seeds'][index]
    return {k:s[k] for k in ('seed_u32','seed_u64')}

def preflight(row):
    import pact_place_v1010_contract as contract
    import run_pact_place_v1010_preflight as old
    contract.build_row=lambda *args:row
    contract.cell_seed=lambda f,s,p,i:seed_for(row,0 if i==0 else i-1000)
    source=inspect.getsource(old.check_cell).replace('"/tmp/claude-0/v1010_preflight"',repr(str(WORK/'preflight/raw')))
    exec(compile(source,old.__file__,'exec'),old.__dict__)
    result=old.check_cell(tuple(row['cell'].split('|')))
    return result

def collect(row,directory):
    import run_pact_place_v108_collect as old
    import pact_place_v108_contract as old_contract
    import molmo_spaces.utils.save_utils as save_utils
    import molmo_spaces.data_generation.pipeline as pipeline
    old.SAMPLER_CLASS=SAMPLER_CLASS
    old_contract.cell_seed=lambda f,s,p,i:seed_for(row,i-int(row['attempt_index'])*1000)
    original_frames=save_utils.save_frames_to_mp4
    def bounded_frames(frames,path,fps,extra_kwargs=None):
        kwargs=dict(extra_kwargs or {})
        kwargs['output_params']=[*kwargs.get('output_params',[]),'-threads','1']
        return original_frames(frames,path,fps,extra_kwargs=kwargs)
    save_utils.save_frames_to_mp4=bounded_frames
    original_video=save_utils.save_videos_from_raw_observations
    def wrist_video(observations,*args,**kwargs):
        return original_video([{k:v for k,v in obs.items() if k=='wrist_camera'} for obs in observations],*args,**kwargs)
    save_utils.save_videos_from_raw_observations=wrist_video
    original_setup=pipeline.setup_policy
    def checked_setup(config,task,*args,**kwargs):
        params=task.scene_params
        assert config.task_sampler_config.task_sampler_class.__name__==SAMPLER_CLASS
        assert params['pact_place_environment_version']=='pact_place_corridor_v10_10_four_object'
        assert params['pact_v1010_active_clutter_count']==4
        assert params['pact_v1010_identity_sha256']==row['pact_v1010_identity_sha256']
        assert config.proximity_sensor_period_ms==16.6667
        assert [c.name for c in config.camera_config.cameras][0]=='wrist_camera'
        return original_setup(config,task,*args,**kwargs)
    pipeline.setup_policy=checked_setup
    result=old.run_attempt({'row':row,'row_dir':str(directory),'scene_xml':str(ROOT/row['pact_v1010_scene_relative'])})
    validation=old.validate_trainable(directory) if result.get('published') else {'passed':False}
    defects=old.row_defects(result)
    result.update(schema_validation=validation,accepted=not defects and validation['passed'],defects=defects)
    result['row']=row
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',type=Path,required=True)
    args=parser.parse_args();os.environ.update(environment());pin_torch()
    job=read(args.job);row=job['row'];directory=Path(job['directory'])
    assert directory.is_relative_to(WORK)
    directory.mkdir(parents=True,exist_ok=True)
    assert row['row_sha256']==digest({k:v for k,v in row.items() if k!='row_sha256'})
    try:
        result=preflight(row) if job['kind']=='preflight' else collect(row,directory)
        result.update(worker_pid=os.getpid(),finished_utc=now(),job_sha256=sha(args.job))
        freeze(directory/'worker_result.json',result)
        return 0 if result.get('status')!='infrastructure_failure' else 1
    except BaseException:
        freeze(directory/'worker_error.json',{'utc':now(),'error':traceback.format_exc(),'job_sha256':sha(args.job)})
        raise

if __name__=='__main__': raise SystemExit(main())
