"""Native spatial detail, fixed adapter, scene isolation and scientific accounting."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import pact_v1010c_vl53l8ux_16x16_128d as study
from pact_vl53l8ux_16x16_sensor import (
    PROFILE, SIDE_FOV_DEG, acquisition_tick, encode_zones, policy_grid,
    MultizoneProfile, compare_initial,
)


def test_pooling_preserves_block_identity_and_nearest_return():
    native=np.full((16,16),2.,dtype=np.float32)
    expected=np.arange(64,dtype=np.float32).reshape(8,8)/100+.1
    for y in range(8):
        for x in range(8):
            # Change which of the four source cells is nearest across blocks.
            native[2*y+y%2,2*x+x%2]=expected[y,x]
    assert np.array_equal(policy_grid(native),expected)
    assert np.array_equal(policy_grid(np.stack([native,native])),np.stack([expected,expected]))
    assert len(np.unique(policy_grid(native)))==64
    for invalid in (np.ones((8,8)),np.full((16,16),np.nan),np.zeros((16,16)),np.full((16,16),4.)):
        with pytest.raises(ValueError): policy_grid(invalid)


def test_range_floor_and_far_returns_remain_distinct_through_pooling():
    depth=np.full((16,16),4.)
    depth[1,1]=.005
    depth[0,2]=.1244
    encoded,valid,below=encode_zones(depth)
    pooled=policy_grid(encoded)
    assert pooled[0,0]==np.float32(.02) and pooled[0,1]==np.float32(.124)
    assert pooled[7,7]==np.float32(3.)
    assert below[1,1] and not valid[1,1] and not valid[15,15]
    assert encoded.shape==(16,16) and encoded.dtype==np.float32
    assert np.array_equal(encode_zones(np.full((16,16),.1234))[0],np.full((16,16),.123,dtype=np.float32))
    for invalid in (np.zeros((16,16)),np.full((16,16),np.inf),np.ones((8,8))):
        with pytest.raises(ValueError): encode_zones(invalid)


def test_declared_optical_assumption_reconstructs_diagonal_and_preserves_rate():
    diagonal=np.rad2deg(2*np.arctan(np.sqrt(2)*np.tan(np.deg2rad(SIDE_FOV_DEG/2))))
    assert diagonal==pytest.approx(65.)
    assert PROFILE['native_shape']==[16,16] and PROFILE['native_zones']==256
    assert PROFILE['native_frequency_hz']==15 and PROFILE['public_advertised_frame_rate_hz']==60
    assert len(PROFILE['documented_features'])==4 and len(PROFILE['assumed_features'])==5


def test_clock_is_causal_and_does_not_accumulate_rounding_drift():
    ticks=np.array([acquisition_tick(i) for i in range(893)])
    assert ticks[:7].tolist()==[0,34,67,100,134,167,200]
    assert ticks[891]==29700 and ticks[892]>29700
    lag=ticks*.002-np.arange(len(ticks))/15
    assert lag.min()>-1e-12 and lag.max()<.002
    for invalid in (-1,1.5):
        with pytest.raises(ValueError): acquisition_tick(invalid)


def test_actual_native_16x16_sees_thin_obstacle_missed_by_source_8x8():
    os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco
    names=[f'sensor_{i}' for i in range(40)]
    cameras=''.join(f'<camera name="{n}" pos="0 0 0" fovy="45"/>' for n in names)
    halfwidth=.5*np.tan(np.deg2rad(SIDE_FOV_DEG/2))
    x,y=halfwidth*3/16,halfwidth*5/16
    model=mujoco.MjModel.from_xml_string('<mujoco><option timestep=".002"/><worldbody>'+cameras+
        '<camera name="wrist" pos="0 0 1" fovy="56"/>'+
        '<geom name="wall" type="box" pos="0 0 -1.01" size="2 2 .01"/>'+
        f'<geom name="thin_hazard" type="box" pos="{x} {y} -.51" size=".003 .003 .01"/>'+
        '</worldbody></mujoco>')
    data=mujoco.MjData(model); mujoco.mj_forward(model,data)
    env=SimpleNamespace(current_model=model,current_data=data,camera_manager=SimpleNamespace(
        registry={n:SimpleNamespace(fov=45) for n in names}),_proximity_depth_frames={})
    env._proximity_cam_full_name=lambda n:n
    env.record_proximity_depths=lambda ns:None
    env.step=lambda n=1:mujoco.mj_step(model,data,n)
    env.reset_proximity_depth_buffer=lambda ns:setattr(env,'_proximity_depth_frames',{n:[] for n in ns})
    option=mujoco.MjvOption(); option.geomgroup[2]=0
    env._get_proximity_scene_option=lambda:option
    sensors={n:SimpleNamespace(max_substeps=4,img_resolution=(8,8),camera_name=n) for n in names}
    sensors.update({'sensor_param_'+n:SimpleNamespace(img_resolution=(8,8),camera_name=n) for n in names})
    task=SimpleNamespace(env=env,_proximity_camera_names=names,_n_sim_steps_per_proximity=8,
        _n_ctrl_steps_per_policy=33,_n_sim_steps_per_ctrl=1,sensor_suite=SimpleNamespace(sensors=sensors))
    renderer=mujoco.Renderer(model,height=8,width=8); renderer.enable_depth_rendering()
    renderer.update_scene(data,camera=names[0]); source=renderer.render().copy(); renderer.close()
    geometry=model.geom_pos.copy(); clips=(model.vis.map.znear,model.vis.map.zfar)
    original_fovy=model.cam_fovy.copy()
    profile=MultizoneProfile(task,names)
    try:
        native=profile.native_depth[0][0]
        assert source.min()>.9 and native.min()==pytest.approx(.5,abs=.001)
        assert native.shape==(16,16) and np.ptp(native)>.4
        assert policy_grid(native).min()==pytest.approx(.5,abs=.001)
        env.record_proximity_depths(names)
        raw=np.stack([np.repeat(np.array(env._proximity_depth_frames[n]),4,axis=0) for n in names])
        profile.check_consumed(raw,0)
        for control in range(1,4):
            env.reset_proximity_depth_buffer(names)
            for tick in range(1,34):
                env.step()
                if tick%8==0: env.record_proximity_depths(names)
            raw=np.stack([env._proximity_depth_frames[n] for n in names])
            if control==1: assert len(profile.native_times)==1
            if control==2:
                assert len(profile.native_times)==2 and profile.native_times[-1]==pytest.approx(.068)
                assert [r[1] for r in profile.samples[names[0]][-4:]]==[1]*4
            if control<3: profile.check_consumed(raw,control)
        _,_,audit=profile.audit_arrays(3)
        assert audit['native_acquisitions_per_sensor']==3 and audit['spatial_minimum_checks']==3
        assert audit['native_zones_per_sensor']==256 and audit['policy_spatial_values_per_sensor']==64
        assert np.array_equal(geometry,model.geom_pos) and model.cam_fovy[-1]==56
        assert clips==(model.vis.map.znear,model.vis.map.zfar)
        assert all(s.img_resolution==((16,16) if n.startswith('sensor_param_') else (8,8)) for n,s in sensors.items())
        assert data.time==pytest.approx(.198)
        corrupt=raw.copy(); corrupt[0,0,0,0]=.2
        with pytest.raises(ValueError,match='declared held'): profile.check_consumed(corrupt,3)
    finally: profile.close()
    assert np.array_equal(model.cam_fovy,original_fovy)
    assert all(s.img_resolution==(8,8) for s in sensors.values())


def test_pairing_rejects_physical_camera_and_intrinsic_drift(tmp_path):
    import h5py
    import pact_v1010c_core  # establish original dependency paths
    a,b=tmp_path/'a.h5',tmp_path/'b.h5'
    focal=float(np.float32(8/np.tan(np.deg2rad(SIDE_FOV_DEG/2))))
    for path in (a,b):
        with h5py.File(path,'w') as h:
            h['observation/wrist_camera']=np.full((240,320,3),127,dtype=np.uint8)
            h['observation/s']=np.full((4,8,8),.2,dtype=np.float32)
            h['observation/sensor_param_s/intrinsic_cv']=np.array([[focal,0,8],[0,focal,8],[0,0,1]])
            h['observation/sensor_param_s/extrinsic_cv']=np.zeros((3,4))
            h['physics/qpos']=np.zeros(9)
            h['configuration/row']=np.bytes_('same-scene')
    assert compare_initial(a,b,['s'])['passed']
    with h5py.File(b,'a') as h: h['observation/s'][0,0,0]=4.
    assert not compare_initial(a,b,['s'])['passed']
    with h5py.File(b,'a') as h:
        h['observation/s'][0,0,0]=.2; h['physics/qpos'][0]=.1
    assert 'physics/qpos' in compare_initial(a,b,['s'])['violations']
    with h5py.File(b,'a') as h:
        h['physics/qpos'][0]=0; h['observation/sensor_param_s/intrinsic_cv'][0,0]=99.
    assert any('intrinsic' in v for v in compare_initial(a,b,['s'])['violations'])


def test_nonsignificant_difference_does_not_establish_retention():
    source=[1]*24+[0]*26
    assert study.retention_statistics(source,source)['decision']=='RETENTION_SUPPORTED'
    changed=source.copy(); changed[0]=changed[1]=0
    result=study.retention_statistics(source,changed)
    assert result['exact_mcnemar_p']>.05 and result['decision']=='INCONCLUSIVE_RETENTION'
    assert study.retention_statistics(source,[0]*50)['decision']=='LOSS_EXCEEDS_MARGIN'


@pytest.mark.parametrize('transient_first',[False,True])
def test_scientific_failure_is_not_retried_and_technical_retry_preserves_scene(tmp_path,monkeypatch,transient_first):
    monkeypatch.setattr(study,'OUT',tmp_path/'study'); monkeypatch.setattr(study.base,'OUT',study.OUT)
    study.OUT.mkdir()
    study.freeze(study.OUT/'source_manifest.json',study.self_hashed({'scenes':[dict(index=0,scene_id='scene-0')]}))
    study.freeze(study.OUT/'sensor_profile.json',study.self_hashed(PROFILE))
    monkeypatch.setattr(study,'environment',lambda:{})
    monkeypatch.setattr(study,'resources',lambda:dict(ram_fraction=.1,pid_fraction=.1,vram_fraction=.1,
        disk_free_bytes=20*2**30,oom_kill=0))
    monkeypatch.setattr(study.time,'sleep',lambda seconds:None)
    started=[]
    class FakeProcess:
        def __init__(self,command,**kwargs):
            job=study.read(command[-1]); directory=Path(job['directory'])
            self.pid=10000+len(started); self.returncode=1 if transient_first and not started else 0
            started.append(job)
            if self.returncode: study.freeze(directory/'failure.json',{'retryable':True,'message':'test transient IO'})
            else: study.freeze(directory/'result.json',dict(status='policy_failure',task_success=False,
                collision_free_task_success=False,schedule_row_sha256=job['sha256']))
        def poll(self): return self.returncode
    monkeypatch.setattr(study.subprocess,'Popen',FakeProcess)
    request=[dict(scene_index=0,arm=study.ARMS[0])]
    study.run_jobs(request,1,'test_protocol'); study.run_jobs(request,1,'test_protocol')
    assert len(study.completed_jobs())==1 and len(started)==(2 if transient_first else 1)
    if transient_first:
        assert started[0]['sensor_profile_sha256']==started[1]['sensor_profile_sha256']
        assert started[0]['scene_id']==started[1]['scene_id']
    for job in started: assert (Path(job['directory'])/'exit_receipt.json').exists()
