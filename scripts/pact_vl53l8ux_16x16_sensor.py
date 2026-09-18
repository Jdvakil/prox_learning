"""Idealized L8UX-inspired 16x16 acquisition with fixed 2x2 minimum pooling.

Only 256 zones, nominal 65-degree FoV, 3m dark range and advertised 60fps
capability are documented by ST presentations. Other settings are declared
simulation assumptions, not a validated L8UX hardware or firmware model.
"""
from __future__ import annotations

from types import MethodType
import numpy as np

# Interpret the portfolio's 65-degree FoV as a diagonal of a square pinhole.
# This conversion is an explicit modeling assumption, not an ST optical map.
SIDE_FOV_DEG = float(np.rad2deg(2*np.arctan(np.tan(np.deg2rad(65/2))/np.sqrt(2))))
PROFILE = dict(
    name='VL53L8UX_INSPIRED_IDEAL_16X16_MIN_TO_8X8_15HZ_V1',
    native_zones=256, native_shape=[16,16], policy_spatial_shape=[8,8],
    policy_spatial_values_per_sensor=64,
    reported_fov_deg=65., assumed_diagonal_fov_deg=65.,
    horizontal_fov_deg=SIDE_FOV_DEG, vertical_fov_deg=SIDE_FOV_DEG,
    min_range_m=.02, max_range_m=3., quantization_m=.001, native_frequency_hz=15,
    public_advertised_frame_rate_hz=60,
    source_status='ST CES 2024 demonstration and portfolio presentations; no public detailed L8UX datasheet/driver verified',
    documented_features=['16x16 / 256 zones', '65-degree nominal FoV (axis unspecified in cited portfolio)',
        '300cm dark ranging', '256-zone output at 60fps advertised in CES demo'],
    assumed_features=['65-degree FoV interpreted as diagonal of a square ideal pinhole',
        '15Hz simultaneous native captures selected for consistency with prior L7CX studies',
        '2cm floor and 1mm rounding retained as simulation conventions, not L8UX specifications',
        'optical-axis depth, ideal per-zone central rays, no noise/integration/readout latency',
        'all >3m returns encoded as 3m; below-range saturates rather than disappearing'],
    native_clock='initial frame at t0; first 2ms physics tick at/after t0+k/15',
    delivery='zero-order hold at the original four polls per 66ms interval, followed by original temporal min pooling',
    subframe_interpretation='held polls are not independent acquisitions',
    depth_conversion='optical-axis depth; modeling convention, not verified L8UX firmware behavior',
    zone_model='one ideal 16x16 render pixel per zone; no reflectance or histogram integration',
    adapter='nonoverlapping 2x2 spatial minimum after range encoding: native 16x16 -> policy 8x8',
    aggregation_design='minimum fixed before any new scientific outcomes; no performance-based operator selection',
    no_return='above-range becomes 3m; below 2cm saturates at 2cm; validity logged, not an extra policy channel',
    physical_parameter_intrinsics='native 16x16 pinhole at declared side FoV; frozen encoder calibration unchanged',
    integration_noise_latency='not modeled: instantaneous synchronous frames, no SPAD/firmware/noise/crosstalk/bus model',
    range_limit_qualification='3m advertised in darkness; fixed envelope does not model target-dependent detection',
    quantization_qualification='1mm rounding is an experimental assumption, not hardware accuracy',
    angular_qualification='65-degree diagonal is inferred from portfolio convention; square-pinhole conversion is assumed',
    timing_qualification='15Hz chosen below advertised 60fps capability; L8UX selectable modes/integration timing unverified',
    frozen_encoder_calibration='original preprocessing, ray calibration, eight-frame history, weights and normalization unchanged',
    compared_with='original PACT seed-3103 45-degree 8x8 stream, with ACT descriptive context; multiple profile parameters change',
    experiment_type='zero-shot idealized L8UX-inspired profile sensitivity, not physical sensor validation or a pure resolution ablation',
    sources=[
        'https://www.st.com/content/dam/static-page/events/ces-2024/23-immersive-ski-game.pdf',
        'https://www.st.com/content/dam/OLM%20Email%20Marketing/2024/Asia%20Pac/Event/industrial-summit-2024/track-3-automation/en/a3-03-imaging-3d-sensing-and-iot-solutions-approved.pdf',
    ],
    documentation='CES presentation pages 5-6; Industrial Summit 2024 page 6; consulted 2026-09-18',
)


def acquisition_tick(index):
    """Exact ceil((index/15)/.002), with no floating-point clock drift."""
    if not isinstance(index, (int, np.integer)) or index < 0:
        raise ValueError('nonnegative integer acquisition index required')
    return (100*int(index)+2)//3


def encode_zones(depth):
    """Bound/round each optical-axis zone independently, preserving spatial data."""
    depth = np.asarray(depth)
    if depth.shape != (16, 16) or not np.isfinite(depth).all() or (depth <= 0).any():
        raise ValueError('invalid fresh 16x16 depth render')
    below = depth < PROFILE['min_range_m']
    valid = (depth >= PROFILE['min_range_m']) & (depth <= PROFILE['max_range_m'])
    clipped = np.clip(depth.astype(np.float64), PROFILE['min_range_m'], PROFILE['max_range_m'])
    values = np.clip(np.rint(clipped/PROFILE['quantization_m'])*PROFILE['quantization_m'],
                     PROFILE['min_range_m'], PROFILE['max_range_m']).astype(np.float32)
    return values, valid, below


def policy_grid(native):
    """Each 2x2 block retains its nearest encoded range and its spatial identity."""
    native=np.asarray(native)
    if (native.shape[-2:] != (16,16) or not np.isfinite(native).all() or
        (native < np.float32(PROFILE['min_range_m'])).any() or
        (native > np.float32(PROFILE['max_range_m'])).any()):
        raise ValueError('expected finite encoded 16x16 distances within declared range')
    return native.reshape(*native.shape[:-2],8,2,8,2).min(axis=(-3,-1))


class MultizoneProfile:
    """Install instance-local render/clock hooks after original scenario sampling."""
    def __init__(self, task, names):
        import mujoco
        self.task, self.env, self.names = task, task.env, list(names)
        if len(self.names) != 40 or set(task._proximity_camera_names) != set(self.names):
            raise ValueError('sensor inventory changed')
        if (task._n_sim_steps_per_proximity != 8 or task._n_ctrl_steps_per_policy != 33 or
            task._n_sim_steps_per_ctrl != 1):
            raise ValueError('source control/polling cadence differs')
        self.model = self.env.current_model
        if abs(self.model.opt.timestep-.002) > 1e-12:
            raise ValueError('physics timestep changed')
        self.camera_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA,
                           self.env._proximity_cam_full_name(n)) for n in self.names]
        if min(self.camera_ids) < 0 or len(set(self.camera_ids)) != 40:
            raise ValueError('missing or duplicate physical sensor cameras')
        self.original_fovy = self.model.cam_fovy.copy()
        self.original_pos, self.original_quat = self.model.cam_pos.copy(), self.model.cam_quat.copy()
        self.original_registry = {n: self.env.camera_manager.registry[n].fov for n in self.names}
        self.original_record, self.original_step = self.env.record_proximity_depths, self.env.step
        self.original_param_sizes = {}
        for name in self.names:
            sensor = task.sensor_suite.sensors[name]
            if sensor.max_substeps != 4 or sensor.img_resolution != (8, 8):
                raise ValueError('source acquisition tensor changed')
            if task.sensor_suite.sensors['sensor_param_'+name].img_resolution != (8, 8):
                raise ValueError('physical 8x8 intrinsics missing')
            self.original_param_sizes[name] = (8, 8)
            task.sensor_suite.sensors['sensor_param_'+name].img_resolution = (16, 16)
        self.model.cam_fovy[self.camera_ids] = PROFILE['vertical_fov_deg']
        for name in self.names:
            self.env.camera_manager.registry[name].fov = PROFILE['vertical_fov_deg']
        self.renderer = mujoco.Renderer(self.model, height=16, width=16)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        self.renderer.enable_depth_rendering()
        self.env.record_proximity_depths = MethodType(lambda env, names: self.record(names), self.env)
        self.env.step = MethodType(lambda env, n_steps=1: self.step(n_steps), self.env)
        self.reset()

    def reset(self):
        self.env.reset_proximity_depth_buffer(self.names)
        self.samples = {n: [] for n in self.names}  # observation-poll time and native frame index
        self.native_times, self.native_depth, self.native_valid, self.native_below = [], [], [], []
        self.physics_ticks = self.consumed_checks = 0
        self.t0 = float(self.env.current_data.time)
        self.capture()

    def assert_optics(self):
        expected = self.original_fovy.copy()
        expected[self.camera_ids] = PROFILE['vertical_fov_deg']
        if (not np.array_equal(self.model.cam_fovy, expected) or
            not np.array_equal(self.model.cam_pos, self.original_pos) or
            not np.array_equal(self.model.cam_quat, self.original_quat)):
            raise ValueError('undeclared camera or mounting change')
        if any(self.env.camera_manager.registry[n].fov != PROFILE['vertical_fov_deg'] for n in self.names):
            raise ValueError('registry optics disagree with native renderer')

    def step(self, n_steps=1):
        if n_steps != 1:
            raise ValueError('expected one unchanged physics step per call')
        result = self.original_step(n_steps)
        self.physics_ticks += 1
        if abs(float(self.env.current_data.time)-self.t0-self.physics_ticks*.002) > 1e-8:
            raise ValueError('unaccounted physics time change')
        if self.physics_ticks == acquisition_tick(len(self.native_times)):
            self.capture()
        return result

    def capture(self):
        self.assert_optics()
        if self.physics_ticks != acquisition_tick(len(self.native_times)):
            raise ValueError('native acquisition outside the fixed 15Hz clock')
        frames, valid, below = [], [], []
        option = self.env._get_proximity_scene_option()
        previous = (self.model.vis.map.znear, self.model.vis.map.zfar)
        try:
            self.model.vis.map.znear = .0001/self.model.stat.extent
            self.model.vis.map.zfar = 4./self.model.stat.extent
            for name in self.names:
                self.renderer.update_scene(self.env.current_data,
                    camera=self.env._proximity_cam_full_name(name), scene_option=option)
                depth, good, close = encode_zones(self.renderer.render())
                frames.append(depth); valid.append(good); below.append(close)
        finally:
            self.model.vis.map.znear, self.model.vis.map.zfar = previous
        self.native_times.append(float(self.env.current_data.time))
        self.native_depth.append(np.stack(frames))
        self.native_valid.append(np.stack(valid))
        self.native_below.append(np.stack(below))

    def record(self, camera_names):
        """Poll the most recently acquired full grid; never manufacture a new one."""
        if not set(camera_names) <= set(self.names):
            raise ValueError('attempt to alter a non-proximity camera')
        timestamp = float(self.env.current_data.time)
        native_index = len(self.native_times)-1
        if self.native_times[native_index] > timestamp+1e-10:
            raise ValueError('future sensor sample')
        for name in camera_names:
            if self.samples[name] and timestamp <= self.samples[name][-1][0]:
                raise ValueError('duplicate observation poll')
            self.samples[name].append((timestamp, native_index))
            i = self.names.index(name)
            self.env._proximity_depth_frames.setdefault(name, []).append(policy_grid(self.native_depth[native_index][i]))

    def check_consumed(self, raw, step):
        if raw.shape != (40, 4, 8, 8) or step != self.consumed_checks:
            raise ValueError('policy input shape/history order drift')
        for i, name in enumerate(self.names):
            if len(self.samples[name]) != 1+4*step:
                raise ValueError('observation poll count drift')
            polls = self.samples[name][-4:] if step else self.samples[name]*4
            expected = policy_grid(np.stack([self.native_depth[index][i] for _, index in polls]))
            if not np.array_equal(raw[i], expected):
                raise ValueError('policy did not consume the declared held 16x16 acquisitions pooled to 8x8')
        self.consumed_checks += 1

    def audit_arrays(self, horizon):
        polls = np.asarray([self.samples[n] for n in self.names], dtype=np.float64)
        count = horizon*33*3//100+1
        if (polls.shape != (40, 1+4*horizon, 2) or len(self.native_times) != count or
            self.physics_ticks != 33*horizon or self.consumed_checks != horizon):
            raise ValueError('incomplete native acquisition/delivery audit')
        if not np.array_equal(polls, np.broadcast_to(polls[:1], polls.shape)):
            raise ValueError('unsynchronized sensor array')
        expected_poll = np.concatenate(([0.], (np.arange(horizon)[:, None]*.066+
                                               np.array([.016,.032,.048,.064])).ravel()))
        expected_native = np.array([acquisition_tick(i)*.002 for i in range(count)])
        if (not np.allclose(polls[0, :, 0]-self.t0, expected_poll, atol=1e-8, rtol=0) or
            not np.allclose(np.array(self.native_times)-self.t0, expected_native, atol=1e-8, rtol=0)):
            raise ValueError('sampling/polling cadence drift')
        poll_ticks = np.rint(expected_poll/.002).astype(int)
        latest = np.searchsorted([acquisition_tick(i) for i in range(count)], poll_ticks, side='right')-1
        if not np.array_equal(polls[0, :, 1].astype(int), latest):
            raise ValueError('stale or future native frame delivered')
        depths = np.asarray(self.native_depth)
        return polls, depths, dict(native_acquisitions_per_sensor=count,
            consumed_control_frames=self.consumed_checks, full_grid_delivery_checks=self.consumed_checks,
            native_zones_per_sensor=256, spatial_minimum_checks=self.consumed_checks,
            policy_spatial_values_per_sensor=64,
            sampling_clock_exact=True, observation_polls_per_sensor=len(expected_poll),
            held_poll_fraction=float(np.mean(np.diff(latest)==0)),
            spatially_nonconstant_sensor_frames=int(np.count_nonzero(np.ptp(depths, axis=(-1,-2))>0)),
            camera_mounts_unchanged=True, wrist_optics_unchanged=True)

    def save_audit(self, path):
        import h5py
        polls, depths, report = self.audit_arrays(900)
        with h5py.File(path, 'x') as h:
            h.create_dataset('native_sim_time_s', data=self.native_times)
            h.create_dataset('native_distance_m', data=depths, compression='gzip')
            h.create_dataset('valid_nominal_range', data=self.native_valid, compression='gzip')
            h.create_dataset('below_min_range', data=self.native_below, compression='gzip')
            h.create_dataset('observation_poll_sim_time_s', data=polls[0,:,0])
            h.create_dataset('observation_poll_native_index', data=polls[0,:,1].astype(np.int32))
            h.create_dataset('sensor_names', data=np.array(self.names, dtype='S'))
            h.attrs['native_zones_per_sensor'] = 256
            h.attrs['native_frequency_hz'] = 15
            h.attrs['policy_adapter'] = PROFILE['adapter']
            h.create_dataset('pooled_distance_m', data=policy_grid(depths), compression='gzip')
            h.attrs['terminal_data'] = 'last four polls and terminal native frame recorded, not queried by policy'
        return report

    def close(self):
        self.env.record_proximity_depths, self.env.step = self.original_record, self.original_step
        self.model.cam_fovy[:] = self.original_fovy
        for name, value in self.original_registry.items():
            self.env.camera_manager.registry[name].fov = value
        for name, value in self.original_param_sizes.items():
            self.task.sensor_suite.sensors['sensor_param_'+name].img_resolution = value
        self.renderer.close()


def compare_initial(reference_path, current_path, names):
    """Allow exactly proximity samples/intrinsics to change; inspect everything else."""
    import h5py
    from pact_v1010b_storage import names as h5_names, same_value
    from pact_wrist288_analysis import rgb_difference
    allowed_raw = {'observation/' + n for n in names}
    allowed_intrinsics = {'observation/sensor_param_' + n + '/intrinsic_cv' for n in names}
    detail = dict(exact_non_rgb=[], allowed_sensor_differences=[], allowed_projection_annotations=[], violations=[], rgb=None)
    with h5py.File(reference_path) as a, h5py.File(current_path) as b:
        na, nb = h5_names(a), h5_names(b)
        if na != nb:
            detail['violations'].append('dataset/group inventory')
        if not allowed_raw | allowed_intrinsics <= set(na) & set(nb):
            detail['violations'].append('missing sensor data or intrinsics')
        for name in ['', *sorted(set(na) & set(nb))]:
            av, bv = (a[name], b[name]) if name else (a, b)
            if set(av.attrs) != set(bv.attrs) or any(not same_value(av.attrs[k], bv.attrs[k]) for k in set(av.attrs) & set(bv.attrs)):
                detail['violations'].append('attributes/' + name)
            if not isinstance(av, h5py.Dataset):
                continue
            if av.shape != bv.shape or av.dtype != bv.dtype:
                detail['violations'].append('shape/dtype/' + name)
                continue
            x, y = av[()], bv[()]
            if name == 'observation/wrist_camera':
                detail['rgb'], _ = rgb_difference(x, y)
                if not detail['rgb']['passed']:
                    detail['violations'].append('wrist RGB tolerance')
            elif name in allowed_raw:
                if (y.shape != (4, 8, 8) or not np.isfinite(y).all() or
                    y.min() < PROFILE['min_range_m'] or y.max() > np.float32(PROFILE['max_range_m'])):
                    detail['violations'].append('invalid 8x8 multizone input/' + name)
                detail['allowed_sensor_differences'].append(name)
            elif name in allowed_intrinsics:
                focal = np.float32(8 / np.tan(np.deg2rad(PROFILE['vertical_fov_deg']/2)))
                expected = np.array([[focal, 0, 8], [0, focal, 8], [0, 0, 1]])
                if not np.array_equal(y, expected):
                    detail['violations'].append('incorrect native 16x16 intrinsics/' + name)
                detail['allowed_sensor_differences'].append(name)
            elif (len(name.split('/')) == 5 and name.startswith('observation/object_image_points/') and
                  name.split('/')[3] in names and name.split('/')[4] in ('points','num_points')):
                # These segmentation-derived diagnostics change with sensor-camera FoV.
                # The unchanged policy consumes only wrist RGB, qpos, and proximity.
                group = b[name.rsplit('/',1)[0]]
                points, count = group['points'][()], int(group['num_points'][()][0])
                if (not 0 <= count <= len(points) or not np.isfinite(points[:count]).all() or
                    (points[:count] < 0).any() or (points[:count] > 1).any() or
                    not np.isnan(points[count:]).all()):
                    detail['violations'].append('invalid diagnostic projection/' + name)
                detail['allowed_projection_annotations'].append(name)
            elif same_value(x, y):
                detail['exact_non_rgb'].append(name)
            else:
                detail['violations'].append(name)
    if detail['rgb'] is None or not any(n.startswith('physics/') for n in detail['exact_non_rgb']):
        detail['violations'].append('missing physical/RGB audit')
    detail['passed'] = not detail['violations']
    return detail
