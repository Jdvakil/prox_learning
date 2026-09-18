# Native 16×16 proximity sensing: code and collection handoff

The sensor implementation used in the completed seed-3103 experiment is
[`scripts/pact_vl53l8ux_16x16_sensor.py`](../scripts/pact_vl53l8ux_16x16_sensor.py).
It renders **40 independent 16×16 depth grids** in MuJoCo. The original policy
still receives **8×8 grids**, obtained by taking the minimum in each 2×2 block.
For a dataset containing all 256 zones per sensor, save the native stream
described below, rather than relying on the ordinary trajectory's proximity
fields, which remain 8×8.

This is an **idealized VL53L8UX-inspired simulation profile**. It is not an ST
hardware driver. The original experiment implementation is published unchanged;
the collection recipe in this document is a proposed integration into an expert
collector, not a previously completed native-16×16 demonstration collection.

## Files to use

| File | Purpose |
|---|---|
| [`pact_vl53l8ux_16x16_sensor.py`](../scripts/pact_vl53l8ux_16x16_sensor.py) | Sensor replacement, native capture, fixed spatial adapter, cleanup |
| [`pact_v1010c_vl53l8ux_16x16_128d_eval.py`](../scripts/pact_v1010c_vl53l8ux_16x16_128d_eval.py) | Executed integration: `run()`, `TransferPolicy.reset()`, and `get_info()` |
| [`pact_v1010c_vl53l8ux_16x16_128d.py`](../scripts/pact_v1010c_vl53l8ux_16x16_128d.py) | Historical 50-scene evaluation, preflight, pairing, analysis |
| [`pact_v1010c_permuted_128d.py`](../scripts/pact_v1010c_permuted_128d.py) | Shared provenance/contact/statistics helpers imported by that evaluator |
| [`pact_v1010c_vl53l8ux_16x16_128d_watch.py`](../scripts/pact_v1010c_vl53l8ux_16x16_128d_watch.py) | Optional evaluation health monitor |
| [`test_pact_v1010c_vl53l8ux_16x16_128d.py`](../tests/test_pact_v1010c_vl53l8ux_16x16_128d.py) | Rendering, resolution, range, clock, scene isolation, and evaluation-accounting tests |
| [`PACT_V1010C_VL53L8UX_16X16_128D_3103.md`](PACT_V1010C_VL53L8UX_16X16_128D_3103.md) | Full experimental protocol and source-specification qualifications |

The sensor module can be imported by a collector without importing the PACT
model or the historical evaluation orchestrator. Its capture path uses NumPy
and MuJoCo; HDF5 export uses h5py. Use the repository's compatible MolmoSpaces
environment and assets. The executed submodule revisions are
`molmospaces: 70dedc07f34ed7f8335aed7f694ddef7ef823d3d` and
`act: f4f59d7975e7d1d52403df92ee8a789fbc3e14c3`.

From a checkout of branch `experiment/pact-valid-ablation-followup-v1`:

```bash
git submodule update --init submodules/molmospaces submodules/act
PYTHONPATH=scripts python -c 'from pact_vl53l8ux_16x16_sensor import PROFILE; print(PROFILE)'
```

Submodule access and the project's simulator dependencies are required for the
full collector/evaluator. The historical evaluation additionally requires the
original checkpoint pair, normalization, source rollouts, and metadata at its
recorded paths; these large artifacts are not provided by this code publication.
Its `PYTHON` constant refers to the original machine's virtual environment, and
shared evaluation code also binds `/root/prox_learning_pact_remediation` as its
project root. Those historical paths and scientific hashes were deliberately
preserved; a different evaluation installation needs an explicit path adaptation.
The sensor module itself has no checkpoint or absolute interpreter dependency.

## Sensor settings used

| Setting | Value |
|---|---|
| Sensor count and mounts | Original 40 sensors; positions/orientations unchanged |
| Native resolution | 16×16 (256 ideal render pixels per sensor) |
| Native acquisition | 15 Hz; an initial capture followed by causal physics-clock captures |
| Field of view | 48.5009095° per axis, assuming the advertised 65° is a square diagonal |
| Encoded distances | Optical-axis depth in meters; 0.02–3 m; 1 mm rounding |
| Below/above range | Saturate to 0.02/3 m; validity and below-range masks retained |
| Frozen-policy adapter | Nonoverlapping 2×2 spatial minimum, yielding 8×8 |

ST's [CES presentation](https://www.st.com/content/dam/static-page/events/ces-2024/23-immersive-ski-game.pdf)
and [Industrial Summit presentation](https://www.st.com/content/dam/OLM%20Email%20Marketing/2024/Asia%20Pac/Event/industrial-summit-2024/track-3-automation/en/a3-03-imaging-3d-sensing-and-iot-solutions-approved.pdf)
support the 256-zone resolution, nominal 65° FoV, and advertised 3 m dark range;
the CES demonstration advertises 60 fps capability. The experiment chose 15 Hz.
The diagonal interpretation, 2 cm floor, rounding, ideal rays, and distance
convention are modeling assumptions. Noise, integration, reflectance, crosstalk,
and firmware behavior are not simulated. Preserve this distinction in dataset
metadata and paper descriptions.

## Integrating it into demonstration collection

The existing v1010 expert collection entry point is
[`run_pact_place_v1010_collect.py::worker`](../scripts/run_pact_place_v1010_collect.py),
which delegates to
[`run_pact_place_v108_collect.py::run_attempt`](../scripts/run_pact_place_v108_collect.py).
That shared runner also serves older datasets. Add a separate opt-in collector
or configuration for the new dataset; importing this module alone does not
change the current collector.

1. Sample/settle the original task and prepare/register the expert as usual.
   Retain the original ordered list of 40 sensor names. Do not sort it anew.
2. Construct `profile = MultizoneProfile(task, sensor_names)` after sampling and
   before the first recorded `task.reset()`. Construction installs hooks on this
   task's environment instance. Do not share a profile between tasks/workers.
3. Reset the profile **at the end of the registered expert policy's `reset()`**,
   after its original reset/planning has finished and before the task records
   its initial observation. Follow the ordering in `TransferPolicy.reset()`.
   This also clears the capture made during profile construction. Every new
   episode needs a fresh sensor clock and native buffer.
4. Pass the resulting `initial_reset_result` to
   `ParallelRolloutRunner.run_single_rollout()` so it does not reset again.
   Keep the existing expert, action recording, RGB recording, scene metadata,
   and collection acceptance rules.
5. Save the native sidecar before cleanup, alongside the ordinary trajectory
   and its sensor intrinsics/extrinsics. Save the episode/scene identity and
   control timestamps to permit action alignment.
6. Always call `profile.close()` before the existing environment cleanup,
   including rejected samples and exceptions. This restores camera settings,
   parameter-sensor sizes, and environment methods. Resample/create a new
   profile if the underlying MuJoCo model is replaced.

The profile intentionally checks the executed timing contract: 2 ms physics,
one physics step per `env.step()` call, 33 steps per control action, and four
proximity polls per action (at steps 8, 16, 24, 32). Thus control actions are
66 ms apart, while native sensor captures are 15 Hz, rounded causally to the
next physics tick. Four observation polls can refer to the same native frame.
Do not treat them as four independent measurements or index-align native and
action arrays without their timestamps.

## Saving all native zones

`profile.native_depth` contains encoded, unpooled `(40,16,16)` arrays;
`profile.native_times`, `native_valid`, and `native_below` use the same frame
order. The masks describe range validity before clipping. These are encoded
sensor distances, not the renderer's unbounded floating-point depths.

The following export recipe accepts the completed profile and the **actual**
control/action timestamps recorded by the collector. It writes a separate file
without changing the existing trajectory schema. Supply an unused path and
JSON-serializable episode metadata (scene ID, seed, collector revision, outcome).

```python
import json
import h5py
import numpy as np
from pact_vl53l8ux_16x16_sensor import PROFILE


def save_native_sidecar(profile, path, control_sim_time_s, episode_metadata):
    times = np.asarray(profile.native_times, dtype=np.float64)
    depth = np.asarray(profile.native_depth, dtype=np.float32)
    control_times = np.asarray(control_sim_time_s, dtype=np.float64)
    assert depth.shape == (len(times), 40, 16, 16)
    assert len(times) > 0 and np.all(np.diff(times) > 0)
    assert control_times.ndim == 1 and np.isfinite(control_times).all()
    assert np.all(np.diff(control_times) > 0)
    assert np.all(control_times >= times[0])
    assert np.all(control_times <= profile.env.current_data.time)
    # Causal alignment: most recent native acquisition at each action time.
    indices = np.searchsorted(times, control_times, side="right") - 1
    polls = np.asarray([profile.samples[n] for n in profile.names])
    assert polls.shape[0] == 40 and polls.shape[-1] == 2
    assert np.array_equal(polls, np.broadcast_to(polls[:1], polls.shape))
    with h5py.File(path, "x") as h:
        h.create_dataset("native_sim_time_s", data=times)
        h.create_dataset("native_distance_m", data=depth, compression="gzip")
        h.create_dataset("valid_nominal_range", data=profile.native_valid,
                         compression="gzip")
        h.create_dataset("below_min_range", data=profile.native_below,
                         compression="gzip")
        h.create_dataset("sensor_names", data=np.asarray(profile.names, dtype="S"))
        h.create_dataset("control_sim_time_s", data=control_times)
        h.create_dataset("control_native_index", data=indices.astype(np.int32))
        h.create_dataset("observation_poll_sim_time_s", data=polls[0, :, 0])
        h.create_dataset("observation_poll_native_index",
                         data=polls[0, :, 1].astype(np.int32))
        h.attrs["sensor_profile_json"] = json.dumps(PROFILE, sort_keys=True)
        h.attrs["episode_metadata_json"] = json.dumps(episode_metadata, sort_keys=True)
```

Do **not** use `MultizoneProfile.save_audit()` as an arbitrary-length expert
exporter: it enforces the historical **900-action** evaluation and verifies
900 encoder-consumption checks. An expert collector does not make those calls.
The recipe above reads the native buffers directly and does not require PACT.

The normal observation sensor remains `(4,8,8)` per control observation, while
`sensor_param_<name>` supplies the native **16×16** camera intrinsics. Associate
those intrinsics with `native_distance_m`, not blindly with the pooled 8×8
observations. Retain the trajectory's time-varying sensor extrinsics and robot
state; tasks that require extrinsics at each native acquisition should also
record them at capture time. Training an encoder on all 256 zones requires a
separate loader/model change; existing ACT conversion is not automatically a
native-16×16 training pipeline.

## Verification before a collection campaign

The five sensor-only tests can run in a different checkout with NumPy, pytest,
and an EGL-capable MuJoCo installation, without checkpoints or scene assets:

```bash
MUJOCO_GL=egl python -m pytest -q tests/test_pact_v1010c_vl53l8ux_16x16_128d.py \
  -k 'pooling or range_floor or declared_optical or clock_is or actual_native'
```

Using the original full project environment and its recorded paths:

```bash
MUJOCO_GL=egl python -m pytest -q tests/test_pact_v1010c_vl53l8ux_16x16_128d.py
```

The nine existing tests include a real MuJoCo thin-obstacle render showing that
the native 16×16 grid resolves detail missed by the source 8×8 grid, plus pooling,
timing, range, and cleanup checks. They do not exercise a new expert-collection
integration. Check one short expert episode first: inspect a nonconstant
`(N,40,16,16)` sidecar, sensor order, timestamps, masks, correspondence with the
ordinary trajectory, and successful restoration after `close()`. Use a new
dataset directory and retain failed-attempt metadata under the collector's
existing rules.

The executed sensor source SHA-256 is
`ac523a1851d1a5826f06644747573b0900ca09079fad93639374be408515315b`.
The completed evaluation and its qualifications are reported in
[`ablation.md`](../ablation.md).
