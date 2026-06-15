# fr3_skin

Franka FR3 + Robotiq 2F-85 with proximity-sensor skin (29 SPAD-style sensors
embedded in `link2`, `link3`, `link5`, `link6`).

Each sensor is an 8x8 depth camera with HFOV 45° (focal_length=10.86mm,
horizontal_aperture=9.0mm) and clipping range 0.05-4.0m.

## Layout

```
fr3_skin/
├── README.md
├── LICENSE                       (Franka MJCF license; see also robotiq_2f85_v4/LICENSE)
├── model.xml                     (sub-model: fr3 + robotiq + skin; for embedding in larger scenes)
├── scene.xml                     (standalone scene: includes model.xml + floor/light/skybox)
├── assets/                       (Franka FR3 visual + collision meshes)
├── skin_meshes/                  (skin "fancy" STL meshes per link)
└── robotiq_2f85_v4/              (Robotiq 2F-85 gripper sub-model + meshes)
```

## Loading

### Standalone (preview in MuJoCo viewer)

```bash
python -m mujoco.viewer --mjcf=scene.xml
```

### As a sub-model (programmatic)

```python
import mujoco
m = mujoco.MjModel.from_xml_path("scene.xml")          # standalone
m = mujoco.MjModel.from_xml_path("model.xml")          # robot only (no floor/light)
```

`model.xml` exposes `fr3_link0` as the root body for parent scenes that want to
attach the robot at a custom pose; it has no floor/light/global-visual settings.

## Sensors

29 cameras named `link{2,3,5,6}_sensor_{N}` (counts: link2=7, link3=8, link5=6,
link6=8). All have `fovy="45.0" resolution="8 8"` and are mounted in their
respective `link{N}_skin` body. To render depth from a sensor in MuJoCo:

```python
renderer = mujoco.Renderer(m, height=8, width=8)
renderer.enable_depth_rendering()
renderer.update_scene(d, camera="link6_sensor_0")
depth = renderer.render()  # (8, 8) float32 depth in meters
```

The `<visual><map znear="0.005" zfar="10"/></visual>` in scene.xml enforces the
depth clipping range; clip values returned by the renderer to [0.05, 4.0] for
SPAD-faithful output.

## Sensor placement

Sensor positions are defined relative to the parent skin body (`link2_skin`,
etc.). To export their world poses, traverse the body tree:

```python
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("scene.xml")
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
for i in range(m.ncam):
    name = m.camera(i).name
    if "_sensor_" not in name: continue
    body = m.body(m.camera(i).bodyid)
    print(name, "world pos:", d.xpos[body.id], "world quat:", d.xquat[body.id])
```

## Licensing

- Franka FR3 meshes: see `LICENSE` (Franka Apache 2.0 / Franka EULA)
- Robotiq 2F-85: see `robotiq_2f85_v4/LICENSE`
- Skin meshes: derived from the gentact_ros_tools self-cap meshes (please
  attribute the original authors).

## Known caveats

- `model.xml` is configured as a sub-model for embedding; it sets `<compiler
  meshdir="assets"/>` and assumes `skin_meshes/` lives one level up. If you
  move `model.xml` to a different location, update the relative paths or
  switch to absolute paths.
- The skin geom has `contype="0" conaffinity="0"` so it is purely visual.
- No actuators are defined for the gripper outside what `robotiq_2f85_v4/2f85.xml`
  exposes; the FR3 has 7 position actuators (`fr3_joint1..7`).
