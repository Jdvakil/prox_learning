"""Studio PHOTOSHOOT of the FR3 + dense hybrid skin (model_photoshoot.xml) for the paper:
  - a 360 turntable sweep (orbit) -> individual frames + a contact-sheet grid + an mp4
  - a pose sweep (several arm configurations) -> a strip
  - a high-res hero shot
Dark studio background, soft key/fill lighting, blue skin + red SPAD dots.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import OpenGL.EGL as _EGL  # noqa: E402
_dpy = _EGL.eglGetDisplay(_EGL.EGL_DEFAULT_DISPLAY)
_a, _b = _EGL.EGLint(), _EGL.EGLint()
if _EGL.eglInitialize(_dpy, _a, _b):
    import mujoco.egl as _me
    _me.EGL_DISPLAY = _dpy

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROBOT = Path("/home/jaydv/code/prox_learning/assets/robots/franka_skin/model_photoshoot.xml")
OUT = Path("/home/jaydv/code/prox_learning/diagnostics_output/20260611_skin_photoshoot")
OUT.mkdir(parents=True, exist_ok=True)
FRAMES = OUT / "turntable_frames"
FRAMES.mkdir(exist_ok=True)

HERO = [0.0, -0.35, 0.0, -2.10, 0.0, 1.95, 0.79]
POSE_SWEEP = {
    "stow": [0.0, -0.6, 0.0, -2.6, 0.0, 2.0, 0.79],
    "reach": [0.0, -0.35, 0.0, -2.1, 0.0, 1.95, 0.79],
    "extend": [0.0, 0.1, 0.0, -1.5, 0.0, 1.6, 0.79],
    "twist": [0.7, -0.5, 0.5, -2.1, 0.4, 1.9, 0.4],
    "up": [0.0, 0.4, 0.0, -1.0, 0.0, 1.4, 0.79],
}
W, H = 1200, 1200


def build():
    spec = mujoco.MjSpec.from_file(str(ROBOT))
    spec.visual.global_.offwidth = max(W, 1600)
    spec.visual.global_.offheight = max(H, 1600)
    # studio lights + a soft dark floor
    spec.worldbody.add_light(pos=[0.6, 0.8, 2.6], dir=[-0.15, -0.25, -1],
                             diffuse=[0.95, 0.95, 0.98], specular=[0.35, 0.35, 0.4])
    spec.worldbody.add_light(pos=[-1.1, -0.6, 1.8], dir=[0.55, 0.3, -1],
                             diffuse=[0.35, 0.38, 0.5], specular=[0.1, 0.1, 0.15])
    spec.worldbody.add_light(pos=[0.0, -1.4, 1.2], dir=[0.0, 1.0, -0.4],
                             diffuse=[0.25, 0.28, 0.4], specular=[0.05, 0.05, 0.1])
    fl = spec.worldbody.add_geom()
    fl.type = mujoco.mjtGeom.mjGEOM_PLANE
    fl.size = [4, 4, 0.1]
    fl.rgba = [0.10, 0.11, 0.13, 1.0]
    return spec.compile()


def set_pose(model, data, q):
    for i, v in enumerate(q, start=1):
        data.qpos[model.joint(f"fr3_joint{i}").qposadr[0]] = v
    mujoco.mj_forward(model, data)


def render(rndr, data, az, el=-12, dist=1.45, lookat=(0.32, 0.0, 0.72)):
    cam = mujoco.MjvCamera()
    cam.lookat = list(lookat)
    cam.distance = dist
    cam.azimuth = az
    cam.elevation = el
    rndr.update_scene(data, cam)
    rndr.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0     # clean dark bg
    rndr.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    img = rndr.render().copy()
    # subtle vignette + gamma for a studio look
    g = (np.clip((img.astype(np.float32) / 255) ** 0.88 * 1.06, 0, 1) * 255).astype(np.uint8)
    return g


def main():
    model = build()
    data = mujoco.MjData(model)
    rndr = mujoco.Renderer(model, H, W)

    # 1) turntable: 36 frames around, hero pose
    set_pose(model, data, HERO)
    n = 36
    frames = []
    for k in range(n):
        az = 360.0 * k / n
        img = render(rndr, data, az)
        cv2.imwrite(str(FRAMES / f"turn_{k:03d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        frames.append(img)
    # contact sheet (4x4 of every ~2nd-3rd frame)
    pick = frames[::max(1, n // 16)][:16]
    cells = [cv2.cvtColor(cv2.resize(f, (W // 2, H // 2)), cv2.COLOR_RGB2BGR) for f in pick]
    rows = [np.concatenate(cells[i:i + 4], axis=1) for i in range(0, 16, 4)]
    cv2.imwrite(str(OUT / "turntable_contact_sheet.png"), np.concatenate(rows, axis=0))
    # mp4
    vw = cv2.VideoWriter(str(OUT / "turntable.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 18, (W, H))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()

    # 2) pose sweep strip (hero angle)
    strip = []
    for name, q in POSE_SWEEP.items():
        set_pose(model, data, q)
        img = render(rndr, data, az=135)
        lab = cv2.cvtColor(cv2.resize(img, (W // 2, H // 2)), cv2.COLOR_RGB2BGR)
        cv2.putText(lab, name, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 240), 2, cv2.LINE_AA)
        strip.append(lab)
    cv2.imwrite(str(OUT / "pose_sweep.png"), np.concatenate(strip, axis=1))

    # 3) hero shot (high res, 3/4 view) + forearm close-up showing the dense SPAD coverage
    set_pose(model, data, HERO)
    rndr_h = mujoco.Renderer(model, 1600, 1600)
    hero = render(rndr_h, data, az=145, el=-12, dist=1.30, lookat=(0.34, 0.0, 0.74))
    cv2.imwrite(str(OUT / "hero.png"), cv2.cvtColor(hero, cv2.COLOR_RGB2BGR))
    closeup = render(rndr_h, data, az=120, el=-8, dist=0.62, lookat=(0.18, 0.0, 0.78))
    cv2.imwrite(str(OUT / "closeup_sensors.png"), cv2.cvtColor(closeup, cv2.COLOR_RGB2BGR))

    print(f"turntable {n} frames, contact sheet, mp4, pose sweep, hero, closeup -> {OUT}")


if __name__ == "__main__":
    main()
