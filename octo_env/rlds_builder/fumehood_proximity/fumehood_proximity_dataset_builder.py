"""RLDS builder: fume-hood datagen episodes -> Octo-ready TFDS dataset.

Reads the molmospaces datagen layout directly (no intermediate conversion):

    <RAW_DIR>/house_<k>/trajectories_batch_1_of_1.h5
        traj_<i>/obs/agent/qpos        (T, 2000) uint8, JSON rows {"arm":[7],"gripper":[2]}
        traj_<i>/actions/joint_pos     (T, 2000) uint8, JSON rows {"arm":[7],"gripper":[1]}
        traj_<i>/obs/proximity/<name>  (T, 4, 8, 8) float32 depth
        traj_<i>/success, fail         (T,) bool
    <RAW_DIR>/house_<k>/episode_<i:08d>_<camera>_batch_1_of_1.mp4

Per step it emits image_primary (exo), image_wrist, an 8-dim joint action, and
`state` = qpos(9) ++ per-sensor min depth (N_SENSORS) — proximity is appended
into the lowdim state vector so Octo's stock data pipeline carries it without
modification, exactly the "append our data in Octo's format" idea. A separate
`proximity` feature keeps the raw vector for analysis.

Two builder configs give the paired comparison for free:
    with_proximity : state = qpos(9) + proximity(N_SENSORS)
    vision_only    : state = qpos(9)

Usage:
    export FUMEHOOD_RAW_DIR=/path/to/datagen/cluttered_fumehood_v1
    cd octo_env/rlds_builder/fumehood_proximity && tfds build --config with_proximity
"""
import glob
import json
import os

import cv2
import h5py
import numpy as np
import tensorflow_datasets as tfds

N_SENSORS = 40           # hybrid skin; assert-checked against the h5 at build time
DEAD_PIXEL_M = 0.005     # depths below this are dead/invalid returns
FAR_M = 4.0              # fill value when a sensor has no valid return
PRIMARY_HW = (256, 256)  # octo-small primary camera size
WRIST_HW = (128, 128)    # octo-small wrist camera size
INSTRUCTION = "pick up the red cup in the fume hood"


def _decode_json_rows(arr):
    out = []
    for row in np.asarray(arr):
        raw = bytes(row).split(b"\x00")[0].decode("utf-8", "ignore").strip()
        out.append(json.loads(raw) if raw else {})
    return out


def _read_video(path, hw):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


class FumehoodProximity(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}
    BUILDER_CONFIGS = [
        tfds.core.BuilderConfig(name="with_proximity",
                                description="state = qpos(9) + min-depth per sensor"),
        tfds.core.BuilderConfig(name="vision_only",
                                description="state = qpos(9); proximity feature still stored"),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        state_dim = 9 + (N_SENSORS if self.builder_config.name == "with_proximity" else 0)
        return tfds.core.DatasetInfo(
            builder=self,
            description="Fume-hood manipulation with whole-body proximity skin.",
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image_primary": tfds.features.Image(
                            shape=(*PRIMARY_HW, 3), dtype=np.uint8, encoding_format="jpeg"),
                        "image_wrist": tfds.features.Image(
                            shape=(*WRIST_HW, 3), dtype=np.uint8, encoding_format="jpeg"),
                        "state": tfds.features.Tensor(shape=(state_dim,), dtype=np.float32),
                        "proximity": tfds.features.Tensor(shape=(N_SENSORS,), dtype=np.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                    "traj_key": tfds.features.Text(),
                    "success": tfds.features.Scalar(dtype=np.bool_),
                }),
            }),
        )

    def _split_generators(self, dl_manager):
        raw_dir = os.environ.get("FUMEHOOD_RAW_DIR")
        assert raw_dir, "set FUMEHOOD_RAW_DIR to the datagen output directory"
        return {"train": self._generate_examples(raw_dir)}

    def _generate_examples(self, raw_dir):
        include_prox = self.builder_config.name == "with_proximity"
        for h5_path in sorted(glob.glob(os.path.join(raw_dir, "house_*", "trajectories_batch_*.h5"))):
            house_dir = os.path.dirname(h5_path)
            with h5py.File(h5_path, "r") as f:
                for traj_key in sorted(f.keys()):
                    g = f[traj_key]
                    if "obs/proximity" not in g:
                        continue
                    if "fail" in g and bool(np.asarray(g["fail"])[-1]):
                        continue  # train on successes only, matching the ACT recipe

                    qpos_rows = _decode_json_rows(g["obs/agent/qpos"])
                    act_rows = _decode_json_rows(g["actions/joint_pos"])

                    sensors = sorted(g["obs/proximity"].keys())
                    assert len(sensors) == N_SENSORS, (
                        f"{h5_path}:{traj_key} has {len(sensors)} sensors, "
                        f"builder expects {N_SENSORS} — edit N_SENSORS")
                    depth = np.stack([np.asarray(g[f"obs/proximity/{s}"]) for s in sensors])
                    # (S, T, 4, 8, 8) -> per-step min valid depth per sensor (T, S)
                    d = depth.reshape(depth.shape[0], depth.shape[1], -1)
                    d = np.where(d > DEAD_PIXEL_M, d, np.inf).min(axis=-1)
                    prox = np.where(np.isfinite(d), d, FAR_M).T.astype(np.float32)

                    idx = int(traj_key.split("_")[-1])
                    vids = {}
                    for cam, hw in (("exo_camera_1", PRIMARY_HW), ("wrist_camera", WRIST_HW)):
                        mp4 = os.path.join(house_dir,
                                           f"episode_{idx:08d}_{cam}_batch_1_of_1.mp4")
                        if not os.path.exists(mp4):
                            vids = None
                            break
                        vids[cam] = _read_video(mp4, hw)
                    if not vids:
                        continue

                    T = min(len(qpos_rows), len(act_rows), prox.shape[0],
                            len(vids["exo_camera_1"]), len(vids["wrist_camera"]))
                    steps = []
                    for t in range(T):
                        q, a = qpos_rows[t], act_rows[t]
                        if "arm" not in q or "arm" not in a:
                            break  # trailing empty rows
                        qvec = np.asarray(q["arm"][:7] + list(q.get("gripper", [0, 0]))[:2],
                                          dtype=np.float32)
                        avec = np.asarray(a["arm"][:7] + list(a.get("gripper", [0]))[:1],
                                          dtype=np.float32)
                        state = np.concatenate([qvec, prox[t]]) if include_prox else qvec
                        steps.append({
                            "observation": {
                                "image_primary": vids["exo_camera_1"][t],
                                "image_wrist": vids["wrist_camera"][t],
                                "state": state.astype(np.float32),
                                "proximity": prox[t],
                            },
                            "action": avec,
                            "discount": np.float32(1.0),
                            "reward": np.float32(0.0),
                            "is_first": t == 0,
                            "is_last": False,
                            "is_terminal": False,
                            "language_instruction": INSTRUCTION,
                        })
                    if len(steps) < 8:
                        continue
                    steps[-1]["is_last"] = True
                    steps[-1]["is_terminal"] = True
                    steps[-1]["reward"] = np.float32(1.0)
                    yield f"{os.path.basename(house_dir)}_{traj_key}", {
                        "steps": steps,
                        "episode_metadata": {
                            "file_path": h5_path,
                            "traj_key": traj_key,
                            "success": True,
                        },
                    }
