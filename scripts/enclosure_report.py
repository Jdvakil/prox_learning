"""One-shot report for an enclosure-reach run. Writes a DATE-TAGGED folder in the repo:

  diagnostics_output/<YYYYMMDD_HHMM>_enclosure_<tag>/
    probes.txt          - the 4 advisor probes + cell/behavior composition
    frames_exo.png      - brightened exo frames from up to 8 episodes (start/mid/grasp)
    frames_wrist.png    - same for wrist camera
    episodes.csv        - per-episode: cell, behavior, success, clearance, light, depth

Usage: python scripts/enclosure_report.py --run-dir assets/datagen/enclosure_smoke --tag finetune1
"""
from __future__ import annotations
import argparse, glob, json, subprocess, sys
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np

REPO = Path("/home/jaydv/code/prox_learning")


def latest_run(run_dir: Path) -> Path:
    stamps = sorted(run_dir.glob("*/2026*"))
    return stamps[-1] if stamps else run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out = REPO / "diagnostics_output" / f"{ts}_enclosure_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    run = latest_run(Path(args.run_dir))
    h5s = sorted(run.glob("house_*/trajectories_batch_*.h5"))
    print(f"run: {run}  h5 files: {len(h5s)}")

    # 1) probes
    probes = subprocess.run(
        [sys.executable, str(REPO / "scripts/dataset_probes.py"),
         "--glob", str(run / "house_*/trajectories_batch_*.h5")],
        capture_output=True, text=True)
    (out / "probes.txt").write_text(probes.stdout + probes.stderr)

    # 2) per-episode table + 3) frames
    rows = ["house,traj,cell,behavior,success,clearance_cm,light,depth_m,cam_visible"]
    exo_rows, wrist_rows = [], []
    for h5p in h5s:
        house = h5p.parent.name
        with h5py.File(h5p, "r") as f:
            for k in sorted(x for x in f if x.startswith("traj_")):
                t = f[k]
                raw = t["obs_scene"]; raw = raw[()] if raw.shape == () else raw[0]
                s = raw.tobytes().decode("utf-8", "ignore").rstrip("\x00") if isinstance(raw, np.ndarray) \
                    else raw.decode("utf-8", "ignore").rstrip("\x00")
                meta = json.loads(s)
                sp = meta.get("scene_params", {})
                rows.append(f"{house},{k},{sp.get('cell','?')},{meta.get('behavior_class','?')},"
                            f"{bool(t['success'][-1])},{sp.get('clearance',0)*100:.1f},"
                            f"{sp.get('light_scale',0):.3f},{sp.get('depth',0):.2f},{sp.get('cam_visible','?')}")
        for cam, acc in (("exo_camera_1", exo_rows), ("wrist_camera", wrist_rows)):
            for v in sorted(h5p.parent.glob(f"episode_*_{cam}_*.mp4"))[:1]:
                cap = cv2.VideoCapture(str(v)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frames = []
                for frac in (0.1, 0.45, 0.7):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac)); ok, fr = cap.read()
                    if ok:
                        g = (np.clip((fr.astype(np.float32) / 255) ** 0.7 * 1.25, 0, 1) * 255).astype(np.uint8)
                        frames.append(g)
                cap.release()
                if frames:
                    acc.append(np.concatenate(frames, axis=1))
        if len(exo_rows) >= 8:
            break
    (out / "episodes.csv").write_text("\n".join(rows))
    if exo_rows:
        cv2.imwrite(str(out / "frames_exo.png"), np.concatenate(exo_rows, axis=0))
    if wrist_rows:
        cv2.imwrite(str(out / "frames_wrist.png"), np.concatenate(wrist_rows, axis=0))
    print(f"report -> {out}")
    print((out / "probes.txt").read_text()[-900:])


if __name__ == "__main__":
    main()
