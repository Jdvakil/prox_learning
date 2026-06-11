"""Assemble the overnight hybrid-skin package into labeled contact sheets + an index.
CPU/cv2 only (no GPU). Run after the viz workflows finish.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REPO = Path("/home/jaydv/code/prox_learning")
DIAG = REPO / "diagnostics_output"
OUT = DIAG / "20260611_hybrid_skin_REPORT"
OUT.mkdir(parents=True, exist_ok=True)

SECTIONS = {
    "00_environments": ("Cool environments — the skin sensing real geometry", [
        "20260611_hybrid_overnight/env_corner_cavity.png",
        "20260611_hybrid_overnight/env_pipe_tunnel.png",
        "20260611_hybrid_overnight/env_narrow_slot.png",
        "20260611_hybrid_overnight/env_cluttered_shelf.png",
        "20260611_hybrid_overnight/env_peg_forest.png",
        "20260611_hybrid_overnight/env_overhang.png",
    ]),
    "01_accurate": ("ACCURATE — sensors read metric truth", [
        "20260611_hybrid_viz_suite/range_accuracy_scatter.png",
        "20260611_hybrid_viz_suite/plane_distance_sweep.png",
        "20260611_hybrid_viz_suite/plane_tilt_sweep.png",
        "20260611_hybrid_viz_suite/single_sensor_anatomy.png",
        "20260611_hybrid_overnight/acc_repeatability_noise.png",
        "20260611_hybrid_overnight/acc_range_linearity_perlink.png",
        "20260611_hybrid_overnight/acc_angular_resolution.png",
        "20260611_hybrid_viz_suite/known_shapes_cloud.png",
    ]),
    "02_useful": ("USEFUL — whole-arm clearance + mapping from proximity", [
        "20260611_hybrid_overnight/clearance_controller.png",
        "20260611_hybrid_overnight/use_whole_arm_clearance.png",
        "20260611_hybrid_overnight/use_cloud_accumulation.png",
        "20260611_hybrid_viz_suite/cavity_reconstruction_3d.png",
    ]),
    "03_needed": ("NEEDED — vision fails, the skin does not", [
        "20260611_hybrid_overnight/need_vision_vs_skin.png",
        "20260611_hybrid_overnight/need_blur_and_dark.png",
        "20260611_hybrid_overnight/need_coverage_behind.png",
        "20260611_hybrid_viz_suite/fov_coverage_map.png",
    ]),
    "04_verification": ("VERIFICATION — 38/40 sensors, cloud vs ground truth", [
        "20260611_hybrid_sensor_verify/cloud_vs_geometry.png",
        "20260611_hybrid_viz_suite/sensor_gallery_cavity.png",
        "20260610_hybrid_skin_rich/hybrid_skin_rich_test.png",
    ]),
}


def label(img, text, sub=""):
    w = img.shape[1]
    bar = np.full((46 if sub else 34, w, 3), 28, np.uint8)
    cv2.putText(bar, text, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 245), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(bar, sub, (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 200, 240), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def fit_width(img, W):
    h = int(img.shape[0] * W / img.shape[1])
    return cv2.resize(img, (W, h))


def main():
    W = 1400
    index_rows = []
    for sec, (title, files) in SECTIONS.items():
        imgs = []
        present = []
        for f in files:
            p = DIAG / f
            if p.exists():
                im = fit_width(cv2.imread(str(p)), W)
                imgs.append(label(im, "", ""))
                present.append(f.split("/")[-1])
        if not imgs:
            continue
        sheet = np.concatenate(imgs, 0)
        sheet = label(sheet, title, f"{len(present)} panels")
        outp = OUT / f"{sec}.png"
        cv2.imwrite(str(outp), sheet)
        index_rows.append(f"{sec}: {len(present)} panels -> {outp.name}")
        print(f"{sec}: {len(present)} panels -> {outp}")
    (OUT / "INDEX.txt").write_text(
        "HYBRID SKIN — overnight package\n\n" + "\n".join(index_rows) +
        "\n\nModel: assets/robots/franka_skin/model_hybrid.xml (40 SPAD sensors, verified 38/40)\n")
    print(f"\nREPORT -> {OUT}")


if __name__ == "__main__":
    main()
