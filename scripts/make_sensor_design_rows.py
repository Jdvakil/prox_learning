"""Build one training dataset per simulated sensor design from recorded rows.

Copies a rows-format dataset (manifest.json + rows/*/ each holding an episode
h5 with traj_0/obs/proximity/<sensor> in metres) and rewrites only the
proximity datasets through scripts/sensor_designs.apply_design. Everything
else - actions, qpos, images, manifest - is byte-identical, so a PACT variant
trained on the output differs from the original in nothing but what its
sensors could see.

  python scripts/make_sensor_design_rows.py \
      --src ~/v1010_data/pact_place_corridor_v10_10 \
      --design vl53l5cx_4x4 --out ~/v1010_design_sets --seed 3103

Noise is seeded per (seed, row, sensor), so rebuilding a dataset reproduces it
bit for bit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_designs import DESIGNS, apply_design


def transform_row(src_h5: Path, dst_h5: Path, design: str, seed: int, row_name: str) -> int:
    shutil.copy2(src_h5, dst_h5)
    n = 0
    with h5py.File(dst_h5, "r+") as handle:
        for traj_key in list(handle.keys()):
            grp = handle[traj_key]
            if "obs/proximity" not in grp:
                continue
            prox = grp["obs/proximity"]
            for sensor in list(prox.keys()):
                digest = hashlib.sha256(f"{seed}|{row_name}|{sensor}".encode()).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
                data = prox[sensor][()].astype(np.float32)
                prox[sensor][...] = apply_design(data, design, rng)
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="dataset dir holding manifest.json and rows/")
    ap.add_argument("--design", required=True, choices=sorted(DESIGNS))
    ap.add_argument("--out", required=True, help="parent dir; output lands in <out>/<srcname>__<design>")
    ap.add_argument("--seed", type=int, default=3103)
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.out).resolve() / f"{src.name}__{args.design}"
    if dst.exists():
        raise SystemExit(f"refusing to overwrite existing {dst}")
    dst.mkdir(parents=True)
    shutil.copy2(src / "manifest.json", dst / "manifest.json")

    rows = sorted(p for p in (src / "rows").iterdir() if p.is_dir())
    total = 0
    for i, row in enumerate(rows):
        out_row = dst / "rows" / row.name
        out_row.mkdir(parents=True)
        for item in row.iterdir():
            if item.suffix in (".h5", ".hdf5"):
                total += transform_row(item, out_row / item.name, args.design, args.seed, row.name)
            else:
                shutil.copy2(item, out_row / item.name)
        if (i + 1) % 24 == 0:
            print(f"  {i + 1}/{len(rows)} rows", flush=True)

    (dst / "sensor_design.json").write_text(json.dumps(
        {"design": args.design, **DESIGNS[args.design],
         "source_dataset": str(src), "noise_seed": args.seed}, indent=2))
    print(f"done: {dst}  ({len(rows)} rows, {total} sensor streams rewritten)")


if __name__ == "__main__":
    main()
