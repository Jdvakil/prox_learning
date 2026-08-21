#!/usr/bin/env python3
"""B2b Pass 2: run one real V5 episode per V8B clutter family."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for path in (ROOT / "scripts", MOLMO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pact_place_corridor_contract import sha256_payload  # noqa: E402
from run_pact_place_expert_screen import (  # noqa: E402
    _protected_eval_processes,
    run_row,
    write_json_atomic,
)
from run_pact_place_clutter_sweep_v8b import FAMILIES  # noqa: E402

CONFIG_PATH = ROOT / "configs/pact_place_corridor_v8b_pass1.json"
OUTPUT_ROOT = ROOT / "diagnostics_output/pact_place_corridor_v8b_pass2c"
SCENE_XML = MOLMO / "molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v5.xml"


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", "/root/prox_learning/assets")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.pop("DISPLAY", None)
    protected = _protected_eval_processes()
    if protected:
        raise SystemExit(f"protected confirmatory evaluation is active: {protected}")
    config = json.loads(CONFIG_PATH.read_text())
    rows_by_family_attempt = {
        (row["family"], int(row["family_attempt"])): row
        for row in config["family_review_rows"]
    }
    results = []
    realized = []
    for family in FAMILIES:
        for attempt in range(1, 5):
            row = rows_by_family_attempt[(family, attempt)]
            result = run_row(
                row,
                config_sha256=config["config_sha256"],
                output_root=str(OUTPUT_ROOT),
                scene_xml=str(SCENE_XML),
            )
            record = {
                "family": row["family"],
                "family_attempt": attempt,
                "role_index": row["role_index"],
                "episode_id": row["episode_id"],
                "layout_id": row["layout_id"],
                "status": result["status"],
                "task_success": bool(result.get("task_success")),
                "clean_success": bool(result.get("clean_success")),
                "selected_seed": result.get("selected_seed"),
                "retry_history": result.get("retry_history") or [],
                "result_path": str(
                    (
                        OUTPUT_ROOT
                        / "expert_screen_rows"
                        / f"{int(row['role_index']):02d}_{row['episode_id'][:16]}"
                        / "result.json"
                    ).relative_to(ROOT)
                ),
            }
            results.append(record)
            print(
                f"{row['family']} attempt={attempt}: status={record['status']} "
                f"clean={record['clean_success']}",
                flush=True,
            )
            if record["status"] != "sampling_failure":
                realized.append(record)
                break
    report = {
        "schema_version": "pact_place_v8b_pass2_realized_v1",
        "role": "six_family_realized_measurement_not_a_gate",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "config_sha256": config["config_sha256"],
        "n_families": len(FAMILIES),
        "n_attempts": len(results),
        "n_realized_episodes": len(realized),
        "one_real_episode_per_family": len(realized) == len(FAMILIES),
        "realized_results": realized,
        "attempts": results,
    }
    report["realized_runs_sha256"] = sha256_payload(report)
    write_json_atomic(OUTPUT_ROOT / "realized_runs.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
