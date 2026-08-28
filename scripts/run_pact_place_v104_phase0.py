#!/usr/bin/env python3
"""V10.4 Step-4: the single untouched 24-row Phase-0 gate.

Implemented and tested before owner review, executed only against a valid
owner-supplied approval. No review seed, outcome-based substitution, retry, or
extra row is permitted. Even a passing gate sets only ``phase0_passed`` and
leaves every downstream authorization false.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_corridor_contract import sha256_file  # noqa: E402
from pact_place_v104_contract import (  # noqa: E402
    CAUSAL_ROOT,
    CONTRACT_VERSION,
    ENVIRONMENT_VERSION,
    GATE_MASTER_SEED,
    GATE_MIN_CLEARANCE_M,
    GATE_STREAM,
    N_GATE_ROWS,
    PHASE0_ROOT,
    PREFLIGHT_ROOT,
    PRODUCTION_ROOT,
    REVIEW_ROOT,
    SAMPLER_CLASS,
    assert_phase0_approval,
    build_contract,
    empty_authorization,
    gate_eligibility,
    implementation_sha256,
    is_clean_success,
    row_defects,
    sha256_payload,
    verify_protected_artifacts,
    write_immutable_create_only,
)
from pact_place_v104_geometry import SCENE_XML_RELATIVE_V104  # noqa: E402
from run_pact_place_expert_screen import (  # noqa: E402
    TERMINAL_STATUSES,
    _result_path,
    _validate_existing,
    run_row,
)

SCENE_XML = ROOT / SCENE_XML_RELATIVE_V104


def _artifact_sha(path: Path) -> str:
    return json.loads(path.read_text())["artifact_sha256"]


def expected_bindings(contract: dict[str, Any]) -> dict[str, str]:
    review_root = ROOT / REVIEW_ROOT
    bindings = {
        "contract_sha256": contract["contract_sha256"],
        "implementation_sha256": contract["implementation_sha256"],
        "scene_sha256": contract["scene_sha256"],
        "preflight_sha256": _artifact_sha(ROOT / PREFLIGHT_ROOT / "preflight.json"),
        "production_manifest_sha256": _artifact_sha(
            ROOT / PRODUCTION_ROOT / "production_manifest.json"
        ),
        "causal_sha256": _artifact_sha(ROOT / CAUSAL_ROOT / "causal.json"),
        "review_manifest_sha256": _artifact_sha(review_root / "review_manifest.json"),
    }
    for video in sorted((review_root / "videos").glob("*.mp4")):
        bindings[f"video_sha256:{video.name}"] = sha256_file(video)
    return bindings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / PHASE0_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    contract = build_contract()
    provenance = verify_protected_artifacts(contract["protected_artifacts"])
    if not provenance["passed"]:
        raise SystemExit(f"protected artifact drift: {provenance['mismatches'][:3]}")
    approval_path = args.approval.resolve()
    if not approval_path.is_file():
        raise PermissionError(f"missing owner approval: {approval_path}")
    approval = json.loads(approval_path.read_text())
    assert_phase0_approval(approval, expected_bindings(contract))

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = list(contract["gate_rows"])
    if len(rows) != N_GATE_ROWS:
        raise RuntimeError("the Phase-0 manifest must contain exactly 24 rows")
    review_ids = {row["episode_id"] for row in contract["review_rows"]}
    if {row["episode_id"] for row in rows} & review_ids:
        raise RuntimeError("a review row leaked into the Phase-0 manifest")

    config = {
        "schema_version": "pact_place_v104_phase0_config_v1",
        "contract_version": CONTRACT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "implementation_sha256": implementation_sha256(),
        "scene_sha256": contract["scene_sha256"],
        "sampler_class": SAMPLER_CLASS,
        "gate_stream": GATE_STREAM,
        "gate_master_seed": GATE_MASTER_SEED,
        "approval_path": str(approval_path.relative_to(ROOT)),
        "approval_sha256": sha256_file(approval_path),
        "expected_clean_range": [21, 24],
        "pass_threshold": 20,
        "frozen_before_row_0": True,
        "expert_screen_rows": rows,
        **empty_authorization(),
    }
    config["config_sha256"] = sha256_payload(config)
    write_immutable_create_only(output_root / "gate_manifest.json", config)

    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in rows:
        existing = _result_path(output_root, row)
        if existing.is_file():
            document = json.loads(existing.read_text())
            if document.get("status") in TERMINAL_STATUSES:
                kept = _validate_existing(existing, row, config["config_sha256"])
                if kept is None:
                    raise RuntimeError(
                        f"refusing to replace terminal gate row {row['role_index']}"
                    )
                results.append(kept)
                continue
        pending.append(row)
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, min(args.workers, max(1, len(pending)))),
        mp_context=context,
        max_tasks_per_child=1,
    ) as executor:
        for future in concurrent.futures.as_completed(
            [
                executor.submit(
                    run_row,
                    row,
                    config_sha256=config["config_sha256"],
                    output_root=str(output_root),
                    scene_xml=str(SCENE_XML),
                )
                for row in pending
            ]
        ):
            results.append(future.result())
    results.sort(key=lambda item: int(item["role_index"]))
    eligibility = gate_eligibility(rows, results)
    document = {
        "schema_version": "pact_place_v104_phase0_gate_v1",
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "config_sha256": config["config_sha256"],
        "approval_sha256": config["approval_sha256"],
        "implementation_sha256": implementation_sha256(),
        "n_rows": N_GATE_ROWS,
        "rows": [
            {
                "role_index": item["role_index"],
                "intrusion_side": item.get("intrusion_side"),
                "status": item["status"],
                "v104_clean_success": is_clean_success(
                    item, min_clearance_m=GATE_MIN_CLEARANCE_M
                ),
                "v104_defects": row_defects(item, min_clearance_m=GATE_MIN_CLEARANCE_M),
                "pact_v104_frame_telemetry": item.get("pact_v104_frame_telemetry"),
                "result_sha256": item.get("result_sha256"),
            }
            for item in results
        ],
        "eligibility": eligibility,
        "no_row_replaced_or_reseeded": True,
        **empty_authorization(),
        "phase0_passed": bool(eligibility["phase0_passed"]),
        "permanent_stop": not bool(eligibility["phase0_passed"]),
    }
    digest = write_immutable_create_only(output_root / "gate.json", document)
    print(json.dumps({
        "phase0_passed": document["phase0_passed"],
        "clean_successes": eligibility["clean_successes"],
        "clean_by_side": eligibility["clean_by_side"],
        "artifact_sha256": digest,
    }, indent=2))
    return 0 if document["phase0_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
