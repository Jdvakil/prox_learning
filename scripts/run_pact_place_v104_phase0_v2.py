#!/usr/bin/env python3
"""V10.4 Phase0-v2: the frozen 24-row gate, behind an owner-authored approval.

Frozen before the review packet is generated, so approval requires no code
change. The verifier recomputes every binding from file bytes; an embedded
self-hash is never taken on trust, because a tampered artifact would carry a
tampered self-hash too.

Nothing here runs until a valid owner record exists. The gate directory is not
created, and no row is executed, before the approval passes.
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
    ENVIRONMENT_VERSION,
    GATE_MASTER_SEED,
    GATE_MIN_CLEARANCE_M,
    GATE_STREAM,
    N_GATE_ROWS,
    build_contract,
    gate_eligibility,
    is_clean_success,
    row_defects,
    sha256_payload,
    verify_protected_artifacts,
)
from pact_place_v104_geometry import SCENE_XML_RELATIVE_V104  # noqa: E402
from pact_place_v104_review_v2_contract import (  # noqa: E402
    CONTRACT_VERSION_V2,
    EXECUTED_V1_CONTRACT_SHA256,
    EXECUTED_V1_IMPLEMENTATION_SHA256,
    N_REVIEW_V2_VIDEOS,
    PHASE0_V2_ROOT,
    REVIEW_V2_ROOT,
    SCENE_METADATA_SHA256,
    SCHEMA_PREFIX,
    build_provenance_bridge,
    empty_authorization,
    gate_v2_implementation_sha256,
    review_v2_implementation_sha256,
    scoped_production_sha256,
    sha256_bytes_of,
    write_immutable_create_only,
)
from run_pact_place_expert_screen import (  # noqa: E402
    TERMINAL_STATUSES,
    _result_path,
    _validate_existing,
    run_row,
)

SCENE_XML = ROOT / SCENE_XML_RELATIVE_V104
SAMPLER_CLASS = "PactPlaceCorridorV104Sampler"


class ApprovalError(PermissionError):
    """The owner record is missing, stale, partial, or agent-created."""


def _recomputed_payload_sha256(path: Path) -> str:
    """Recompute a document's self-hash. Never returns the embedded value."""
    document = json.loads(Path(path).read_text())
    return sha256_payload(
        {key: value for key, value in document.items() if key != "artifact_sha256"}
    )


def review_video_hashes(review_root: Path) -> dict[str, str]:
    return {
        video.name: sha256_file(video)
        for video in sorted((Path(review_root) / "videos").glob("*.mp4"))
    }


def expected_bindings_v2(review_root: Path | None = None) -> dict[str, str]:
    """Every binding the owner must sign, recomputed from bytes."""
    review_root = Path(review_root) if review_root else ROOT / REVIEW_V2_ROOT
    bindings = {
        "contract_version_v2": CONTRACT_VERSION_V2,
        "scoped_production_sha256": scoped_production_sha256(),
        "review_v2_implementation_sha256": review_v2_implementation_sha256(),
        "gate_v2_implementation_sha256": gate_v2_implementation_sha256(),
        "production_scene_sha256": sha256_bytes_of(ROOT / SCENE_XML_RELATIVE_V104),
        "scene_metadata_sha256": SCENE_METADATA_SHA256,
        "executed_v1_contract_sha256": EXECUTED_V1_CONTRACT_SHA256,
        "executed_v1_implementation_sha256": EXECUTED_V1_IMPLEMENTATION_SHA256,
        "provenance_bridge_sha256": _recomputed_payload_sha256(
            review_root / "provenance_bridge.json"
        ),
        "control_certificates_sha256": _recomputed_payload_sha256(
            review_root / "control_certificates.json"
        ),
        "review_preflight_sha256": _recomputed_payload_sha256(
            review_root / "review_preflight.json"
        ),
        "review_manifest_sha256": _recomputed_payload_sha256(
            review_root / "review_manifest.json"
        ),
    }
    for name, digest in review_video_hashes(review_root).items():
        bindings[f"video_sha256:{name}"] = digest
    return bindings


def verify_review_packet(review_root: Path) -> dict[str, Any]:
    """The packet must be complete, self-consistent, and eligible."""
    review_root = Path(review_root)
    problems: list[str] = []
    required = (
        "provenance_bridge.json",
        "control_certificates.json",
        "review_preflight.json",
        "review_manifest.json",
        "REVIEW.md",
    )
    for name in required:
        if not (review_root / name).is_file():
            problems.append(f"missing artifact: {name}")
    if problems:
        return {"passed": False, "problems": problems}

    for name in required[:-1]:
        document = json.loads((review_root / name).read_text())
        if document.get("artifact_sha256") != _recomputed_payload_sha256(
            review_root / name
        ):
            problems.append(f"self-hash mismatch: {name}")

    bridge = json.loads((review_root / "provenance_bridge.json").read_text())
    if not bridge.get("bridge_passed"):
        problems.append("stored provenance bridge did not pass")
    live_bridge = build_provenance_bridge()
    if not live_bridge["bridge_passed"]:
        problems.append(
            f"live provenance bridge fails now: {live_bridge['failed_sections']}"
        )

    certificates = json.loads((review_root / "control_certificates.json").read_text())
    if not certificates.get("all_certified"):
        problems.append("not every diagnostic control was certified")
    for certificate in certificates.get("certificates", []):
        if not certificate.get("certified"):
            problems.append(f"control not certified: {certificate.get('control')}")

    manifest = json.loads((review_root / "review_manifest.json").read_text())
    if not manifest.get("eligible_for_human_review"):
        problems.append("review manifest is not eligible for human review")
    if int(manifest.get("n_videos", 0)) != N_REVIEW_V2_VIDEOS:
        problems.append(f"expected {N_REVIEW_V2_VIDEOS} videos in the manifest")
    if int(manifest.get("n_production_successes", 0)) != 3:
        problems.append("expected exactly three production successes")
    if int(manifest.get("n_diagnostic_controls", 0)) != 3:
        problems.append("expected exactly three diagnostic controls")

    observed = review_video_hashes(review_root)
    declared = dict(manifest.get("video_sha256") or {})
    if len(observed) != N_REVIEW_V2_VIDEOS:
        problems.append(
            f"expected exactly {N_REVIEW_V2_VIDEOS} MP4 files, found {len(observed)}"
        )
    if set(observed) != set(declared):
        problems.append("published MP4 inventory does not match the manifest")
    for name, digest in declared.items():
        if observed.get(name) != digest:
            problems.append(f"video bytes changed since publication: {name}")

    production = json.loads(
        (ROOT / "diagnostics_output/pact_place_v104_review_production"
         "/production_manifest.json").read_text()
    )
    if not production["eligibility"]["production_pack_passed"]:
        problems.append("the six-row production pack did not pass")
    causal = json.loads(
        (ROOT / "diagnostics_output/pact_place_v104_causal/causal.json").read_text()
    )
    if not causal.get("causal_passed"):
        problems.append("the panel causal check did not pass")
    return {
        "passed": not problems,
        "problems": problems,
        "n_videos": len(observed),
        "video_sha256": observed,
    }


def assert_phase0_v2_approval(
    approval: dict[str, Any] | None,
    expected: dict[str, str],
    *,
    video_names: list[str],
) -> None:
    """An owner record must exist, be theirs, and bind every recomputed hash."""
    if not approval:
        raise ApprovalError("Phase 0 requires an owner-supplied human_approval.json")
    if approval.get("decision") != "approve_phase0":
        raise ApprovalError(f"Phase 0 refused: decision={approval.get('decision')!r}")
    if approval.get("created_by_agent"):
        raise ApprovalError("Phase 0 refuses an agent-created approval record")
    missing = sorted(key for key in expected if key not in approval)
    if missing:
        raise ApprovalError(f"Phase 0 approval is missing bindings: {missing}")
    for key, digest in expected.items():
        if approval.get(key) != digest:
            raise ApprovalError(
                f"Phase 0 approval binding is stale for {key}: "
                f"{approval.get(key)!r} != {digest!r}"
            )
    reviewed = approval.get("reviewed_videos")
    if not isinstance(reviewed, list):
        raise ApprovalError("Phase 0 approval must list the reviewed videos")
    if sorted(reviewed) != sorted(video_names):
        extra = sorted(set(reviewed) - set(video_names))
        absent = sorted(set(video_names) - set(reviewed))
        raise ApprovalError(
            f"Phase 0 approval video inventory is wrong: extra={extra} missing={absent}"
        )
    if len(reviewed) != N_REVIEW_V2_VIDEOS:
        raise ApprovalError(
            f"Phase 0 approval must review exactly {N_REVIEW_V2_VIDEOS} videos"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=ROOT / REVIEW_V2_ROOT)
    parser.add_argument("--approval", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / PHASE0_V2_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)

    review_root = args.review_root.resolve()
    approval_path = (
        args.approval.resolve()
        if args.approval
        else review_root / "human_approval.json"
    )

    # Everything below the approval gate is refused before any directory or row
    # is created. The output root is not touched until the record validates.
    contract = build_contract()
    provenance = verify_protected_artifacts(contract["protected_artifacts"])
    if not provenance["passed"]:
        raise SystemExit(f"protected artifact drift: {provenance['mismatches'][:3]}")
    packet = verify_review_packet(review_root)
    if not packet["passed"]:
        raise SystemExit(f"review packet is not gate-ready: {packet['problems']}")
    if not approval_path.is_file():
        raise ApprovalError(
            f"missing owner approval: {approval_path}. "
            "The agent must not create this file."
        )
    approval = json.loads(approval_path.read_text())
    expected = expected_bindings_v2(review_root)
    assert_phase0_v2_approval(
        approval, expected, video_names=sorted(packet["video_sha256"])
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = list(contract["gate_rows"])
    if len(rows) != N_GATE_ROWS:
        raise RuntimeError("the Phase-0 manifest must contain exactly 24 rows")
    review_ids = {row["episode_id"] for row in contract["review_rows"]}
    if {row["episode_id"] for row in rows} & review_ids:
        raise RuntimeError("a review row leaked into the Phase-0 manifest")

    config = {
        "schema_version": f"{SCHEMA_PREFIX}_phase0_config_v1",
        "contract_version": CONTRACT_VERSION_V2,
        "environment_version": ENVIRONMENT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "scoped_production_sha256": scoped_production_sha256(),
        "gate_v2_implementation_sha256": gate_v2_implementation_sha256(),
        "scene_sha256": contract["scene_sha256"],
        "sampler_class": SAMPLER_CLASS,
        "gate_stream": GATE_STREAM,
        "gate_master_seed": GATE_MASTER_SEED,
        "approval_path": str(approval_path),
        "approval_sha256": sha256_file(approval_path),
        "pass_threshold": 20,
        "min_clean_per_side": 9,
        "clearance_floor_m": GATE_MIN_CLEARANCE_M,
        "frozen_before_row_0": True,
        "no_substitution_reseeding_or_threshold_change": True,
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
        "schema_version": f"{SCHEMA_PREFIX}_phase0_gate_v1",
        "contract_version": CONTRACT_VERSION_V2,
        "contract_sha256": contract["contract_sha256"],
        "config_sha256": config["config_sha256"],
        "approval_sha256": config["approval_sha256"],
        "scoped_production_sha256": scoped_production_sha256(),
        "review_manifest_sha256": expected["review_manifest_sha256"],
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
        # empty_authorization() is spread FIRST on purpose: spreading it after
        # would silently reset a passing gate back to false.
        "phase0_passed": bool(eligibility["phase0_passed"]),
        "permanent_stop": not bool(eligibility["phase0_passed"]),
    }
    digest = write_immutable_create_only(output_root / "gate.json", document)
    print(
        json.dumps(
            {
                "phase0_passed": document["phase0_passed"],
                "clean_successes": eligibility["clean_successes"],
                "clean_by_side": eligibility["clean_by_side"],
                "artifact_sha256": digest,
                "authorizes_collection": False,
                "authorizes_training": False,
                "authorizes_evaluation": False,
            },
            indent=2,
        )
    )
    return 0 if document["phase0_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
