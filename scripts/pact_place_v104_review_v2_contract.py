#!/usr/bin/env python3
"""V10.4 review-v2 contract: preserve v1, bridge its provenance, repair controls.

Nothing here changes production geometry, routing, speeds, seeds, or results.
The six V10.4 production episodes are reused through a scoped provenance
bridge; this module states exactly what that bridge is allowed to forgive (one
superseded review runner) and fails closed on everything else.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v104_contract import (  # noqa: E402
    CAUSAL_ROOT,
    GATE_MIN_CLEARANCE_M,
    ImmutableArtifactError,
    MIN_GATE_CLEAN,
    MIN_GATE_CLEAN_PER_SIDE,
    N_GATE_ROWS,
    POLICY_TIMESTEP_MS,
    PREFLIGHT_ROOT,
    PRODUCTION_ROOT,
    REVIEW_FPS,
    REVIEW_FRAME_STRIDE,
    REVIEW_MIN_CLEARANCE_M,
    empty_authorization,
    sha256_bytes_of,
    sha256_payload,
    write_immutable_create_only,
)
from pact_place_v104_geometry import SCENE_XML_RELATIVE_V104  # noqa: E402

CONTRACT_VERSION_V2 = "pact_place_v104_review_packet_v2"
SCHEMA_PREFIX = "pact_place_v104_review_v2"

REVIEW_V2_ROOT = "diagnostics_output/pact_place_v104_review_v2"
PHASE0_V2_ROOT = "diagnostics_output/pact_place_v104_phase0_v2"

SCENE_METADATA_RELATIVE_V104 = (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v10_4_metadata.json"
)
BASE_SCENE_V3_RELATIVE = (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v3.xml"
)
BASE_SCENE_V5_RELATIVE = (
    "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/"
    "pact_place_corridor_v5.xml"
)

# ---------------------------------------------------------------------------
# What v1 actually produced. These are audited constants, not recomputed
# values: if the tree drifts, the bridge must fail rather than re-baseline.
# ---------------------------------------------------------------------------
V1_INPUTS: dict[str, dict[str, str]] = {
    "preflight": {
        "path": f"{PREFLIGHT_ROOT}/preflight.json",
        "payload_sha256": (
            "fe64e285332a3c530cab30599d2b862823a3c1e2db661a6961a0a1461f2c41d5"
        ),
        "file_sha256": (
            "134f79cf4fa6201dc7542386cc4e8422eca90b394607033035ed9e3ad3e6b129"
        ),
    },
    "production": {
        "path": f"{PRODUCTION_ROOT}/production_manifest.json",
        "payload_sha256": (
            "fdcf757b4bff512c71c6e3ac241c151742523c89ec7531b46132af715e92b3af"
        ),
        "file_sha256": (
            "13707b197a42a4f5934c7fee5dc66e429c5debb4f73266e375e3cf3e570db78f"
        ),
    },
    "causal": {
        "path": f"{CAUSAL_ROOT}/causal.json",
        "payload_sha256": (
            "a30c863d61537edb58d24cc91b13291fa5d9efc47c7521b08b2304943f2f2ffc"
        ),
        "file_sha256": (
            "5216216e7c5783b286fff75931177241bb0c205cd496632fef53a5652520ee96"
        ),
    },
}

PRODUCTION_SCENE_SHA256 = (
    "01d8adf34808a9f419cb3a9d07668ec1069d3a5acfa8cb01885c622ea09876f7"
)
SCENE_METADATA_SHA256 = (
    "7df36c5e26364f9b5bd6da98e59108d7745c2dbd1270cc3ca73d307a656b809c"
)

# The contract/implementation aggregate the v1 artifacts were produced under.
# The later live aggregate (455379b8.../bf4af91...) is NOT what ran; it is the
# value the tree reached after the remaining v1 runners and tests were written.
EXECUTED_V1_CONTRACT_SHA256 = (
    "eb8f1174142976561495827b4cd3a8609569465fbce23c7a46b4a53885fe875e"
)
EXECUTED_V1_IMPLEMENTATION_SHA256 = (
    "bd135e68303618ceefbe57f1ad8a6e6a5d81ae2d29930f641f04648d4847ec90"
)

# The entire provenance-bridge allowlist: one superseded review runner.
BRIDGE_ALLOWLIST_PATH = "scripts/run_pact_place_v104_review_video.py"
BRIDGE_ALLOWLIST_OLD_SHA256 = (
    "b40e5a0fb2a9e469e04eef79257575d1046246c4c257ca1e3bac3a26e03a8fe9"
)
BRIDGE_ALLOWLIST_NEW_SHA256 = (
    "ddf962255726969945378779790f050c90b14098d1909bc022ce8036ccacd068"
)

# ---------------------------------------------------------------------------
# Deterministic selection, frozen for v2.
# ---------------------------------------------------------------------------
SUCCESS_ROLES: tuple[tuple[int, str], ...] = ((0, "left"), (3, "right"), (4, "left"))

CONTROL_ORDER_V2 = ("left_lobe_contact", "right_lobe_contact", "stem_contact")
CONTROL_SPEC: dict[str, dict[str, Any]] = {
    "left_lobe_contact": {"source_role_index": 0, "component": "lobe_0"},
    "right_lobe_contact": {"source_role_index": 3, "component": "lobe_1"},
    "stem_contact": {"source_role_index": 0, "component": "stem_0"},
}

# 0.000-0.200 m inclusive in 0.001 m increments: 201 points. Diagnostic-only.
CONTROL_SHIFT_GRID_V2_M: tuple[float, ...] = tuple(
    round(0.001 * index, 3) for index in range(0, 201)
)
CONTROL_PENETRATION_BAND_M = (0.005, 0.030)

# Audited anchors. A certified control must reproduce these within 0.1 mm.
CONTROL_ANCHORS: dict[str, dict[str, Any]] = {
    "left_lobe_contact": {
        "shift_m": 0.175,
        "penetration_m": 0.005044,
        "max_frame": 88,
        "limiting_robot_body": "fr3_link7",
    },
    "right_lobe_contact": {
        "shift_m": 0.132,
        "penetration_m": 0.005239,
        "max_frame": 245,
        "limiting_robot_body": "gripper/base",
    },
    "stem_contact": {
        "shift_m": 0.083,
        "penetration_m": 0.005455,
        "max_frame": 212,
        "limiting_robot_body": "fr3_link7",
    },
}
CONTROL_ANCHOR_TOLERANCE_M = 0.0001

# Trimmed contact-centered control windows.
CONTROL_WINDOW_LEAD_FRAMES = 45
CONTROL_WINDOW_TRAIL_FRAMES = 15
CONTROL_WINDOW_ANCHORS: dict[str, dict[str, int]] = {
    "left_lobe_contact": {"first_frame": 40, "last_frame": 89, "n_frames": 50},
    "right_lobe_contact": {"first_frame": 197, "last_frame": 260, "n_frames": 64},
    "stem_contact": {"first_frame": 164, "last_frame": 227, "n_frames": 64},
}
# Recorded in the certificate, excluded from the left-lobe clip. These are two
# different frames: the stem first touches at 90 (0.103 mm), which is what
# trims the clip, and only reaches its deepest 38.15 mm much later at 193. The
# plan quoted both numbers against frame 90; measurement separates them.
LEFT_LOBE_SECONDARY_STEM_FIRST_FRAME = 90
LEFT_LOBE_SECONDARY_STEM_MAX_PENETRATION_M = 0.03815
LEFT_LOBE_SECONDARY_STEM_MAX_FRAME = 193
LEFT_LOBE_SECONDARY_STEM_FRAME = LEFT_LOBE_SECONDARY_STEM_FIRST_FRAME

N_REVIEW_V2_VIDEOS = 6
N_REVIEW_V2_SUCCESSES = 3
N_REVIEW_V2_CONTROLS = 3

CONTROL_BANNER_LINES = (
    "DIAGNOSTIC NEGATIVE CONTROL",
    "TRIMMED CONTACT WINDOW",
    "NOT PRODUCTION GEOMETRY - NOT AN EPISODE",
)

# ---------------------------------------------------------------------------
# Implementation binding
# ---------------------------------------------------------------------------
# Production-affecting code and data. The review and gate runners are
# deliberately excluded: changing how a video is drawn cannot invalidate an
# episode that already ran, and pretending otherwise is what forced the v1
# aggregate to drift away from what actually executed.
SCOPED_PRODUCTION_PATHS = (
    "scripts/pact_place_v104_contract.py",
    "scripts/pact_place_v104_geometry.py",
    "scripts/pact_place_v104_clearance.py",
    "scripts/pact_place_v104_runtime.py",
    "scripts/run_pact_place_v104_preflight.py",
    "scripts/run_pact_place_v104_review_production.py",
    "scripts/run_pact_place_v104_causal.py",
    "scripts/pact_geom_distance.py",
    "scripts/run_pact_place_expert_screen.py",
    SCENE_XML_RELATIVE_V104,
    SCENE_METADATA_RELATIVE_V104,
    "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
    "submodules/molmospaces/molmo_spaces/tasks/pact_place_contact_audit.py",
)

REVIEW_V2_IMPLEMENTATION_PATHS = (
    "scripts/pact_place_v104_review_v2_contract.py",
    "scripts/pact_place_v104_control_certify.py",
    "scripts/run_pact_place_v104_review_v2.py",
)

GATE_V2_IMPLEMENTATION_PATHS = (
    "scripts/pact_place_v104_review_v2_contract.py",
    "scripts/run_pact_place_v104_phase0_v2.py",
)


def _hashes(paths) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in paths:
        target = ROOT / path
        out[path] = sha256_bytes_of(target) if target.is_file() else "absent"
    return out


def scoped_production_hashes() -> dict[str, str]:
    return _hashes(SCOPED_PRODUCTION_PATHS)


def scoped_production_sha256() -> str:
    return sha256_payload(scoped_production_hashes())


def review_v2_implementation_hashes() -> dict[str, str]:
    return _hashes(REVIEW_V2_IMPLEMENTATION_PATHS)


def review_v2_implementation_sha256() -> str:
    return sha256_payload(review_v2_implementation_hashes())


def gate_v2_implementation_hashes() -> dict[str, str]:
    return _hashes(GATE_V2_IMPLEMENTATION_PATHS)


def gate_v2_implementation_sha256() -> str:
    return sha256_payload(gate_v2_implementation_hashes())


# ---------------------------------------------------------------------------
# Provenance bridge
# ---------------------------------------------------------------------------
def write_immutable_text_create_only(path: Path, text: str) -> str:
    """Atomic create-if-absent for a text artifact. Returns its SHA-256."""
    import os
    import tempfile

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise ImmutableArtifactError(
                f"refusing to replace an existing artifact: {target}"
            ) from error
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256_bytes_of(target)


class ProvenanceBridgeError(RuntimeError):
    """A production-affecting input no longer matches what v1 executed."""


def _payload_sha256_of(document: dict[str, Any]) -> str:
    """Recompute a document's self-hash the way the writer computed it."""
    payload = {k: v for k, v in document.items() if k != "artifact_sha256"}
    return sha256_payload(payload)


def verify_v1_inputs(root: Path | None = None) -> dict[str, Any]:
    """Recompute payload and raw-file hashes. Never trusts embedded values."""
    base = Path(root) if root is not None else ROOT
    checks: list[dict[str, Any]] = []
    for name, spec in V1_INPUTS.items():
        target = base / spec["path"]
        entry: dict[str, Any] = {"input": name, "path": spec["path"]}
        if not target.is_file():
            entry.update({"present": False, "passed": False, "reason": "absent"})
            checks.append(entry)
            continue
        document = json.loads(target.read_text())
        embedded = document.get("artifact_sha256")
        recomputed = _payload_sha256_of(document)
        file_digest = sha256_bytes_of(target)
        entry.update(
            {
                "present": True,
                "expected_payload_sha256": spec["payload_sha256"],
                "embedded_payload_sha256": embedded,
                "recomputed_payload_sha256": recomputed,
                "expected_file_sha256": spec["file_sha256"],
                "observed_file_sha256": file_digest,
                "payload_self_consistent": embedded == recomputed,
                "payload_matches_expected": recomputed == spec["payload_sha256"],
                "file_matches_expected": file_digest == spec["file_sha256"],
            }
        )
        entry["passed"] = bool(
            entry["payload_self_consistent"]
            and entry["payload_matches_expected"]
            and entry["file_matches_expected"]
        )
        checks.append(entry)
    return {"checks": checks, "passed": all(item["passed"] for item in checks)}


def verify_scene_and_metadata(root: Path | None = None) -> dict[str, Any]:
    base = Path(root) if root is not None else ROOT
    scene = base / SCENE_XML_RELATIVE_V104
    metadata = base / SCENE_METADATA_RELATIVE_V104
    observed_scene = sha256_bytes_of(scene) if scene.is_file() else "absent"
    observed_metadata = sha256_bytes_of(metadata) if metadata.is_file() else "absent"
    return {
        "scene_path": SCENE_XML_RELATIVE_V104,
        "expected_scene_sha256": PRODUCTION_SCENE_SHA256,
        "observed_scene_sha256": observed_scene,
        "metadata_path": SCENE_METADATA_RELATIVE_V104,
        "expected_metadata_sha256": SCENE_METADATA_SHA256,
        "observed_metadata_sha256": observed_metadata,
        "passed": (
            observed_scene == PRODUCTION_SCENE_SHA256
            and observed_metadata == SCENE_METADATA_SHA256
        ),
    }


def verify_scoped_implementation(root: Path | None = None) -> dict[str, Any]:
    """Every file the old preflight bound must match, except the allowlisted one."""
    base = Path(root) if root is not None else ROOT
    preflight_path = base / V1_INPUTS["preflight"]["path"]
    if not preflight_path.is_file():
        return {"passed": False, "reason": "preflight absent"}
    bound = json.loads(preflight_path.read_text())["implementation_files"]
    matched: list[str] = []
    bridged: list[dict[str, Any]] = []
    mismatched: list[dict[str, Any]] = []
    for path, old in sorted(bound.items()):
        target = base / path
        observed = sha256_bytes_of(target) if target.is_file() else "absent"
        if observed == old:
            matched.append(path)
            continue
        if (
            path == BRIDGE_ALLOWLIST_PATH
            and old == BRIDGE_ALLOWLIST_OLD_SHA256
            and observed == BRIDGE_ALLOWLIST_NEW_SHA256
        ):
            bridged.append(
                {"path": path, "old_sha256": old, "current_sha256": observed}
            )
            continue
        mismatched.append(
            {"path": path, "expected": old, "observed": observed}
        )
    return {
        "n_bound_files": len(bound),
        "n_matched": len(matched),
        "matched": matched,
        "bridged": bridged,
        "mismatched": mismatched,
        "allowlist": [BRIDGE_ALLOWLIST_PATH],
        "allowlist_size": 1,
        # The bridge forgives exactly one path and only that path's exact
        # old->new transition. Anything else is a hard failure.
        "passed": not mismatched and len(bridged) <= 1,
    }


def verify_production_rows(root: Path | None = None) -> dict[str, Any]:
    """Row bindings, config binding, result and trajectory bytes."""
    base = Path(root) if root is not None else ROOT
    manifest_path = base / V1_INPUTS["production"]["path"]
    if not manifest_path.is_file():
        return {"passed": False, "reason": "production manifest absent"}
    manifest = json.loads(manifest_path.read_text())
    production_root = manifest_path.parent
    config_path = production_root / "config.json"
    # config_sha256 is a self-hash: the writer hashed the config payload before
    # inserting the key, so it is neither the raw file hash nor the payload
    # hash of the file as it now stands. Recompute it the way it was made.
    if config_path.is_file():
        config_document = json.loads(config_path.read_text())
        config_file_sha256 = sha256_bytes_of(config_path)
        config_digest = sha256_payload(
            {k: v for k, v in config_document.items() if k != "config_sha256"}
        )
        config_embedded = config_document.get("config_sha256")
    else:
        config_file_sha256 = "absent"
        config_digest = "absent"
        config_embedded = None
    rows = {int(row["role_index"]): row for row in manifest["rows"]}
    entries: list[dict[str, Any]] = []
    failures: list[str] = []
    for result in sorted(manifest["results"], key=lambda item: int(item["role_index"])):
        role = int(result["role_index"])
        row = rows[role]
        directory = (
            production_root
            / "expert_screen_rows"
            / f"{role:02d}_{row['episode_id'][:16]}"
        )
        result_path = directory / "result.json"
        trajectory_path = directory / "trajectory.json"
        entry: dict[str, Any] = {
            "role_index": role,
            "intrusion_side": str(row["intrusion_side"]),
            "episode_id": str(row["episode_id"]),
            "row_sha256": str(row["row_sha256"]),
            "result_present": result_path.is_file(),
            "trajectory_present": trajectory_path.is_file(),
        }
        if not (entry["result_present"] and entry["trajectory_present"]):
            failures.append(f"role {role}: missing retained payload")
            entry["passed"] = False
            entries.append(entry)
            continue
        stored_result = json.loads(result_path.read_text())
        stored_trajectory = json.loads(trajectory_path.read_text())
        entry.update(
            {
                "result_file_sha256": sha256_bytes_of(result_path),
                "trajectory_file_sha256": sha256_bytes_of(trajectory_path),
                "n_frames": int(stored_trajectory["n"]),
                "clean_success": bool(result.get("clean_success")),
                "trajectory_row_binding_ok": (
                    stored_trajectory.get("row_sha256") == row["row_sha256"]
                ),
                "trajectory_config_binding_ok": (
                    stored_trajectory.get("config_sha256") == config_digest
                ),
                "result_row_binding_ok": (
                    stored_result.get("episode_id") == row["episode_id"]
                ),
                "row_implementation_ok": (
                    row.get("implementation_sha256")
                    == EXECUTED_V1_IMPLEMENTATION_SHA256
                ),
                "row_scene_ok": (
                    row.get("pact_v104_scene_sha256") == PRODUCTION_SCENE_SHA256
                ),
            }
        )
        problems = [
            key
            for key in (
                "trajectory_row_binding_ok",
                "trajectory_config_binding_ok",
                "result_row_binding_ok",
                "row_implementation_ok",
                "row_scene_ok",
                "clean_success",
            )
            if not entry[key]
        ]
        entry["passed"] = not problems
        if problems:
            failures.append(f"role {role}: {sorted(problems)}")
        entries.append(entry)
    n_clean = sum(1 for item in entries if item.get("clean_success"))
    by_side: dict[str, int] = {"left": 0, "right": 0}
    for item in entries:
        if item.get("clean_success"):
            by_side[item["intrusion_side"]] += 1
    return {
        "config_sha256": config_digest,
        "config_file_sha256": config_file_sha256,
        "config_embedded_sha256": config_embedded,
        "config_self_consistent": config_embedded == config_digest,
        "config_binding_ok": (
            config_digest == manifest.get("config_sha256")
            and config_embedded == config_digest
        ),
        "n_rows": len(entries),
        "rows": entries,
        "n_clean": n_clean,
        "clean_by_side": by_side,
        "reconciled_all_strict_clean": n_clean == len(entries) == 6,
        "min_observed_clearance_m": manifest["eligibility"]["min_observed_clearance_m"],
        "pendant_contact_rows": manifest["eligibility"]["pendant_contact_rows"],
        "failures": failures,
        "replacement_episodes_generated": False,
        "passed": (
            not failures
            and config_digest == manifest.get("config_sha256")
            and config_embedded == config_digest
            and n_clean == 6
            and by_side == {"left": 3, "right": 3}
        ),
    }


def verify_causal(root: Path | None = None) -> dict[str, Any]:
    base = Path(root) if root is not None else ROOT
    causal_path = base / V1_INPUTS["causal"]["path"]
    if not causal_path.is_file():
        return {"passed": False, "reason": "causal artifact absent"}
    causal = json.loads(causal_path.read_text())
    raw = causal_path.parent / "raw"
    npz: dict[str, str] = {}
    for side in ("left", "right"):
        target = raw / f"{side}.npz"
        npz[f"{side}.npz"] = sha256_bytes_of(target) if target.is_file() else "absent"
    return {
        "causal_passed": bool(causal.get("causal_passed")),
        "npz_sha256": npz,
        "npz_present": all(value != "absent" for value in npz.values()),
        "production_manifest_binding_ok": (
            causal.get("production_manifest_sha256")
            == V1_INPUTS["production"]["payload_sha256"]
        ),
        "passed": bool(
            causal.get("causal_passed")
            and all(value != "absent" for value in npz.values())
            and causal.get("production_manifest_sha256")
            == V1_INPUTS["production"]["payload_sha256"]
        ),
    }


def build_provenance_bridge(root: Path | None = None) -> dict[str, Any]:
    """The complete v1 -> v2 bridge. Fails closed on any drift."""
    inputs = verify_v1_inputs(root)
    scene = verify_scene_and_metadata(root)
    implementation = verify_scoped_implementation(root)
    rows = verify_production_rows(root)
    causal = verify_causal(root)
    sections = {
        "v1_inputs": inputs,
        "scene_and_metadata": scene,
        "scoped_implementation": implementation,
        "production_rows": rows,
        "causal": causal,
    }
    failed = sorted(name for name, item in sections.items() if not item.get("passed"))
    document = {
        "schema_version": f"{SCHEMA_PREFIX}_provenance_bridge_v1",
        "contract_version": CONTRACT_VERSION_V2,
        "executed_v1_contract_sha256": EXECUTED_V1_CONTRACT_SHA256,
        "executed_v1_implementation_sha256": EXECUTED_V1_IMPLEMENTATION_SHA256,
        "executed_v1_note": (
            "The v1 artifacts were produced under this contract/implementation "
            "pair. The later live aggregate is a different value and is not "
            "what executed; it is recorded separately and never bound."
        ),
        "live_aggregate_is_not_the_executed_aggregate": True,
        "scoped_production_sha256": scoped_production_sha256(),
        "scoped_production_hashes": scoped_production_hashes(),
        "review_v2_implementation_sha256": review_v2_implementation_sha256(),
        "gate_v2_implementation_sha256": gate_v2_implementation_sha256(),
        **sections,
        "failed_sections": failed,
        "verified_from_file_bytes": True,
        "trusted_embedded_hashes": False,
        **empty_authorization(),
        "bridge_passed": not failed,
    }
    return document


__all__ = [
    "BRIDGE_ALLOWLIST_NEW_SHA256",
    "BRIDGE_ALLOWLIST_OLD_SHA256",
    "BRIDGE_ALLOWLIST_PATH",
    "CONTRACT_VERSION_V2",
    "CONTROL_ANCHORS",
    "CONTROL_ANCHOR_TOLERANCE_M",
    "CONTROL_BANNER_LINES",
    "CONTROL_ORDER_V2",
    "CONTROL_PENETRATION_BAND_M",
    "CONTROL_SHIFT_GRID_V2_M",
    "CONTROL_SPEC",
    "CONTROL_WINDOW_ANCHORS",
    "CONTROL_WINDOW_LEAD_FRAMES",
    "CONTROL_WINDOW_TRAIL_FRAMES",
    "EXECUTED_V1_CONTRACT_SHA256",
    "EXECUTED_V1_IMPLEMENTATION_SHA256",
    "GATE_V2_IMPLEMENTATION_PATHS",
    "ImmutableArtifactError",
    "LEFT_LOBE_SECONDARY_STEM_FIRST_FRAME",
    "LEFT_LOBE_SECONDARY_STEM_FRAME",
    "LEFT_LOBE_SECONDARY_STEM_MAX_FRAME",
    "LEFT_LOBE_SECONDARY_STEM_MAX_PENETRATION_M",
    "N_REVIEW_V2_CONTROLS",
    "N_REVIEW_V2_SUCCESSES",
    "N_REVIEW_V2_VIDEOS",
    "PHASE0_V2_ROOT",
    "PRODUCTION_SCENE_SHA256",
    "ProvenanceBridgeError",
    "REVIEW_V2_IMPLEMENTATION_PATHS",
    "REVIEW_V2_ROOT",
    "SCENE_METADATA_RELATIVE_V104",
    "SCENE_METADATA_SHA256",
    "SCOPED_PRODUCTION_PATHS",
    "SUCCESS_ROLES",
    "build_provenance_bridge",
    "empty_authorization",
    "gate_v2_implementation_sha256",
    "review_v2_implementation_sha256",
    "scoped_production_sha256",
    "sha256_bytes_of",
    "sha256_payload",
    "verify_causal",
    "verify_production_rows",
    "verify_scene_and_metadata",
    "verify_scoped_implementation",
    "verify_v1_inputs",
    "write_immutable_create_only",
    "write_immutable_text_create_only",
]
