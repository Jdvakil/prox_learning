#!/usr/bin/env python3
"""Analyze attempt-2 expert screens and apply the frozen gates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pact_geometry_generalization_v2_contract import (  # noqa: E402
    CANDIDATES,
    load_manifest,
    select_candidates,
    sha256_file,
    sha256_payload,
)
from run_pact_geometry_v2_expert_supervisor import result_path  # noqa: E402


V1_MANIFEST_SHA256 = "33e48ab83dfe398fbeb78f64565312c48a5a8b09cb1a873a2a2521e06fcbe7b2"
V1_EXPERT_SCREEN_SHA256 = "3bf7d5c8f86814b9c10308c10cf1576488e992d0d20564359cb911d312d78a2c"


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def add_self_hash(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result[key] = sha256_payload(result)
    return result


def load_dispatch(output_root: Path, phase: str, manifest_sha256: str) -> dict[str, Any]:
    path = output_root / phase / "dispatch.json"
    value = json.loads(path.read_text())
    payload = dict(value)
    observed = payload.pop("dispatch_sha256", None)
    if observed != sha256_payload(payload):
        raise RuntimeError(f"dispatch self-hash mismatch: {path}")
    if value.get("manifest_sha256") != manifest_sha256 or not value.get("reconciled"):
        raise RuntimeError(f"screen dispatch did not reconcile: {path}")
    return value


def derive_row(result: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    if result.get("episode_id") != row["episode_id"]:
        raise RuntimeError("result episode identity mismatch")
    if result.get("row_sha256") != row["row_sha256"]:
        raise RuntimeError("result row hash mismatch")
    status = result.get("status")
    if status not in {"complete", "sampling_failure"}:
        raise RuntimeError(f"non-scientific expert status: {status!r}")
    totals = result.get("contact_audit", {}).get("contact_class_totals", {})
    hazard = int(totals.get("hazard_bar", 0))
    other = int(totals.get("other_environment", 0))
    task_success = result.get("task_success") is True
    clean = bool(status == "complete" and task_success and hazard == 0 and other == 0)
    if result.get("clean_success") is not clean:
        raise RuntimeError(
            f"stored clean-success disagrees with independent derivation: {row['episode_id']}"
        )
    return {
        "episode_id": row["episode_id"],
        "condition_id": row["condition_id"],
        "instance_index": int(row["instance_index"]),
        "intrusion_side": row["intrusion_side"],
        "status": status,
        "task_success": task_success,
        "clean_success": clean,
        "hazard_bar_contact_samples": hazard,
        "other_environment_contact_samples": other,
    }


def summarize(rows: list[dict[str, Any]], threshold: int) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for condition_id in sorted({row["condition_id"] for row in rows}):
        cell = [row for row in rows if row["condition_id"] == condition_id]
        clean = sum(row["clean_success"] for row in cell)
        task = sum(row["task_success"] for row in cell)
        sampling_failures = sum(row["status"] == "sampling_failure" for row in cell)
        hazard_episodes = sum(row["hazard_bar_contact_samples"] > 0 for row in cell)
        other_episodes = sum(row["other_environment_contact_samples"] > 0 for row in cell)
        table[condition_id] = {
            "n": len(cell),
            "task_successes": task,
            "clean_successes": clean,
            "clean_success_rate": clean / len(cell),
            "hazard_contact_episodes": hazard_episodes,
            "other_environment_contact_episodes": other_episodes,
            "hazard_bar_contact_samples": sum(
                row["hazard_bar_contact_samples"] for row in cell
            ),
            "other_environment_contact_samples": sum(
                row["other_environment_contact_samples"] for row in cell
            ),
            "sampling_failures": sampling_failures,
            "gate_threshold": threshold,
            "passed": clean >= threshold,
            "action": "eligible" if clean >= threshold else "drop_without_retuning",
        }
    return table


def read_phase_rows(
    manifest: dict[str, Any], output_root: Path, phase: str, selected: list[str] | None
) -> list[dict[str, Any]]:
    source_rows = (
        manifest["phase0a_rows"]
        if phase == "phase0a"
        else manifest["phase0b_candidate_rows"]
    )
    if selected is not None:
        source_rows = [row for row in source_rows if row["condition_id"] in selected]
    derived = []
    for row in source_rows:
        path = result_path(output_root, phase, row)
        if not path.exists():
            raise RuntimeError(f"missing expert result: {path}")
        derived.append(derive_row(json.loads(path.read_text()), row))
    return derived


def verify_v1_carried_screen() -> dict[str, Any]:
    manifest_path = ROOT / "configs" / "pact_geometry_generalization_v1.json"
    screen_path = ROOT / "diagnostics_output" / "pact_geometry_generalization" / "expert_screen.json"
    manifest = json.loads(manifest_path.read_text())
    screen = json.loads(screen_path.read_text())
    if manifest.get("manifest_sha256") != V1_MANIFEST_SHA256:
        raise RuntimeError("attempt-1 manifest identity changed")
    if screen.get("expert_screen_sha256") != V1_EXPERT_SCREEN_SHA256:
        raise RuntimeError("attempt-1 expert screen identity changed")
    if screen["conditions"]["C0"]["clean_successes"] != 11:
        raise RuntimeError("carried C0 screen result changed")
    if screen["conditions"]["C2"]["clean_successes"] != 12:
        raise RuntimeError("carried C2 screen result changed")
    return {
        "attempt1_manifest_sha256": V1_MANIFEST_SHA256,
        "attempt1_expert_screen_sha256": V1_EXPERT_SCREEN_SHA256,
        "attempt1_manifest_file_sha256": sha256_file(manifest_path),
        "attempt1_expert_screen_file_sha256": sha256_file(screen_path),
        "C0": {"clean_successes": 11, "n": 12, "passed": True},
        "C2": {"clean_successes": 12, "n": 12, "passed": True},
    }


def analyze_phase0a(manifest: dict[str, Any], output_root: Path) -> None:
    dispatch = load_dispatch(output_root, "phase0a", manifest["manifest_sha256"])
    rows = read_phase_rows(manifest, output_root, "phase0a", None)
    if len(rows) != 56:
        raise RuntimeError(f"phase0a expected 56 results, found {len(rows)}")
    threshold = int(manifest["phase0a_gate"]["minimum_clean_successes"])
    conditions = summarize(rows, threshold)
    pass_fail = {condition_id: conditions[condition_id]["passed"] for condition_id in CANDIDATES}
    selected = select_candidates(pass_fail)
    envelope = add_self_hash(
        {
            "schema_version": "pact_geometry_generalization_v2_envelope_map",
            "manifest_sha256": manifest["manifest_sha256"],
            "dispatch_sha256": dispatch["dispatch_sha256"],
            "post_hoc": False,
            "gate_frozen_before_execution": True,
            "clean_success_definition": "task success and zero hazard_bar and zero other_environment contact samples",
            "threshold": {"minimum_clean_successes": threshold, "n": 8},
            "conditions": conditions,
            "row_count": len(rows),
            "rows": rows,
            "pass_fail": pass_fail,
            "selected_candidate_ids_by_frozen_priority": selected,
            "enough_candidates_for_phase0b": len(selected) == 2,
        },
        "envelope_map_sha256",
    )
    write_json_atomic(output_root / "envelope_map.json", envelope)
    selection = add_self_hash(
        {
            "schema_version": "pact_geometry_generalization_v2_phase0b_selection",
            "manifest_sha256": manifest["manifest_sha256"],
            "envelope_map_sha256": envelope["envelope_map_sha256"],
            "selection_rule": manifest["selection_priority_frozen_before_phase0a"],
            "selection_inputs_pass_fail_only": pass_fail,
            "selected_candidate_ids": selected,
            "selected_candidate_axes": [CANDIDATES[item]["axis"] for item in selected],
            "phase0b_authorized": len(selected) == 2,
            "carried_screen": verify_v1_carried_screen(),
            "stop_reason": None if len(selected) == 2 else "fewer_than_two_single_axis_candidates_passed_phase0a",
        },
        "phase0b_selection_sha256",
    )
    write_json_atomic(output_root / "phase0b_selection.json", selection)
    print(json.dumps({"conditions": conditions, "selected": selected}, sort_keys=True))


def analyze_phase0b(manifest: dict[str, Any], output_root: Path) -> None:
    selection_path = output_root / "phase0b_selection.json"
    selection = json.loads(selection_path.read_text())
    payload = dict(selection)
    observed = payload.pop("phase0b_selection_sha256", None)
    if observed != sha256_payload(payload) or not selection.get("phase0b_authorized"):
        raise RuntimeError("phase0b was not authorized by a valid frozen selection")
    selected = selection["selected_candidate_ids"]
    dispatch = load_dispatch(output_root, "phase0b", manifest["manifest_sha256"])
    if dispatch.get("selection_sha256") != observed:
        raise RuntimeError("phase0b dispatch used a different selection")
    rows = read_phase_rows(manifest, output_root, "phase0b", selected)
    if len(rows) != 24:
        raise RuntimeError(f"phase0b expected 24 results, found {len(rows)}")
    threshold = int(manifest["phase0b_gate"]["minimum_clean_successes"])
    conditions = summarize(rows, threshold)
    passed_new = [item for item in selected if conditions[item]["passed"]]
    shifted = ["C2", *passed_new]
    proceed = len(shifted) >= 2
    carried = verify_v1_carried_screen()
    screen = add_self_hash(
        {
            "schema_version": "pact_geometry_generalization_v2_expert_screen",
            "manifest_sha256": manifest["manifest_sha256"],
            "phase0a_envelope_map_sha256": json.loads(
                (output_root / "envelope_map.json").read_text()
            )["envelope_map_sha256"],
            "phase0b_selection_sha256": observed,
            "phase0b_dispatch_sha256": dispatch["dispatch_sha256"],
            "gate_frozen_before_execution": True,
            "carried_attempt1_conditions": carried,
            "phase0b_conditions": conditions,
            "selected_candidate_ids": selected,
            "passed_new_shifted_condition_ids": passed_new,
            "surviving_condition_ids": ["C0", *shifted],
            "surviving_shifted_condition_ids": shifted,
            "continue_to_policy_evaluation": proceed,
            "main_policy_rollout_count": (1 + len(shifted)) * 3 * 3 * 25 if proceed else 0,
            "stop_reason": None if proceed else "fewer_than_two_total_shifted_conditions_survived",
            "phase0b_row_count": len(rows),
            "phase0b_rows": rows,
        },
        "expert_screen_sha256",
    )
    write_json_atomic(output_root / "expert_screen.json", screen)
    print(json.dumps({"conditions": conditions, "proceed": proceed}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("phase0a", "phase0b"))
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.phase == "phase0a":
        analyze_phase0a(manifest, args.output_root)
    else:
        analyze_phase0b(manifest, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
