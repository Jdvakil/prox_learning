#!/usr/bin/env python3
"""Finalize the Phase-0 hard stop for geometry generalization."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from pact_geometry_generalization_contract import load_manifest, sha256_file, sha256_payload


ROOT = Path(__file__).resolve().parents[1]
TOKEN = "GEOMETRY_TEST_INCONCLUSIVE"
FROZEN_CONTACT_HASHES = {
    "docs/PACT_CONTACT_ENDPOINT_DECISION.md": "6ea97ea789e1c5d57572c7e231b30c867881f0e21b81ad11f950cef9bf860f1b",
    "diagnostics_output/pact_contact_endpoint/analysis.json": "4bbf25c97472bbfd8f13c3352b6b1c8afae9284cd00351e0502149362ccde1f7",
    "diagnostics_output/pact_contact_endpoint/final_decision.json": "52ce515bc09771d9ecdc8a39e31111f66e7e632f5949e82ed2497908ac2458bf",
}


def validate_self_hash(document: dict[str, Any], key: str, label: str) -> str:
    payload = dict(document)
    observed = payload.pop(key, None)
    if observed != sha256_payload(payload):
        raise ValueError(f"{label} self-hash mismatch")
    return str(observed)


def write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_rows(manifest: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    results = []
    for row in manifest["expert_screen_rows"]:
        path = (
            output_root
            / "expert_screen_rows"
            / row["condition_id"]
            / f"{row['role_index']:02d}_{row['episode_id'][:16]}"
            / "result.json"
        )
        result = json.loads(path.read_text())
        if result.get("episode_id") != row["episode_id"] or result.get("row_sha256") != row["row_sha256"]:
            raise ValueError(f"expert result identity mismatch: {path}")
        results.append({**result, "_path": str(path)})
    return results


def condition_summary(condition_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    cell = [row for row in rows if row["condition_id"] == condition_id]
    hazard_episodes = sum(
        int(row.get("contact_audit", {}).get("contact_class_totals", {}).get("hazard_bar", 0)) > 0
        for row in cell
    )
    other_episodes = sum(
        int(row.get("contact_audit", {}).get("contact_class_totals", {}).get("other_environment", 0)) > 0
        for row in cell
    )
    clean = sum(row.get("clean_success") is True for row in cell)
    task = sum(row.get("task_success") is True for row in cell)
    return {
        "n": len(cell),
        "status_counts": dict(sorted(Counter(row["status"] for row in cell).items())),
        "task_successes": task,
        "task_success_rate": task / len(cell),
        "clean_successes": clean,
        "clean_success_rate": clean / len(cell),
        "hazard_contact_episodes": hazard_episodes,
        "other_environment_contact_episodes": other_episodes,
        "gate_threshold": 10,
        "passed": clean >= 10 and not any(
            row["status"] == "infrastructure_failure" for row in cell
        ),
        "row_result_paths": [row["_path"] for row in cell],
    }


def verified_models(registry: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for seed in (3101, 3102, 3103):
        for arm in ("ACT", "PACT"):
            record = registry["seeds"][str(seed)][arm]
            for kind, path_key, hash_key in (
                ("checkpoint", "checkpoint_path", "checkpoint_sha256"),
                ("dataset_stats", "dataset_stats_path", "dataset_stats_sha256"),
                ("surface_encoder", "surface_encoder_path", "surface_encoder_sha256"),
            ):
                raw = record.get(path_key)
                expected = record.get(hash_key)
                if raw is None or (kind, raw) in seen:
                    continue
                seen.add((kind, raw))
                observed = sha256_file(raw)
                if observed != expected:
                    raise ValueError(f"frozen model changed: {raw}")
                output.append(
                    {
                        "kind": kind,
                        "path": raw,
                        "sha256": observed,
                        "checkpoint_seed": seed,
                        "arm": arm,
                    }
                )
    return output


def report_text(
    *,
    manifest: dict[str, Any],
    expert: dict[str, Any],
    summaries: dict[str, Any],
    schedule: dict[str, Any],
) -> str:
    lines = [
        "# PACT held-out geometry generalization",
        "",
        "## Decision",
        "",
        "**GEOMETRY_TEST_INCONCLUSIVE.** The preregistered privileged-expert gate left only one solvable shifted condition, below the required minimum of two. No ACT, PACT, or PACT_PERMUTED policy rollout was launched, so these data make no claim about geometry generalization.",
        "",
        "## Frozen design",
        "",
        "The intended zero-shot evaluation used the existing frozen checkpoints on one fresh in-distribution control and three two-axis geometry shifts. Conditions were fixed before the expert screen; a failed condition was dropped without retuning. Policy evaluation required C0 plus at least two shifted conditions to pass at 10/12 clean expert successes.",
        "",
        "A clean expert success means task success with zero `hazard_bar` and zero `other_environment` contact entries. Target contact remains allowed.",
        "",
        "| Condition | Geometry | Task success | Clean success | Hazard-contact episodes | Other-environment episodes | Gate |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    geometry = {
        "C0": "in-distribution control",
        "C1": "PANEL_X 0.68, PANEL_Z 0.96",
        "C2": "PANEL_INNER_FACE_Y 0.070, aperture 0.70",
        "C3": "PANEL_X 0.55, aperture 1.00",
    }
    for condition in ("C0", "C1", "C2", "C3"):
        item = summaries[condition]
        lines.append(
            f"| {condition} | {geometry[condition]} | {item['task_successes']}/12 | {item['clean_successes']}/12 | {item['hazard_contact_episodes']}/12 | {item['other_environment_contact_episodes']}/12 | {'pass' if item['passed'] else 'drop'} |"
        )
    lines += [
        "",
        "C0 passed 11/12 and C2 passed 12/12. C1 passed only 4/12, with hazard contact in 8/12 episodes. C3 passed 5/12; it had hazard contact in 2/12 episodes and one pre-boundary sampling failure after exhausting its fixed retries. Thus the surviving set was C0 and C2, only one of which was shifted.",
        "",
        "## Hard stop and scientific interpretation",
        "",
        "The 900-row policy schedule was not frozen or executed. `schedule.json` and `dispatch.json` are explicit Phase-0 stop records with zero policy rows. There are no policy outcomes to analyze, no C0 modality-gap reproduction result, and no shifted PACT-minus-PERMUTED estimate.",
        "",
        "This is not evidence that PACT fails to generalize. It is evidence that two of the three proposed shifts were not cleanly solvable by the privileged expert under the fixed scene and planner, making the proposed test inadequate. Per the preregistration, those shifts were not adjusted after their outcomes were seen.",
        "",
        "## Integrity and execution audit",
        "",
        f"- Manifest: `{manifest['manifest_sha256']}`; expert screen: `{expert['expert_screen_sha256']}`.",
        f"- All 48 expert rows reconciled; worker count stayed fixed at {expert['workers']}.",
        "- After 33 terminal rows, the multiprocessing launcher stalled while recycling workers. No worker remained and no pending row had accepted an initial observation. The launcher was terminated and the 15 untouched rows were resumed under identical frozen code; no terminal or boundary-crossed row was rerun.",
        "- Every shifted condition moves at least two axes outside the declared training support; this is asserted by the manifest contract and tests.",
        "- The original `PactCollisionCorridorSampler` body and `pact_collision_corridor.xml` matched their committed byte references when the manifest was frozen. Only additive subclasses were used.",
        "- All six ACT/PACT checkpoints, dataset statistics, and the frozen 32-D encoder were hash-verified even though no policy was run.",
        "- The contact-endpoint decision, analysis, final decision, endpoint, awarded token, and confirmatory41 were not modified.",
        "",
        "## Artifacts",
        "",
        "- `configs/pact_geometry_generalization_v1.json` — candidate conditions, training support, and all fixed expert/policy instances.",
        "- `diagnostics_output/pact_geometry_generalization/expert_screen.json` — reconciled Phase-0 gate.",
        "- `diagnostics_output/pact_geometry_generalization/{schedule,dispatch,analysis,final_decision,provenance}.json` — explicit stopped-experiment record.",
        "- `tests/test_pact_geometry_generalization.py` — geometry, balance, integrity, and decision-boundary tests.",
        "",
        TOKEN,
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expert-screen", required=True, type=Path)
    parser.add_argument("--policy-registry", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    expert = json.loads(args.expert_screen.read_text())
    expert_sha = validate_self_hash(expert, "expert_screen_sha256", "expert screen")
    if expert.get("reconciled") is not True or expert.get("continue_to_policy_evaluation") is not False:
        raise ValueError("expert screen did not reach the preregistered stop")
    if len(expert.get("surviving_shifted_condition_ids", [])) >= 2:
        raise ValueError("two shifted conditions survived; Phase-0 stop is invalid")
    rows = load_rows(manifest, args.output_root)
    summaries = {condition: condition_summary(condition, rows) for condition in ("C0", "C1", "C2", "C3")}
    observed_survivors = [condition for condition, item in summaries.items() if item["passed"]]
    if observed_survivors != expert["surviving_condition_ids"]:
        raise ValueError("independent raw-row gate disagrees with expert summary")
    registry = json.loads(args.policy_registry.read_text())
    registry_sha = validate_self_hash(registry, "policy_registry_sha256", "policy registry")
    models = verified_models(registry)
    for relative, expected in FROZEN_CONTACT_HASHES.items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"frozen contact endpoint changed: {relative}")

    schedule: dict[str, Any] = {
        "schema_version": "pact_geometry_generalization_schedule_stop_v1",
        "status": "not_frozen_phase0_hard_stop",
        "manifest_sha256": manifest["manifest_sha256"],
        "expert_screen_sha256": expert_sha,
        "planned_policy_rollouts": 900,
        "actual_policy_rollouts": 0,
        "surviving_condition_ids": expert["surviving_condition_ids"],
        "surviving_shifted_condition_ids": expert["surviving_shifted_condition_ids"],
        "minimum_shifted_conditions_required": 2,
        "rows": [],
        "policy_outcomes_observed": False,
        "reason": "fewer_than_two_shifted_conditions_passed_expert_screen",
    }
    schedule["schedule_sha256"] = sha256_payload(schedule)
    dispatch: dict[str, Any] = {
        "schema_version": "pact_geometry_generalization_dispatch_stop_v1",
        "status": "not_launched_phase0_hard_stop",
        "schedule_sha256": schedule["schedule_sha256"],
        "workers_planned": 8,
        "policy_subprocesses_launched": 0,
        "smoke_launched": False,
        "outcomes_seen": False,
        "reason": schedule["reason"],
    }
    dispatch["dispatch_contract_sha256"] = sha256_payload(dispatch)
    analysis: dict[str, Any] = {
        "schema_version": "pact_geometry_generalization_analysis_stop_v1",
        "results_available": False,
        "phase0_only": True,
        "manifest_sha256": manifest["manifest_sha256"],
        "expert_screen_sha256": expert_sha,
        "schedule_sha256": schedule["schedule_sha256"],
        "condition_results": summaries,
        "surviving_condition_ids": observed_survivors,
        "surviving_shifted_condition_ids": [item for item in observed_survivors if item != "C0"],
        "required_shifted_condition_count": 2,
        "observed_shifted_condition_count": 1,
        "C0_modality_gap_reproduced": None,
        "policy_contrasts": None,
        "decision": TOKEN,
        "reason": schedule["reason"],
        "execution_incident": {
            "terminal_rows_before_launcher_recycle_stall": 33,
            "active_workers_at_stall": 0,
            "pending_rows_with_initial_observation_accepted": 0,
            "terminal_rows_rerun": 0,
            "boundary_crossed_rows_rerun": 0,
            "remaining_untouched_rows_resumed": 15,
            "scientific_effect": "none",
        },
    }
    analysis["analysis_sha256"] = sha256_payload(analysis)
    final: dict[str, Any] = {
        "schema_version": "pact_geometry_generalization_decision_v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "expert_screen_sha256": expert_sha,
        "schedule_sha256": schedule["schedule_sha256"],
        "analysis_sha256": analysis["analysis_sha256"],
        "decision": TOKEN,
        "reason": schedule["reason"],
        "cannot_award_pact_confirmatory_token": True,
        "policy_rollouts": 0,
    }
    final["final_decision_sha256"] = sha256_payload(final)
    provenance: dict[str, Any] = {
        "schema_version": "pact_geometry_generalization_provenance_v1",
        "root_commit_at_finalize": git_head(ROOT),
        "act_commit_at_finalize": git_head(ROOT / "submodules/act"),
        "molmospaces_commit_at_finalize": git_head(ROOT / "submodules/molmospaces"),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_file_sha256": sha256_file(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "expert_screen_path": str(args.expert_screen.resolve()),
        "expert_screen_file_sha256": sha256_file(args.expert_screen),
        "expert_screen_sha256": expert_sha,
        "policy_registry_sha256": registry_sha,
        "verified_model_artifacts": models,
        "base_sampler_integrity": manifest["base_sampler_integrity"],
        "scene_xml_integrity": manifest["scene_xml_integrity"],
        "frozen_contact_endpoint_hashes": FROZEN_CONTACT_HASHES,
        "confirmatory41_read_or_modified": False,
        "policy_training_or_retraining_performed": False,
        "policy_evaluation_launched": False,
    }
    provenance["provenance_sha256"] = sha256_payload(provenance)
    destination = args.output_root
    write(destination / "schedule.json", schedule)
    write(destination / "dispatch.json", dispatch)
    write(destination / "analysis.json", analysis)
    write(destination / "final_decision.json", final)
    write(destination / "provenance.json", provenance)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        report_text(manifest=manifest, expert=expert, summaries=summaries, schedule=schedule)
    )
    print(TOKEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
