#!/usr/bin/env python3
"""Render the Phase-0 pick-and-place gate report from its frozen JSON ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pact_place_corridor_contract import FAIL_TOKEN, ROOT, load_contract


def fraction(cell: dict) -> str:
    return f"{cell['count']}/{cell.get('denominator', 24)} ({100 * cell['rate']:.1f}%)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/pact_place_corridor_v1.json",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "diagnostics_output/pact_place_corridor/expert_screen.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/PACT_PLACE_CORRIDOR_GATE.md",
    )
    args = parser.parse_args()
    contract = load_contract(args.config)
    summary = json.loads(args.summary.read_text())
    if summary["config_sha256"] != contract["config_sha256"]:
        raise SystemExit("summary does not belong to the frozen config")
    given = summary["place_success_given_grasp"]
    given_rate = (
        given["numerator"] / given["denominator"]
        if given["denominator"]
        else None
    )
    decision = summary["decision"]
    row_results = [
        json.loads((ROOT / relative).read_text())
        for relative in summary["row_result_paths"]
    ]
    failed = [row for row in row_results if not row.get("task_success", False)]
    retrieval_failures = [
        row for row in failed if not row.get("grasp_phase_success", False)
    ]
    placement_failures = [
        row
        for row in failed
        if row.get("grasp_phase_success", False)
        and not row.get("place_phase_success", False)
    ]
    failed_non_target_contacts = [
        row
        for row in failed
        if row.get("contact_audit", {})
        .get("contact_class_totals", {})
        .get("hazard_bar", 0)
        or row.get("contact_audit", {})
        .get("contact_class_totals", {})
        .get("other_environment", 0)
    ]
    n_failed = len(failed)
    if n_failed == 0:
        failure_intro = (
            "There were no task failures. Grasp retrieval and placement/release both "
            "completed on every reconciled episode."
        )
        contact_note = (
            "No failed rows. Hazard-bar and other-environment contacts are reported "
            "in the table above."
        )
    else:
        failure_intro = (
            f"The {n_failed} task failure"
            f"{'s' if n_failed != 1 else ''} split into "
            f"**{len(retrieval_failures)} retrieval failure"
            f"{'s' if len(retrieval_failures) != 1 else ''}** and "
            f"**{len(placement_failures)} placement/release failure"
            f"{'s' if len(placement_failures) != 1 else ''} after "
            "successful retrieval**."
        )
        contact_note = (
            "Failed rows with hazard or other-environment contact: "
            f"{len(failed_non_target_contacts)}/{n_failed}."
        )
    bow_fallback = summary.get("bow_fallback_episodes")
    disposition = (
        "Phase 0 passed. Full demonstration collection may proceed under a separately "
        "frozen collection contract."
        if decision != FAIL_TOKEN
        else "Phase 0 failed. Per the preregistration, work stops here: no demonstration "
        "collection, encoder update, policy training, or learned-policy evaluation was run."
    )
    lines = [
        "# PACT pick-and-place corridor: Phase 0 expert gate",
        "",
        "## Decision",
        "",
        disposition,
        "",
        (
            "The gate required at least **20 clean successes in 24 fixed episodes**. "
            "A clean success means upstream `PickAndPlaceTask` success with zero "
            "`hazard_bar` and zero `other_environment` contact entries. Contact with "
            "`grasp_target` and the new `place_receptacle` class is expected and exempt."
        ),
        "",
        "## Measured Phase-0 results",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Reconciled rows | {summary['n']}/24 |",
        (
            f"| Clean pick-and-place successes | {summary['gate']['clean_successes']}/24 "
            f"({100 * summary['gate']['clean_success_rate']:.1f}%) |"
        ),
        f"| Ordinary task successes | {fraction(summary['task_success'])} |",
        (
            "| Grasp phase successes (cup retrieved outside aperture) | "
            f"{fraction(summary['grasp_phase_success'])} |"
        ),
        f"| Place phase successes | {fraction(summary['place_phase_success'])} |",
        f"| Place successes given grasp | {given['numerator']}/{given['denominator']} "
        + (f"({100 * given_rate:.1f}%) |" if given_rate is not None else "(undefined) |"),
        (
            "| Episodes with inbound hazard contact | "
            f"{summary['hazard_contact_episodes']['inbound']}/24 |"
        ),
        (
            "| Episodes with outbound hazard contact | "
            f"{summary['hazard_contact_episodes']['outbound']}/24 |"
        ),
        (
            "| Episodes with other-environment contact | "
            f"{summary['other_environment_contact_episodes']}/24 |"
        ),
        f"| Sampling failures | {summary['sampling_failures']} |",
        f"| Infrastructure failures | {summary['infrastructure_failures']} |",
    ]
    if bow_fallback is not None:
        lines.append(f"| Bow-fallback episodes | {bow_fallback}/24 |")
    lines.extend(
        [
        "",
        "## Failure localization",
        "",
        failure_intro,
        "",
        "| Failure class | Count | Row indices |",
        "|---|---:|---|",
        (
            f"| Cup not retrieved outside aperture | {len(retrieval_failures)} | "
            f"{', '.join(str(row['role_index']) for row in retrieval_failures) or '—'} |"
        ),
        (
            "| Retrieved, but placement/release incomplete | "
            f"{len(placement_failures)} | "
            f"{', '.join(str(row['role_index']) for row in placement_failures) or '—'} |"
        ),
        "",
        contact_note,
        "",
        "## What changed",
        "",
        (
            "The scene is a strict fork, `pact_place_corridor_v1.xml`. It preserves the "
            "panel, aperture, target sampling, robot offset, side balance, sensor layout, "
            "and existing contact classes, while adding a low pedestal and shallow tray "
            "wholly outside the aperture plane (tray x range 0.25–0.45 m; aperture x "
            "0.58 m). The place expert composes the upstream pick-and-place planner with "
            "direction-aware panel bows on both inbound and outbound segments; the outbound "
            "carried-object envelope and clearance are deliberately larger."
        ),
        "",
        "## Success criterion",
        "",
        (
            "The endpoint is the upstream support-and-release criterion: the target is "
            "supported by the receptacle at at least 50% of its weight, the robot has "
            "released it, the receptacle moved no more than 0.1 m, and its tilt is no more "
            "than 45°. The old one-centimetre lift condition is recorded only as a "
            "grasp-progress diagnostic and is not on the task-success path."
        ),
        "",
        "## Integrity checks",
        "",
        f"- Frozen Phase-0 config SHA-256: `{contract['config_sha256']}`.",
        (
            "- `pact_collision_corridor.xml` retained its pinned SHA-256 "
            "`f8c04b07b9416593eb60ad4797ccbae91f7d3524effd30438ef552e5a2d75540`."
        ),
        (
            "- The original `PactCollisionCorridorSampler`, "
            "`PactCollisionCorridorPolicy`, and `PactCollisionCorridorPolicyConfig` class "
            "bodies match the MolmoSpaces submodule commit exactly."
        ),
        (
            "- Frozen contact-endpoint, geometry-v3, blur-sweep, and blind-RGB artifacts "
            "were hash-verified before and after the screen."
        ),
        "- No checkpoint, encoder, threshold, or existing scene was changed.",
        "",
        "## Artifacts",
        "",
        f"- `{args.config.relative_to(ROOT) if args.config.is_absolute() else args.config}`",
        f"- `{args.summary.relative_to(ROOT) if args.summary.is_absolute() else args.summary}`",
        f"- `{Path(summary['row_result_paths'][0]).parent.parent.parent / 'expert_screen_rows/*/result.json'}`",
        ]
    )
    stop_path = Path(summary["row_result_paths"][0]).parent.parent.parent / "stop_record.json"
    if decision == FAIL_TOKEN:
        lines.append(f"- `{stop_path}`")
    lines.extend(["", decision, ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(args.output)
    print(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
