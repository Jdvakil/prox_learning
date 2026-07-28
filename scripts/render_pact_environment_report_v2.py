#!/usr/bin/env python3
"""Render the remediation-v2 Phase 1 gate without changing its decision."""

# ruff: noqa: ISC004

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOKENS = {
    "PACT_ENVIRONMENT_ADEQUATE",
    "PACT_ENVIRONMENT_INADEQUATE",
    "PACT_EXPERIMENT_INCOMPLETE",
}


def _interval(values: list[float] | None) -> str:
    if values is None:
        return "N/A"
    return f"[{values[0]:.1%}, {values[1]:.1%}]"


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def render(gate: dict, manifest: dict, preregistration: dict) -> str:
    token = gate["decision"]
    if token not in TOKENS:
        raise ValueError(f"invalid Phase 1 decision {token!r}")
    if gate.get("schema_version") != "pact_environment_gate_v2":
        raise ValueError("the remediation report requires a v2 gate")
    if gate["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("gate/manifest identity mismatch")

    expert = gate["expert"]
    surface = gate["surface_observability"]
    points = surface["point_estimates"]
    intervals = surface["intervals_95"]
    act = gate["act"]
    gate_b = gate["gate_b"]
    gate_c = gate["gate_c"]
    classifications = gate["science_gate_classifications"]
    lines = [
        "# PACT environment adequacy",
        "",
        "## Remediation-v2 decision",
        "",
        f"Decision: `{token}`.",
        "",
        "This is a new, independently seeded Phase 1 experiment. The historical "
        "v1 pilot is neither rescored nor pooled. Its preregistered "
        "`PACT_ENVIRONMENT_INADEQUATE` decision remains final under v1.",
        "",
        f"The v2 manifest SHA-256 is `{manifest['manifest_sha256']}` and its "
        f"master seed is `{manifest['master_seed']}`. The route remains "
        f"`{preregistration['route']}` and the physical scene remains "
        f"`{preregistration['environment_version']}`.",
        "",
        "## What changed—and what did not",
        "",
        "The corridor XML, panel geometry, aperture, target distribution, balanced "
        "intrusion sides, and target/side independence are unchanged. The expert "
        "clearance is 0.10 m, and mid-rollout trajectory replans are disabled. "
        "Initial construction retries remain pre-action and deterministic.",
        "",
        "Every fixed attempt remains in the ledger. A demonstration is usable only "
        "when the task succeeds with zero `hazard_bar` and zero "
        "`other_environment` contact. Target contact is exempt.",
        "",
        "## Expert demonstrations and infrastructure",
        "",
        "| Quantity | Result |",
        "|---|---:|",
        f"| Fixed expert attempts | {expert['attempts']} |",
        f"| Scientific outcomes | {expert['scientific_outcomes']} |",
        f"| Ordinary task success | {expert['ordinary_task_success']}/"
        f"{expert['attempts']} |",
        f"| Usable clean demonstrations | "
        f"{expert['usable_clean_demonstrations']}/{expert['attempts']} |",
        f"| Clean-demo Wilson 95% CI | "
        f"{_interval(expert['clean_demo_fraction_wilson_95'])} |",
        f"| Frozen usable-demo floor | {expert['usable_clean_demo_floor']} |",
        f"| No scientific outcome | {expert['no_scientific_outcome']}/"
        f"{expert['attempts']} |",
        f"| No-outcome Wilson 95% CI | "
        f"{_interval(expert['no_scientific_outcome_wilson_95'])} |",
        "",
        "Demonstrator count and infrastructure health are progression criteria, "
        "not science gates. They cannot produce "
        "`PACT_ENVIRONMENT_INADEQUATE`.",
        "",
        "## Surface observability",
        "",
        "| Quantity | Point estimate | 95% interval | Frozen minimum |",
        "|---|---:|---:|---:|",
        f"| Active scientific episodes | "
        f"{points['active_episode_fraction']:.1%} | "
        f"{_interval(intervals['active_episode_fraction_wilson'])} | 83.3% |",
        f"| Pre-grasp frames with panel inside 20 cm | "
        f"{points['inside_20cm']:.1%} | "
        f"{_interval(intervals['inside_20cm_episode_cluster_bootstrap'])} | 30.0% |",
        f"| Pre-grasp frames with panel inside 12 cm | "
        f"{points['inside_12cm']:.1%} | "
        f"{_interval(intervals['inside_12cm_episode_cluster_bootstrap'])} | 5.0% |",
        "",
        f"Robust classification: `{classifications['surface_observability']}`. "
        f"All leave-one-episode-out point estimates pass: "
        f"`{surface['all_leave_one_episode_out_points_pass']}`.",
        "",
    ]
    recovery = gate.get("infrastructure_recovery")
    if recovery is not None:
        prior = recovery["prior_failed_dispatch"]
        repaired = recovery["repaired_dispatch"]
        smoke = recovery["launch_smoke"]
        lines.extend(
            [
                "## Handoff-3 infrastructure recovery",
                "",
                f"The original dispatch made {prior['attempts']} evaluator "
                "attempts but accepted zero initial observations, executed zero "
                "actions, and produced zero scientific outcomes. All 64 failed "
                "before manifest load because the relative manifest path was "
                "invalid from the evaluator working directory.",
                "",
                "The absolute-path fix was committed before any policy result "
                "existed. Path resolution is content-independent: it cannot "
                "select rows by contact or task outcome and cannot change policy "
                "actions after startup. Re-executing these unchanged rows is "
                "therefore pre-observation infrastructure recovery, not "
                "outcome-based replacement.",
                "",
                f"Predeclared launch-smoke row {smoke['schedule_index']} "
                f"(`{smoke['rollout_id']}`) passed once before full dispatch. "
                f"The repaired ledger contains {repaired['scientific_results']} "
                "scientific results, "
                f"{repaired['post_boundary_failures']} post-boundary terminal "
                "failures, and "
                f"{repaired['pre_observation_infrastructure_failures']} "
                "retryable pre-observation failures. Scientific rows rerun: "
                f"{repaired['scientific_rows_rerun']}.",
                "",
                "The expert and surface measurements above are carried forward "
                "byte-for-byte from the settled Phase 1 gate; they were not "
                "recomputed for this resume.",
                "",
            ]
        )
    lines.extend(
        [
        "## Gate B — vision alone is insufficient but solvable",
        "",
        f"ACT collision-free task success is "
        f"{act['collision_free_task_success']}/{act['scientific_outcomes']} "
        f"({_percent(gate_b['point_estimate'])}), Wilson 95% CI "
        f"{_interval(gate_b['wilson_95'])}. The target point band is "
        "[33.3%, 66.7%], with the interval required inside [20%, 80%].",
        "",
        f"Robust classification: `{classifications['gate_b']}`; one-outcome "
        f"stable: `{gate_b['one_outcome_stable']}`.",
        "",
        f"Ordinary ACT task success is {act['ordinary_task_success']}/"
        f"{act['scientific_outcomes']} "
        f"({_percent(act['ordinary_task_success_rate'])}), Wilson 95% CI "
        f"{_interval(act['ordinary_task_success_wilson_95'])}. This is the "
        "secondary endpoint.",
        "",
        "## Gate C — the baseline contacts the intrusion",
        "",
        f"ACT contacted the panel in {act['episodes_with_hazard_bar_contact']}/"
        f"{act['scientific_outcomes']} scientific rows "
        f"({_percent(gate_c['point_estimate'])}), Wilson 95% CI "
        f"{_interval(gate_c['wilson_95'])}. The frozen point minimum is 25%, "
        "with Wilson lower bound above 10%.",
        "",
        f"Robust classification: `{classifications['gate_c']}`; one-outcome "
        f"stable: `{gate_c['one_outcome_stable']}`. "
        f"`other_environment` contact occurred in "
        f"{act['episodes_with_other_environment_contact']} rows and does not "
        "substitute for panel contact.",
        "",
        "## Decision",
        "",
        ]
    )
    if token == "PACT_ENVIRONMENT_ADEQUATE":
        lines.append(
            "All three environment science gates robustly pass, and the "
            "predeclared demo/infrastructure progression requirements are met. "
            "The experiment may proceed to full collection and training."
        )
    elif token == "PACT_ENVIRONMENT_INADEQUATE":
        lines.append(
            "At least one environment science property robustly fails. The "
            "experiment stops before full collection or PACT training."
        )
    else:
        lines.append(
            "No environment-inadequacy claim is made. At least one result is "
            "marginal or a non-science progression/reconciliation requirement "
            "is unmet."
        )
    lines.extend(
        [
            "",
            "The last line is the exact allowed decision token.",
            "",
            token,
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    document = render(
        json.loads(args.gate.read_text()),
        json.loads(args.manifest.read_text()),
        json.loads(args.preregistration.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document)
    print(document.rstrip().splitlines()[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
