#!/usr/bin/env python3
"""Immutable V10.3 Step-0B stop record for the owner-approved early stop.

This is NOT an exhaustive Step-0B completion. Nine of twelve cell/direction
cases were evaluated; the three right-inbound cases were stopped before they
returned and are recorded as ``not_evaluated``. The stop is nonetheless globally
conclusive, because the endpoint certificate shows that no registered height
admits the pinned inbound endpoint on any left-inbound cell, and the plan
requires a geometry to route all six cells in both directions.

Per-case rejection tables and node/edge arrays for the nine completed cases were
held in worker memory and returned only on completion, so they did not survive
the early stop. What is preserved is the workers' own stdout, kept verbatim
alongside this record. This file does not restate counts it cannot evidence.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_corridor_contract import sha256_file  # noqa: E402
from pact_place_v10_runtime import establish_v10_runtime_env, write_immutable  # noqa: E402
from pact_place_v103_contract import (  # noqa: E402
    CONTRACT_VERSION,
    SEARCH_ROOT_RELATIVE,
    all_cell_keys,
    empty_authorization,
    implementation_hashes,
    implementation_sha256,
    search_lattice,
    verify_protected_artifacts,
)
from pact_place_v103_geometry import HEIGHT_LATTICE_M  # noqa: E402

SEARCH_ROOT = ROOT / SEARCH_ROOT_RELATIVE
LOG_NAME = "search_worker_stdout.log"
CERT_NAME = "endpoint_certificate.json"
LINE = re.compile(
    r"^(?P<cell>\S+) (?P<direction>inbound|outbound): "
    r"layers_ok=(?P<ok>\d+)/(?P<total>\d+) "
    r"feasible=(?P<feasible>\{[^}]*\}) ik=(?P<ik>\d+) \((?P<elapsed>[\d.]+)s\)$"
)
RIGHT_INBOUND = (
    "F0_target_side_stagger:right",
    "F1_inner_panel_stagger:right",
    "F2_outer_panel_stagger:right",
)


def parse_completed(log_path: Path) -> list[dict[str, Any]]:
    out = []
    for line in log_path.read_text().splitlines():
        match = LINE.match(line.strip())
        if not match:
            continue
        out.append(
            {
                "cell_key": match["cell"],
                "direction": match["direction"],
                "status": "evaluated",
                "templates_with_complete_layers": int(match["ok"]),
                "templates_evaluated": int(match["total"]),
                "feasible_routes_by_height": json.loads(
                    match["feasible"].replace("'", '"')
                ),
                "ik_calls": int(match["ik"]),
                "elapsed_s": float(match["elapsed"]),
            }
        )
    return sorted(out, key=lambda item: (item["cell_key"], item["direction"]))


def build(log_path: Path, cert_path: Path) -> dict[str, Any]:
    certificate = json.loads(cert_path.read_text())
    completed = parse_completed(log_path)
    evaluated_keys = {(c["cell_key"], c["direction"]) for c in completed}
    cases: list[dict[str, Any]] = list(completed)
    for cell in RIGHT_INBOUND:
        if (cell, "inbound") in evaluated_keys:
            continue
        cases.append(
            {
                "cell_key": cell,
                "direction": "inbound",
                "status": "not_evaluated",
                "reason": "owner_approved_early_stop_before_worker_returned",
                "templates_evaluated": 0,
                "templates_with_complete_layers": None,
                "feasible_routes_by_height": None,
                "ik_calls": None,
                "elapsed_s": None,
            }
        )
    cases.sort(key=lambda item: (item["cell_key"], item["direction"]))

    zero_feasible = [
        case
        for case in completed
        if not any((case["feasible_routes_by_height"] or {}).values())
    ]
    left_inbound_blocked = not certificate["any_height_admits_any_left_inbound_cell"]

    return {
        "schema_version": "pact_place_v103_ik_search_stop_v1",
        "contract_version": CONTRACT_VERSION,
        "role": "owner_approved_globally_conclusive_early_stop",
        "stop_reason": "no_static_geometry_with_twelve_joint_routes",
        "search_exhaustive": False,
        "every_registered_template_evaluated": False,
        "cases_total": 12,
        "cases_completed": len(completed),
        "cases_not_evaluated": [
            f"{case['cell_key']}|{case['direction']}"
            for case in cases
            if case["status"] == "not_evaluated"
        ],
        "global_conclusion_conclusive": True,
        "conclusive_witness": "pinned_endpoint_clearance_below_node_floor",
        "remaining_cases_cannot_change_selection": True,
        "conclusive_argument": (
            "The plan admits a geometry only if a complete joint route exists for "
            "all six cells in both directions. The endpoint certificate shows the "
            "pinned inbound endpoint on all three left-inbound cells violates the "
            "0.020 m node clearance floor against the negative lobe at every "
            "registered height, at both the pinned pregrasp frame and the earlier "
            "retained control frame, penetrating on both instruments. That endpoint "
            "is pinned to the retained qpos and is the first interpolation sample of "
            "every edge leaving it, so no route template, lane, staging buffer, "
            "pass-z offset or pass orientation can rescue those cells. Three "
            "left-inbound cells therefore have zero admissible heights, so no "
            "height can cover twelve cases, whatever the three unevaluated "
            "right-inbound cases would have returned."
        ),
        "height_lattice_m": list(HEIGHT_LATTICE_M),
        "search_lattice": search_lattice(),
        "cases": cases,
        "completed_cases_with_zero_feasible_routes": len(zero_feasible),
        "completed_cases_with_any_feasible_route": len(completed) - len(zero_feasible),
        "left_inbound_blocked_at_every_height": bool(left_inbound_blocked),
        "endpoint_certificate": {
            "path": str(cert_path.relative_to(ROOT)),
            "artifact_sha256": certificate["artifact_sha256"],
            "node_floor_m": certificate["node_floor_m"],
            "any_height_admits_any_left_inbound_cell": certificate[
                "any_height_admits_any_left_inbound_cell"
            ],
        },
        "preserved_worker_stdout": {
            "path": str(log_path.relative_to(ROOT)),
            "sha256": sha256_file(log_path),
            "note": (
                "verbatim worker stdout for the nine completed cases; the only "
                "surviving record of their per-case counts"
            ),
        },
        "registered_artifacts_not_written": {
            "nodes.npz": "worker node tables were returned only on case completion "
            "and did not survive the early stop",
            "edges.npz": "worker edge tables were returned only on case completion "
            "and did not survive the early stop",
            "selected_routes.npz": "no geometry was selected, so there are no "
            "selected routes to serialize",
        },
        "selected_geometry": None,
        "selected_routes": None,
        "search_passed": False,
        "authorizes_step_0c": False,
        "authorizes_smoke": False,
        "authorizes_review": False,
        "authorizes_phase0": False,
        "episodes_generated": 0,
        "env_step_called": False,
        "runtime_built": False,
        "protected_artifacts": verify_protected_artifacts(),
        "implementation_sha256": implementation_sha256(),
        "implementation_files": implementation_hashes(),
        "registered_cell_keys": list(all_cell_keys()),
        **empty_authorization(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", type=Path, default=SEARCH_ROOT)
    args = parser.parse_args()
    establish_v10_runtime_env()
    root = args.search_root.resolve()
    document = build(root / LOG_NAME, root / CERT_NAME)
    digest = write_immutable(root / "search.json", document)
    print(
        json.dumps(
            {
                "stop_reason": document["stop_reason"],
                "search_exhaustive": document["search_exhaustive"],
                "cases_completed": document["cases_completed"],
                "cases_not_evaluated": document["cases_not_evaluated"],
                "global_conclusion_conclusive": document["global_conclusion_conclusive"],
                "conclusive_witness": document["conclusive_witness"],
                "remaining_cases_cannot_change_selection": document[
                    "remaining_cases_cannot_change_selection"
                ],
                "artifact_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
