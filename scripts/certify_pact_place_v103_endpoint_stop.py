#!/usr/bin/env python3
"""V10.3 endpoint certificate for the owner-approved early stop.

Step 0B was stopped before it evaluated every registered route template. This
certificate records, cheaply and immutably, the single fact that makes the
remaining work unable to change the outcome: on all three left-inbound cells the
pinned inbound endpoint already violates the node clearance floor against the
negative lobe, at *every* registered height.

Because that endpoint is pinned to the retained qpos, and because every edge
leaving it has that state as its own first interpolation sample, no route
template and no lane/staging/pass parameter can rescue those cells.

Two independent instruments are recorded per component:

* analytic exact GJK against the posed V10.3 box (the search's own node
  predicate). Unsigned: 0.0 means the shapes intersect.
* hardened signed ``mj_geomDistance`` with the V10.3 component poses written
  into the V10 mocap geom slots as an offline probe only. Negative means
  penetration, and it names the contacting robot geom.

Offline diagnostic. Compiles nothing new, calls no ``env.step``, renders no
observation, generates no episode, and authorizes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_geom_distance import true_distance  # noqa: E402
from pact_place_corridor_contract import PLACE_V5_SCENE_SHA256, sha256_file  # noqa: E402
from pact_place_v10_compound_pendant_contract import (  # noqa: E402
    ALL_GEOMS,
    PENDANT_BODY,
    PLACE_V10_SCENE_SHA256,
    SCENE_XML_RELATIVE,
    V5_SCENE_XML_RELATIVE,
)
from pact_place_v10_exact import verify_v99_inputs  # noqa: E402
from pact_place_v10_runtime import establish_v10_runtime_env, write_immutable  # noqa: E402
from pact_place_v103_contract import (  # noqa: E402
    CONTRACT_VERSION,
    empty_authorization,
    implementation_hashes,
    implementation_sha256,
    verify_protected_artifacts,
)
from pact_place_v103_geometry import (  # noqa: E402
    HEIGHT_LATTICE_M,
    enumerate_v103_assemblies,
    scene_xml_text,
)
from pact_place_v103_joint_route import NODE_MIN_CLEARANCE_M  # noqa: E402
from pact_place_v99_exact import snapshot_jobs_from_reconstruction  # noqa: E402
from pact_place_v99_geometry import window_masks  # noqa: E402
from pact_place_v9_contract import sha256_payload  # noqa: E402
from search_pact_place_v103_joint_route import CellContext, pendant_boxes  # noqa: E402

DEFAULT_OUTPUT = (
    ROOT / "diagnostics_output" / "pact_place_v103_ik_search" / "endpoint_certificate.json"
)
LEFT_INBOUND_ROLES = (600, 602, 604)
# V10 mocap slots reused offline, purely as a signed-distance probe. Nothing in
# V10.3 poses geoms at episode runtime; see the V10.2 broad-phase root cause.
PROBE_SLOTS = {
    "lobe_0": ALL_GEOMS[0],
    "stem_0": ALL_GEOMS[3],
    "lobe_1": ALL_GEOMS[1],
    "stem_1": ALL_GEOMS[4],
    "crossbar": ALL_GEOMS[6],
}


def _scene_state_digest(model, data) -> str:
    payload = {
        "geom_pos": [
            [float(v) for v in model.geom_pos[int(model.geom(n).id)]] for n in ALL_GEOMS
        ],
        "geom_size": [
            [float(v) for v in model.geom_size[int(model.geom(n).id)]] for n in ALL_GEOMS
        ],
        "geom_contype": [int(model.geom_contype[int(model.geom(n).id)]) for n in ALL_GEOMS],
        "geom_conaffinity": [
            int(model.geom_conaffinity[int(model.geom(n).id)]) for n in ALL_GEOMS
        ],
        "mocap_pos": [[float(v) for v in row] for row in np.asarray(data.mocap_pos)],
    }
    return sha256_payload(payload)


def _signed_probe(context: CellContext, assembly: dict[str, Any]) -> dict[str, Any]:
    """Signed distance and nearest robot geom per component, offline probe."""
    import mujoco

    model, data = context.model, context.data
    components = {item["name"]: item for item in assembly["components"]}
    saved = {
        geom: (
            np.asarray(model.geom_pos[int(model.geom(geom).id)], dtype=float).copy(),
            np.asarray(model.geom_size[int(model.geom(geom).id)], dtype=float).copy(),
            int(model.geom_contype[int(model.geom(geom).id)]),
            int(model.geom_conaffinity[int(model.geom(geom).id)]),
            int(model.geom_sameframe[int(model.geom(geom).id)]),
        )
        for geom in PROBE_SLOTS.values()
    }
    saved_mocap = np.asarray(data.mocap_pos, dtype=float).copy()
    try:
        data.mocap_pos[int(model.body_mocapid[int(model.body(PENDANT_BODY).id)])] = (
            np.zeros(3)
        )
        for name, geom in PROBE_SLOTS.items():
            gid = int(model.geom(geom).id)
            model.geom_sameframe[gid] = 0
            model.geom_pos[gid] = np.asarray(components[name]["center_m"], dtype=float)
            model.geom_size[gid] = np.asarray(components[name]["half_m"], dtype=float)
            model.geom_contype[gid] = 8
            model.geom_conaffinity[gid] = 15
        mujoco.mj_forward(model, data)
        out: dict[str, Any] = {}
        for name, geom in PROBE_SLOTS.items():
            gid = int(model.geom(geom).id)
            best = (float("inf"), None, None)
            for robot_gid in context.robot_geom_ids:
                distance = float(true_distance(model, data, [robot_gid], [gid]))
                if distance < best[0]:
                    geom_name = model.geom(int(robot_gid)).name or ""
                    body = str(
                        model.body(int(model.geom_bodyid[int(robot_gid)])).name or ""
                    )
                    best = (distance, geom_name or f"geom_{robot_gid}", body)
            out[name] = {
                "component_geom_probe_slot": geom,
                "signed_distance_m": float(best[0]),
                "penetrating": bool(best[0] < 0.0),
                "penetration_depth_m": float(max(0.0, -best[0])),
                "nearest_robot_geom": best[1],
                "nearest_robot_body": best[2],
            }
        return out
    finally:
        for geom, (pos, size, contype, conaff, sameframe) in saved.items():
            gid = int(model.geom(geom).id)
            model.geom_pos[gid] = pos
            model.geom_size[gid] = size
            model.geom_contype[gid] = contype
            model.geom_conaffinity[gid] = conaff
            model.geom_sameframe[gid] = sameframe
        data.mocap_pos[:] = saved_mocap
        mujoco.mj_forward(model, data)


def certify() -> dict[str, Any]:
    protected = verify_protected_artifacts()
    reconstruction, _snapshot, cells = verify_v99_inputs()
    cells = {int(c["role_index"]): c for c in cells}
    jobs = {
        int(json.loads((Path(job["row_dir"]) / "result.json").read_text())["role_index"]): job
        for job in snapshot_jobs_from_reconstruction(reconstruction)
    }
    assemblies = {
        f"{float(a['lowest_lobe_bottom_z_m']):.3f}": a for a in enumerate_v103_assemblies()
    }
    boxes = pendant_boxes(list(assemblies.values()))
    geometry_hashes = {
        key: {
            "lowest_lobe_bottom_z_m": float(assembly["lowest_lobe_bottom_z_m"]),
            "assembly_id": assembly["assembly_id"],
            "assembly_sha256": sha256_payload(assembly),
            "serialized_scene_sha256": hashlib.sha256(
                scene_xml_text(assembly).encode()
            ).hexdigest(),
            "scene_file_written": False,
        }
        for key, assembly in assemblies.items()
    }

    cell_records: list[dict[str, Any]] = []
    for role in LEFT_INBOUND_ROLES:
        cell = cells[role]
        context = CellContext(jobs[role], cell, "inbound")
        try:
            phases = [str(s.get("policy_phase") or "") for s in context.steps]
            windows = window_masks(phases)
            inbound = np.flatnonzero(np.asarray(cell["inbound_mask"], dtype=bool))
            frames = {
                "pinned_endpoint_pregrasp": int(windows["pregrasp_index"]),
                "control_frame_last_inbound": int(inbound[-1]),
            }
            frame_records = []
            for label, step_index in frames.items():
                context.apply_recorded_step(step_index)
                full_qpos = np.asarray(
                    context.steps[step_index]["qpos"], dtype=float
                )
                arm = context.arm_qpos()
                context.set_arm(arm, carry=False)
                state_before = _scene_state_digest(context.model, context.data)
                analytic = context.pendant_clearances(boxes, include_target=False)
                per_height = {}
                for key in sorted(assemblies):
                    signed = _signed_probe(context, assemblies[key])
                    components = analytic[key]["per_component_m"]
                    minimum = float(min(components.values()))
                    per_height[key] = {
                        "lowest_lobe_bottom_z_m": float(
                            assemblies[key]["lowest_lobe_bottom_z_m"]
                        ),
                        "analytic_gjk_clearance_m": {
                            name: float(value) for name, value in components.items()
                        },
                        "signed_mj_geom_distance": signed,
                        "min_clearance_m": minimum,
                        "node_floor_m": float(NODE_MIN_CLEARANCE_M),
                        "node_clearance_margin_m": float(
                            minimum - NODE_MIN_CLEARANCE_M
                        ),
                        "meets_node_floor": bool(minimum >= NODE_MIN_CLEARANCE_M),
                        "binding_component": min(
                            components, key=lambda n: float(components[n])
                        ),
                    }
                state_after = _scene_state_digest(context.model, context.data)
                frame_records.append(
                    {
                        "frame": label,
                        "step_index": step_index,
                        "policy_phase": phases[step_index],
                        "retained_full_qpos_sha256": hashlib.sha256(
                            np.round(full_qpos, 12).tobytes()
                        ).hexdigest(),
                        "retained_arm_qpos_rad": [float(v) for v in arm],
                        "retained_arm_qpos_sha256": hashlib.sha256(
                            np.round(arm, 12).tobytes()
                        ).hexdigest(),
                        "probe_scene_state_sha256_before": state_before,
                        "probe_scene_state_sha256_after": state_after,
                        "probe_scene_state_restored": bool(
                            state_before == state_after
                        ),
                        "per_height": per_height,
                        "meets_node_floor_any_height": any(
                            item["meets_node_floor"] for item in per_height.values()
                        ),
                    }
                )
            cell_records.append(
                {
                    "role_index": role,
                    "cell_key": f"{cell['family']}:{cell['intrusion_side']}",
                    "family": str(cell["family"]),
                    "intrusion_side": str(cell["intrusion_side"]),
                    "direction": "inbound",
                    "episode_id": str(cell.get("episode_id") or ""),
                    "frames": frame_records,
                    "admits_any_height": any(
                        item["meets_node_floor_any_height"] for item in frame_records
                    ),
                }
            )
        finally:
            context.close()

    heights_admitting = {
        key: [
            record["cell_key"]
            for record in cell_records
            if record["frames"][0]["per_height"][key]["meets_node_floor"]
        ]
        for key in sorted(assemblies)
    }
    return {
        "schema_version": "pact_place_v103_endpoint_certificate_v1",
        "contract_version": CONTRACT_VERSION,
        "role": "conclusive_witness_for_owner_approved_early_stop",
        "claim": (
            "On every registered height, all three left-inbound cells violate the "
            "0.020 m node clearance floor against the negative lobe at the pinned "
            "inbound endpoint, and at the earlier retained control frame. The "
            "endpoint is pinned to the retained qpos and is the first "
            "interpolation sample of every edge leaving it, so no route template "
            "can rescue these cells at any registered height."
        ),
        "node_floor_m": float(NODE_MIN_CLEARANCE_M),
        "height_lattice_m": list(HEIGHT_LATTICE_M),
        "geometry_hashes": geometry_hashes,
        "heights_with_any_admitted_left_inbound_cell": heights_admitting,
        "any_height_admits_any_left_inbound_cell": any(
            heights_admitting[key] for key in heights_admitting
        ),
        "cells": cell_records,
        "instruments": {
            "analytic": "pact_geom_distance.gjk_distance vs posed V10.3 boxes; "
            "unsigned, 0.0 means intersecting",
            "signed": "pact_geom_distance.true_distance (hardened mj_geomDistance) "
            "with V10.3 component poses written into the V10 mocap probe slots "
            "offline; negative means penetration",
            "offline_probe_only": True,
            "runtime_geom_posing_used": False,
        },
        "scene_inputs": {
            "probe_scene_xml": SCENE_XML_RELATIVE,
            "probe_scene_sha256": sha256_file(ROOT / SCENE_XML_RELATIVE),
            "probe_scene_expected_sha256": PLACE_V10_SCENE_SHA256,
            "v5_scene_xml": V5_SCENE_XML_RELATIVE,
            "v5_scene_sha256": sha256_file(ROOT / V5_SCENE_XML_RELATIVE),
            "v5_scene_expected_sha256": PLACE_V5_SCENE_SHA256,
        },
        "protected_artifacts": protected,
        "implementation_sha256": implementation_sha256(),
        "implementation_files": implementation_hashes(),
        "calls_env_step": False,
        "renders_observations": False,
        "generates_episodes": False,
        **empty_authorization(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    establish_v10_runtime_env()
    document = certify()
    digest = write_immutable(args.output.resolve(), document)
    print(
        json.dumps(
            {
                "any_height_admits_any_left_inbound_cell": document[
                    "any_height_admits_any_left_inbound_cell"
                ],
                "n_cells": len(document["cells"]),
                "artifact_sha256": digest,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
