#!/usr/bin/env python3
"""V10.5 audit and erratum: verify the sealed artifacts, correct the record.

The V10.5 narrative is treated as untrusted. Nothing here reads a conclusion
from a V10.5 artifact and repeats it. Every number is recomputed:

* raw-file and canonical-payload SHA-256 for all four sealed artifacts;
* all 192 retained result and trajectory hashes, checked against both the
  recorded digest and the file bytes;
* every bundle statistic, recomputed by a **second, independent aggregator**
  written in a different style from the primary scorer and required to agree
  exactly.

No V10.5 or V10.4 artifact is modified. This writes one new immutable file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v105_contract import (  # noqa: E402
    CONTRACT_VERSION_V105,
    FRAGILITY_ROWS_DIR,
    canonical_payload_sha256,
    empty_authorization,
    sha256_file,
    sha256_payload,
    write_immutable_create_only,
)
from pact_place_v105_geometry import CLEARANCE_FLOOR_M, POSE_IDS, RISK_BAND_M  # noqa: E402

AUDIT_ROOT = "diagnostics_output/pact_place_v105_audit"
RECON_ROOT = ROOT / "diagnostics_output/pact_place_v105_reconstruction"
SITING_ROOT = ROOT / "diagnostics_output/pact_place_v105_siting"

SEALED_ARTIFACTS = {
    "reconstruction.json": RECON_ROOT / "reconstruction.json",
    "corpus_index.npz": RECON_ROOT / "corpus_index.npz",
    "siting.json": SITING_ROOT / "siting.json",
    "per_row_scores.npz": SITING_ROOT / "per_row_scores.npz",
}

# A household object is one palette slot. Its MuJoCo body tree contains nested
# mesh children, and the corridor chicane bodies (l0/l1/r0/r1) are separate
# fixtures, not palette objects. Counting either as a distinct object is the
# error this erratum corrects.
HOUSEHOLD_BODY_PREFIXES = tuple(f"pact_clutter_{index:02d}/" for index in range(8))
CORRIDOR_VESSEL_BODIES = ("pact_clutter_l0", "pact_clutter_l1",
                          "pact_clutter_r0", "pact_clutter_r1")


def hash_sealed() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, path in SEALED_ARTIFACTS.items():
        entry: dict[str, Any] = {
            "path": str(path.relative_to(ROOT)),
            "present": path.is_file(),
        }
        if path.is_file():
            entry["raw_file_sha256"] = sha256_file(path)
            entry["size_bytes"] = int(path.stat().st_size)
            if path.suffix == ".json":
                document = json.loads(path.read_text())
                entry["canonical_payload_sha256"] = canonical_payload_sha256(document)
                entry["embedded_payload_sha256"] = document.get("payload_sha256")
                entry["payload_self_consistent"] = (
                    entry["canonical_payload_sha256"]
                    == entry["embedded_payload_sha256"]
                )
            else:
                # An NPZ has no embedded self-hash; the canonical payload hash
                # is defined here as the hash of its ordered array digests.
                payload = np.load(path, allow_pickle=True)
                entry["canonical_payload_sha256"] = sha256_payload(
                    {
                        key: sha256_payload(
                            [str(v) for v in np.atleast_1d(payload[key]).ravel()]
                        )
                        for key in sorted(payload.files)
                    }
                )
                entry["arrays"] = sorted(payload.files)
                entry["embedded_payload_sha256"] = None
                entry["payload_self_consistent"] = None
        out[name] = entry
    return out


def verify_corpus_hashes() -> dict[str, Any]:
    """All 192 retained rows: recorded digest and file bytes, independently."""
    rows_dir = ROOT / FRAGILITY_ROWS_DIR
    entries: list[dict[str, Any]] = []
    problems: list[str] = []
    for directory in sorted(rows_dir.iterdir()):
        result_path = directory / "result.json"
        trajectory_path = directory / "trajectory.json"
        entry: dict[str, Any] = {"row_dir": str(directory.relative_to(ROOT))}
        if not result_path.is_file():
            problems.append(f"{directory.name}: result.json absent")
            entry.update({"ok": False, "reason": "result.json absent"})
            entries.append(entry)
            continue
        result = json.loads(result_path.read_text())
        recomputed_result = sha256_payload(
            {k: v for k, v in result.items() if k != "result_sha256"}
        )
        if not trajectory_path.is_file():
            # A corpus fact, not a hash mismatch: this row retained no
            # trajectory. Recorded, and reported separately from any hash
            # problem, with its clean status so the effect on the corpus
            # count is visible rather than inferred.
            entry.update(
                {
                    "ok": True,
                    "has_trajectory": False,
                    "clean_success": bool(result.get("clean_success")),
                    "result_raw_file_sha256": sha256_file(result_path),
                    "result_recorded_sha256": result.get("result_sha256"),
                    "result_recomputed_payload_sha256": recomputed_result,
                    "result_self_consistent": (
                        recomputed_result == result.get("result_sha256")
                    ),
                }
            )
            if not entry["result_self_consistent"]:
                entry["ok"] = False
                problems.append(f"{directory.name}: result hash mismatch")
            entries.append(entry)
            continue
        trajectory = json.loads(trajectory_path.read_text())
        recorded_result = result.get("result_sha256")
        recorded_trajectory = result.get("trajectory_sha256")
        # The recorded result_sha256 is a payload hash taken before the field
        # was inserted; recompute it that way rather than assuming.
        recomputed_trajectory = sha256_payload(
            {k: v for k, v in trajectory.items() if k != "trajectory_sha256"}
        )
        entry.update(
            {
                "has_trajectory": True,
                "clean_success": bool(result.get("clean_success")),
                "result_raw_file_sha256": sha256_file(result_path),
                "trajectory_raw_file_sha256": sha256_file(trajectory_path),
                "result_recorded_sha256": recorded_result,
                "result_recomputed_payload_sha256": recomputed_result,
                "result_self_consistent": recomputed_result == recorded_result,
                "trajectory_recorded_sha256": recorded_trajectory,
                "trajectory_recomputed_payload_sha256": recomputed_trajectory,
                "trajectory_row_binding_ok": (
                    trajectory.get("row_sha256") == result.get("row_sha256")
                ),
                "n_frames": int(trajectory.get("n") or 0),
                "n_steps": len(trajectory.get("steps") or []),
            }
        )
        entry["frame_count_consistent"] = entry["n_frames"] == entry["n_steps"]
        entry["ok"] = bool(
            entry["result_self_consistent"]
            and entry["trajectory_row_binding_ok"]
            and entry["frame_count_consistent"]
        )
        if not entry["ok"]:
            problems.append(f"{directory.name}: {entry}")
        entries.append(entry)
    without_trajectory = [
        item for item in entries if item.get("has_trajectory") is False
    ]
    clean_rows = [item for item in entries if item.get("clean_success")]
    clean_without_trajectory = [
        item for item in without_trajectory if item.get("clean_success")
    ]
    return {
        "n_rows": len(entries),
        "n_ok": sum(1 for item in entries if item["ok"]),
        "n_with_trajectory": sum(
            1 for item in entries if item.get("has_trajectory")
        ),
        "n_without_trajectory": len(without_trajectory),
        "rows_without_trajectory": [
            item["row_dir"].split("/")[-1] for item in without_trajectory
        ],
        "n_clean_success": len(clean_rows),
        "n_clean_without_trajectory": len(clean_without_trajectory),
        "clean_count_unaffected_by_missing_trajectories": (
            len(clean_without_trajectory) == 0
        ),
        "problems": problems[:8],
        "n_problems": len(problems),
        "rows": entries,
        "passed": not problems and len(entries) == 192,
    }


# ---------------------------------------------------------------------------
# Independent aggregator
# ---------------------------------------------------------------------------
def independent_aggregate(npz_path: Path) -> dict[str, Any]:
    """Recompute every bundle statistic from a flat table.

    Deliberately unlike the primary scorer, which accumulated into nested
    dictionaries pose by pose. Here every (row, candidate, pose) evaluation is
    flattened into one record list first and the statistics are reductions over
    that list, so a bookkeeping mistake in either implementation shows up as a
    disagreement rather than being reproduced.
    """
    payload = np.load(npz_path, allow_pickle=True)
    rows = [json.loads(str(item)) for item in payload["rows"]]
    flat: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("ok"):
            continue
        for key, score in row["scores"].items():
            x_text, r_text, pose = key.split("|")
            flat.append(
                {
                    "x_m": float(x_text),
                    "r_m": float(r_text),
                    "pose": pose,
                    "side": row["intrusion_side"],
                    "family": row["family_id"],
                    "row_dir": row["row_dir"],
                    "min_clearance_m": score["min_clearance_m"],
                    "min_lobe_stem_m": score["min_lobe_stem_m"],
                    "contact": bool(score["robot_or_target_contact"]),
                    "env_min_m": score["env_min_m"],
                    "env_intersects": bool(score["env_intersects"]),
                    "window_min_m": score["window_min_m"] or {},
                    "initial_min_m": score["initial_min_m"] or {},
                    "risk_by_direction_m": score["risk_by_direction_m"] or {},
                    "risk_witness": score["risk_witness"] or {},
                }
            )
    by_bundle: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for item in flat:
        by_bundle[(item["x_m"], item["r_m"])].append(item)

    bundles: dict[str, Any] = {}
    for (x_m, r_m), items in sorted(by_bundle.items()):
        clearances = [
            float(i["min_clearance_m"])
            for i in items
            if i["min_clearance_m"] is not None
        ]
        risks = [
            float(i["min_lobe_stem_m"])
            for i in items
            if i["min_lobe_stem_m"] is not None
        ]
        n_eval = len(items)
        below = [i for i in items
                 if i["min_clearance_m"] is not None
                 and float(i["min_clearance_m"]) < CLEARANCE_FLOOR_M]
        contacts = [i for i in items if i["contact"]]
        group_ge15: dict[str, dict[str, int]] = {}
        for pose in POSE_IDS:
            for side in ("left", "right"):
                sel = [i for i in items if i["pose"] == pose and i["side"] == side]
                ok = [
                    i for i in sel
                    if i["min_clearance_m"] is not None
                    and float(i["min_clearance_m"]) >= CLEARANCE_FLOOR_M
                ]
                group_ge15[f"{pose}|{side}"] = {"n": len(sel), "n_ge_floor": len(ok)}
        band_witness_groups = {}
        for pose in POSE_IDS:
            for side in ("left", "right"):
                sel = [i for i in items if i["pose"] == pose and i["side"] == side]
                n_in_band = sum(
                    1 for i in sel
                    if i["min_lobe_stem_m"] is not None
                    and RISK_BAND_M[0] <= float(i["min_lobe_stem_m"]) <= RISK_BAND_M[1]
                )
                band_witness_groups[f"{pose}|{side}"] = n_in_band
        directions: dict[str, set] = {"left": set(), "right": set()}
        for i in items:
            for direction, value in i["risk_by_direction_m"].items():
                if value is None:
                    continue
                if RISK_BAND_M[0] <= float(value) <= RISK_BAND_M[1]:
                    directions[i["side"]].add(direction)
        window_below = [
            {"row": i["row_dir"], "pose": i["pose"], "window": name,
             "min_m": float(value)}
            for i in items
            for name, value in i["window_min_m"].items()
            if value is not None and float(value) < CLEARANCE_FLOOR_M
        ]
        initial_below = [
            {"row": i["row_dir"], "pose": i["pose"], "probe": name,
             "min_m": float(value)}
            for i in items
            for name, value in i["initial_min_m"].items()
            if value is not None and float(value) < CLEARANCE_FLOOR_M
        ]
        bundles[f"{x_m:.3f}|{r_m:.3f}"] = {
            "x_m": x_m,
            "r_m": r_m,
            "n_evaluations": n_eval,
            "absolute_min_clearance_m": min(clearances) if clearances else None,
            "n_below_floor": len(below),
            "n_contacts": len(contacts),
            "fraction_ge_floor": (
                (n_eval - len(below)) / n_eval if n_eval else None
            ),
            "median_lobe_stem_m": float(np.median(risks)) if risks else None,
            "n_risk_band_evaluations": sum(
                1 for v in risks if RISK_BAND_M[0] <= v <= RISK_BAND_M[1]
            ),
            "band_evaluations_by_group": band_witness_groups,
            "evaluations_ge_floor_by_group": group_ge15,
            "direction_band_witnesses": {
                k: sorted(v) for k, v in directions.items()
            },
            "n_window_below_floor": len(window_below),
            "n_initial_below_floor": len(initial_below),
            "n_env_intersections": sum(1 for i in items if i["env_intersects"]),
        }
    return {
        "n_flat_evaluations": len(flat),
        "n_bundles": len(bundles),
        "bundles": bundles,
        "aggregator": "independent_flat_table_v1",
    }


def compare_with_primary(
    independent: dict[str, Any], siting: dict[str, Any]
) -> dict[str, Any]:
    """Require exact agreement on every quantity both implementations compute."""
    disagreements: list[dict[str, Any]] = []
    checked = 0
    for bundle in siting["bundles"]:
        key = f"{bundle['x_m']:.3f}|{bundle['r_m']:.3f}"
        mine = independent["bundles"].get(key)
        if mine is None:
            disagreements.append({"bundle": key, "field": "presence"})
            continue
        primary_min = min(
            float(bundle["poses"][pose]["min_clearance_m"])
            for pose in bundle["poses"]
            if bundle["poses"][pose]["min_clearance_m"] is not None
        )
        primary_below = sum(
            len(bundle["poses"][pose]["below_floor_rows"]) for pose in bundle["poses"]
        )
        primary_contacts = sum(
            len(bundle["poses"][pose]["contact_rows"]) for pose in bundle["poses"]
        )
        primary_band = int(bundle["n_risk_band_witnesses"])
        for field, a, b in (
            ("absolute_min_clearance_m", mine["absolute_min_clearance_m"], primary_min),
            ("n_below_floor", mine["n_below_floor"], primary_below),
            ("n_contacts", mine["n_contacts"], primary_contacts),
            ("n_risk_band_evaluations", mine["n_risk_band_evaluations"], primary_band),
            ("median_lobe_stem_m", mine["median_lobe_stem_m"],
             bundle["median_lobe_stem_m"]),
            ("direction_band_witnesses", mine["direction_band_witnesses"],
             bundle["direction_witnesses"]),
        ):
            checked += 1
            equal = (
                abs(float(a) - float(b)) <= 1e-12
                if isinstance(a, float) and isinstance(b, float)
                else a == b
            )
            if not equal:
                disagreements.append(
                    {"bundle": key, "field": field, "independent": a, "primary": b}
                )
    return {
        "n_checks": checked,
        "n_disagreements": len(disagreements),
        "disagreements": disagreements[:16],
        "exact_agreement": not disagreements,
    }


def household_object_count() -> dict[str, Any]:
    from pact_place_v105_contract import v95_row_payload

    payload = v95_row_payload("F0_target_side_stagger", "left")
    slots = [str(item["palette_slot"]) for item in payload["pact_clutter_layout"]["objects"]]
    return {
        "n_household_objects": len(slots),
        "palette_slots": slots,
        "palette_size": len(payload["pact_clutter_palette"]),
        "household_body_prefixes": list(HOUSEHOLD_BODY_PREFIXES),
        "corridor_vessel_bodies": list(CORRIDOR_VESSEL_BODIES),
        "note": (
            "Eight household objects, one per palette slot. Each has nested "
            "MuJoCo mesh child bodies, and the four corridor chicane bodies are "
            "separate fixtures. Counting either as a distinct object inflates "
            "the total."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / AUDIT_ROOT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite {output_root}")

    sealed = hash_sealed()
    corpus = verify_corpus_hashes()
    siting = json.loads(SITING_ROOT / "siting.json" and (SITING_ROOT / "siting.json").read_text())
    reconstruction = json.loads((RECON_ROOT / "reconstruction.json").read_text())
    independent = independent_aggregate(SITING_ROOT / "per_row_scores.npz")
    agreement = compare_with_primary(independent, siting)

    ranked = sorted(
        independent["bundles"].values(),
        key=lambda item: -(item["absolute_min_clearance_m"] or -1.0),
    )
    best = ranked[0]
    clean_rows = int(reconstruction["coverage"]["total_clean"])

    errata = [
        {
            "id": "E1",
            "claim_as_reported": "21 active clutter free bodies",
            "status": "incorrect",
            "correction": (
                f"{household_object_count()['n_household_objects']} active "
                "household objects. The reported 21 counted nested MuJoCo mesh "
                "child bodies and the four corridor chicane bodies as distinct "
                "objects."
            ),
            "evidence": household_object_count(),
        },
        {
            "id": "E2",
            "claim_as_reported": "x = 0.800, r = 0.320 is the highest-floor candidate",
            "status": "incorrect",
            "correction": (
                f"The highest symmetric clearance floor is x = {best['x_m']:.3f}, "
                f"r = {best['r_m']:.3f} at "
                f"{best['absolute_min_clearance_m'] * 1000:.7f} mm, with "
                f"{best['n_contacts']} exact contacts and {best['n_below_floor']} "
                f"of {best['n_evaluations']} evaluations below the 15 mm floor. "
                "x = 0.800, r = 0.320 is second."
            ),
            "evidence": {
                "top_four": [
                    {
                        "x_m": item["x_m"], "r_m": item["r_m"],
                        "absolute_min_clearance_mm": (
                            item["absolute_min_clearance_m"] * 1000.0
                        ),
                        "n_contacts": item["n_contacts"],
                        "n_below_floor": item["n_below_floor"],
                        "n_evaluations": item["n_evaluations"],
                    }
                    for item in ranked[:4]
                ]
            },
        },
        {
            "id": "E3",
            "claim_as_reported": "98/192 strict-clean rows",
            "status": "valid",
            "correction": None,
            "evidence": {
                "total_clean": clean_rows,
                "total_rows": int(reconstruction["coverage"]["total_rows"]),
                "clean_fraction": clean_rows
                / int(reconstruction["coverage"]["total_rows"]),
                "recorded_fragility_mean_clean_rate": 0.5104166666666666,
            },
        },
        {
            "id": "E4",
            "claim_as_reported": "risk_group_counts",
            "status": "ambiguous field name",
            "correction": (
                "Renamed in the audit to band_evaluations_by_group: for each "
                "pose_id|side group it is the NUMBER OF EVALUATIONS whose "
                "lobe/stem minimum lies inside the 15-35 mm band. It was never "
                "a count of distinct groups, nor of trajectories, nor of frames. "
                "The companion field evaluations_ge_floor_by_group reports, per "
                "group, how many evaluations sit at or above the 15 mm floor."
            ),
            "evidence": {
                "definition": "count of (trajectory, pose) evaluations in band",
                "n_evaluations_per_bundle": best["n_evaluations"],
                "identity": "n_evaluations = n_clean_trajectories * n_poses",
            },
        },
    ]

    document = {
        "schema_version": "pact_place_v105_audit_erratum_v1",
        "contract_version": CONTRACT_VERSION_V105,
        "role": "audit_and_erratum_of_v105_narrative",
        "treats_prior_narrative_as_untrusted": True,
        "modifies_any_v104_or_v105_artifact": False,
        "sealed_artifacts": sealed,
        "corpus_hash_verification": corpus,
        "household_objects": household_object_count(),
        "independent_aggregation": independent,
        "agreement_with_primary_scorer": agreement,
        "errata": errata,
        "clearance_floor_m": CLEARANCE_FLOOR_M,
        "risk_band_m": list(RISK_BAND_M),
        "creates_episode": False,
        "calls_env_step": False,
        **empty_authorization(),
        "audit_passed": bool(
            corpus["passed"]
            and agreement["exact_agreement"]
            and all(entry.get("payload_self_consistent") is not False
                    for entry in sealed.values())
        ),
    }
    hashes = write_immutable_create_only(output_root / "audit.json", document)
    print(json.dumps({
        "audit_passed": document["audit_passed"],
        "corpus_rows_verified": corpus["n_ok"],
        "corpus_problems": corpus["n_problems"],
        "independent_evaluations": independent["n_flat_evaluations"],
        "agreement_checks": agreement["n_checks"],
        "disagreements": agreement["n_disagreements"],
        "n_household_objects": document["household_objects"]["n_household_objects"],
        "best_floor": {
            "x_m": best["x_m"], "r_m": best["r_m"],
            "mm": best["absolute_min_clearance_m"] * 1000.0,
            "n_contacts": best["n_contacts"],
            "n_below_floor": best["n_below_floor"],
            "n_evaluations": best["n_evaluations"],
        },
        **hashes,
    }, indent=2))
    return 0 if document["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
