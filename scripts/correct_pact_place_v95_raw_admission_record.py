#!/usr/bin/env python3
"""E0: correct the V9.5 raw-admission record without touching a V9.5 artifact.

The retained `validation.json` reports `passing_variant_count: 1` of 8.  Joining
its variants to the smoke summary's `clean_success` shows that the single pass
is the one variant whose source episode is **not collision-free**
(F3_aperture_side_stagger / left: 351 clutter contacts; its right pair has
2,315).  Every physics-clean variant failed.  The correct headline is
**0 of 6 physics-clean variants passed**.

That matters beyond bookkeeping.  The inbound vessel's only nonzero reading in
all of V9.5 -- 40 changed values on `link5_back_sensor_4`, max 3.1 px at
R = 0.11 m -- comes from that episode.  With the arm already contacting clutter
that is a sensor nearly touching an object, not a detection at range.

This script re-applies the admission rule, now including the clean-source gate
added to `run_pact_place_v9_v0c3_causal_proximity.py`, to the **retained**
per-variant metrics.  It renders nothing, recomputes no tensor, and writes a new
record beside the original.  `validation.json` is read-only here and its digest
is recorded so the correction can be shown not to have edited it.

Nothing here authorizes a gate, collection, or V1b.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pact_place_corridor_contract import sha256_file  # noqa: E402
from pact_place_v9_contract import sha256_payload  # noqa: E402

DEFAULT_VALIDATION = (
    ROOT / "diagnostics_output" / "pact_place_v95_v0c5_raw_prerequisite" / "validation.json"
)
DEFAULT_SMOKE_SUMMARY = (
    ROOT / "diagnostics_output" / "pact_place_v95_raw_smoke" / "summary.json"
)
ROLES = ("panel", "inbound_vessel", "outbound_vessel")


def _signal_passed(variant: dict[str, Any]) -> bool:
    """The rule the retained artifact used: any nonzero pixel, on every role."""
    return all(
        int(variant[f"{role}_causal_effect"]["changed_values"]) > 0
        and variant[f"{role}_causal_effect"]["first_activation"] is not None
        for role in ROLES
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--smoke-summary", type=Path, default=DEFAULT_SMOKE_SUMMARY)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    validation_path = args.validation.resolve()
    smoke_path = args.smoke_summary.resolve()
    validation = json.loads(validation_path.read_text())
    smoke = json.loads(smoke_path.read_text())

    clean_by_episode = {
        str(item["episode_id"]): bool(item.get("clean_success"))
        for item in list(smoke.get("results") or [])
    }
    contacts_by_episode = {
        str(item["episode_id"]): {
            "clutter_contacts": int(item.get("clutter_contacts", 0)),
            "hazard_bar_contacts": int(item.get("hazard_bar_contacts", 0)),
            "collision_free": item.get("collision_free"),
            "task_success": item.get("task_success"),
        }
        for item in list(smoke.get("results") or [])
    }

    rows = []
    for index, variant in enumerate(validation["variants"]):
        episode_id = str(variant["episode_id"])
        if episode_id not in clean_by_episode:
            raise RuntimeError(f"variant {index} has no smoke row: {episode_id}")
        clean = clean_by_episode[episode_id]
        signal = _signal_passed(variant)
        rows.append(
            {
                "index": index,
                "family_id": str(variant["family_id"]),
                "intrusion_side": str(variant["intrusion_side"]),
                "episode_id": episode_id,
                "retained_passed": bool(variant["passed"]),
                "signal_passed": signal,
                "source_physics_clean": clean,
                "corrected_passed": bool(signal and clean),
                "source_contacts": contacts_by_episode[episode_id],
                "changed_values": {
                    role: int(variant[f"{role}_causal_effect"]["changed_values"])
                    for role in ROLES
                },
                "changed_sensors": {
                    role: int(variant[f"{role}_causal_effect"]["changed_sensors"])
                    for role in ROLES
                },
            }
        )

    clean_rows = [row for row in rows if row["source_physics_clean"]]
    dirty_rows = [row for row in rows if not row["source_physics_clean"]]
    retained_headline = (
        f"{sum(row['retained_passed'] for row in rows)} of {len(rows)}"
    )
    corrected_headline = (
        f"{sum(row['corrected_passed'] for row in clean_rows)} of {len(clean_rows)}"
    )

    # Where the only nonzero inbound reading in all of V9.5 came from.
    inbound_nonzero = [
        {
            "family_id": row["family_id"],
            "intrusion_side": row["intrusion_side"],
            "changed_values": row["changed_values"]["inbound_vessel"],
            "source_physics_clean": row["source_physics_clean"],
        }
        for row in rows
        if row["changed_values"]["inbound_vessel"] > 0
    ]

    document = {
        "schema_version": "pact_place_v9_5_raw_admission_correction_v1",
        "role": "record_correction_not_a_new_measurement",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "authorizes_v1b": False,
        "renders_nothing": True,
        "recomputes_no_tensor": True,
        "edits_no_v95_artifact": True,
        "corrected_headline": corrected_headline + " physics-clean variants passed",
        "retained_headline_as_published": retained_headline + " variants passed",
        "variant_count": len(rows),
        "physics_clean_variant_count": len(clean_rows),
        "dirty_source_variant_count": len(dirty_rows),
        "passing_physics_clean_variant_count": sum(
            row["corrected_passed"] for row in clean_rows
        ),
        "retained_pass_was_a_dirty_source": bool(
            any(row["retained_passed"] and not row["source_physics_clean"] for row in rows)
        ),
        "inbound_vessel_nonzero_readings": inbound_nonzero,
        "join_table": rows,
        "dirty_source_variants": dirty_rows,
        "unaffected": {
            "w1_retrodiction": (
                "W1 predicts the renderer's output from posed geometry. It is a claim about "
                "the sensor model, not about whether the pose was reached without contact, so "
                "r = 0.99997 stands unchanged."
            ),
            "w2_structural_finding": (
                "W2's floor required passing in every variant, so the two dirty episodes made "
                "admission harder, not easier. Removing them cannot create an admission that "
                "did not exist."
            ),
        },
        "affected": {
            "w3_pipeline_validation": (
                "The 40 -> 2,604 comparison used F3_aperture_side_stagger as both baseline and "
                "test, on both sides. Both of those source episodes are dirty. The comparison is "
                "internally consistent but must not be quoted as a clean-source result."
            )
        },
        "validation_path": str(validation_path.relative_to(ROOT)),
        "validation_sha256": sha256_file(validation_path),
        "smoke_summary_path": str(smoke_path.relative_to(ROOT)),
        "smoke_summary_sha256": sha256_file(smoke_path),
        "corrector_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "corrector_sha256": sha256_file(Path(__file__).resolve()),
    }
    document["document_sha256"] = sha256_payload(document)

    output = args.output or (validation_path.parent / "admission_correction.json")
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(output)
    print(
        json.dumps(
            {
                "corrected": document["corrected_headline"],
                "as_published": document["retained_headline_as_published"],
                "retained_pass_was_a_dirty_source": document["retained_pass_was_a_dirty_source"],
            },
            sort_keys=True,
        )
    )
    header = f"{'#':<3}{'family':<28}{'side':<7}{'raw_passed':<12}{'source_clean':<14}{'corrected':<10}"
    print(header)
    for row in rows:
        print(
            f"{row['index']:<3}{row['family_id']:<28}{row['intrusion_side']:<7}"
            f"{str(row['retained_passed']):<12}{str(row['source_physics_clean']):<14}"
            f"{str(row['corrected_passed']):<10}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
