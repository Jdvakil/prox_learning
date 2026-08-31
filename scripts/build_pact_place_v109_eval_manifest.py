#!/usr/bin/env python3
"""V10.9 step 8: build and freeze the paired 40-instance evaluation manifest.

Every seed the environment has ever consumed for this pendant is gathered and
the evaluation seeds are asserted disjoint from all of them:

* the 353 V10.8 scientific attempt seeds, reconstructed from their cell streams
  rather than read back from the ledger
* the 141 accepted training rows
* the V10.7 48-row pool and the 24-row Phase-0 gate
* the V10.9 evaluation smoke rows
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v108_contract import cell_seed, cells, quotas  # noqa: E402
from pact_place_v109_contract import (  # noqa: E402
    CANONICAL_SENSOR_NAMES,
    CONTRACT_VERSION_V109,
    EVAL_ROOT,
    SENSOR_ORDER_SHA256,
    WORK_ROOT,
    canonical_payload_sha256,
    empty_authorization,
    write_immutable_create_only,
)
from pact_place_v109_eval_contract import build_manifest, load_manifest  # noqa: E402

MAX_ATTEMPT_INDEX = 64  # V10.8 never exceeded this within a cell


def v108_scientific_seeds() -> tuple[list[int], list[int]]:
    """Reconstruct the attempted seeds from the frozen cell streams, and the
    superset of every seed those streams could have produced."""
    ledger = ROOT / "diagnostics_output/pact_place_v108_collection/ledger.jsonl"
    rows = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    reconstructed: list[int] = []
    for row in rows:
        family, side, pose = row["cell"].split("|")
        seed = cell_seed(family, side, pose, int(row["attempt_index"]))
        if int(seed["seed_u32"]) != int(row["task_seed_u32"]):
            raise SystemExit(
                f"cell stream does not reproduce {row['attempt_id'][:16]}")
        reconstructed.append(int(seed["seed_u32"]))
    stream_superset = [
        int(cell_seed(f, s, p, i)["seed_u32"])
        for f, s, p in cells()
        for i in range(MAX_ATTEMPT_INDEX)
    ]
    return reconstructed, stream_superset


def v107_seeds() -> tuple[list[int], list[int]]:
    pool = json.loads(
        (ROOT / "diagnostics_output/pact_place_v107_pool/pool_manifest.json").read_text()
    )["expert_screen_rows"]
    gate = json.loads(
        (ROOT / "diagnostics_output/pact_place_v107_phase0/gate_manifest.json").read_text()
    )["expert_screen_rows"]
    return ([int(r["task_seed_u32"]) for r in pool],
            [int(r["task_seed_u32"]) for r in gate])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / EVAL_ROOT / "eval_manifest.json")
    args = parser.parse_args()

    source = json.loads((ROOT / WORK_ROOT / "source_manifest.json").read_text())
    accepted_seeds = [int(r["task_seed_u32"]) for r in source["rows"]]
    attempted, superset = v108_scientific_seeds()
    pool, gate = v107_seeds()

    excluded = {
        "v108_scientific_attempts": attempted,
        "v108_accepted_training_rows": accepted_seeds,
        "v108_cell_stream_superset": superset,
        "v107_pool": pool,
        "v107_phase0_gate": gate,
    }
    manifest = build_manifest(
        list(CANONICAL_SENSOR_NAMES), SENSOR_ORDER_SHA256, excluded)
    if not manifest["valid"]:
        print(json.dumps({"valid": False, "problems": manifest["problems"]}, indent=2))
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = write_immutable_create_only(args.out, manifest)
    reloaded = load_manifest(args.out)

    audit: dict[str, Any] = {
        **empty_authorization(),
        "schema_version": "pact_place_v109_eval_seed_audit_v1",
        "contract_version": CONTRACT_VERSION_V109,
        "role": "held-out seed disjointness audit for the V10.9 paired evaluation",
        "manifest_path": str(args.out.relative_to(ROOT)),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_raw_file_sha256": written.get("raw_file_sha256"),
        "excluded_source_sizes": {k: len(v) for k, v in excluded.items()},
        "excluded_unique_seeds": len({s for v in excluded.values() for s in v}),
        "v108_attempted_reproduced_from_cell_streams": True,
        "v107_pool_rows": len(pool),
        "v107_phase0_rows": len(gate),
        "collisions": manifest["held_out_seed_audit"]["collisions"],
        "disjoint": manifest["held_out_seed_audit"]["disjoint"],
        "quota_reference": quotas(),
    }
    audit["payload_sha256"] = canonical_payload_sha256(audit)
    write_immutable_create_only(ROOT / EVAL_ROOT / "eval_seed_audit.json", audit)

    print(json.dumps({
        "valid": reloaded["valid"],
        "instances": reloaded["total_candidates"],
        "rollouts": 2 * reloaded["total_candidates"],
        "balance": reloaded["balance"],
        "doubled_cells": len(reloaded["doubled_cells"]),
        "smoke_instances": reloaded["smoke"]["instances"],
        "seed_disjoint": reloaded["held_out_seed_audit"]["disjoint"],
        "excluded_unique_seeds": audit["excluded_unique_seeds"],
        "manifest_sha256": reloaded["manifest_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
