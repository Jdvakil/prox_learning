#!/usr/bin/env python3
"""Build the predeclared canonical 75/25 subset and the frozen 80/20 split.

Handoff steps 11-13. Implements the selection rule committed in
``configs/hybrid_obstacle_independent_v2.yaml`` and echoed in the manifest's
``canonical_selection_rule``:

    first N successful rows of each hazard stratum, ordered by the predeclared
    ``stratum_rank``; no downstream metric participates.

and the committed split rule::

    train: 60 hazard-present + 20 hazard-absent
    val:   15 hazard-present +  5 hazard-absent

The split is derived by applying the committed per-stratum rank boundaries to
the *selected* set: within each stratum, selection positions [0, train_n) are
training and [train_n, train_n + val_n) are validation. When every row in the
first 75/25 by rank succeeds this reproduces the manifest's own ``split``
column exactly; when an earlier-ranked row failed, a later-ranked successful row
is promoted, which is the only way the committed quota can still be met without
inspecting rollout quality.

Nothing here reads clearance, collisions, trajectory length, retry count,
proximity activation, action statistics, image quality, planner phase timings or
any model score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "controlled_predeclared_canonical_subset"
QUOTA = {"hazard_present": 75, "hazard_absent": 25}
SPLIT = {
    "train": {"hazard_present": 60, "hazard_absent": 20},
    "val": {"hazard_present": 15, "hazard_absent": 5},
}


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--audit", required=True, type=Path, help="integrity audit report JSON")
    ap.add_argument("--source-manifest", required=True, type=Path)
    ap.add_argument("--out-canonical", required=True, type=Path)
    ap.add_argument("--out-split", required=True, type=Path)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    audit = json.loads(args.audit.read_text())
    source_manifest = json.loads(args.source_manifest.read_text())

    rows_by_eid = {r["episode_id"]: r for r in manifest["rows"]}
    audit_by_eid = {r["episode_id"]: r for r in audit["per_row"]}

    # ---- outcomes straight from the ledger; manifest identity only ---------
    outcomes: dict[str, dict] = {}
    for d in sorted((args.run / "rows").iterdir()):
        oc = d / "outcome.json"
        if oc.is_file():
            o = json.loads(oc.read_text())
            outcomes[o["episode_id"]] = o

    successes = {eid: o for eid, o in outcomes.items() if o["status"] == "success"}
    failures = {eid: o for eid, o in outcomes.items() if o["status"] != "success"}

    strata: dict[str, list[dict]] = {"hazard_present": [], "hazard_absent": []}
    for eid in successes:
        row = rows_by_eid[eid]
        key = "hazard_present" if row["hazard_present"] else "hazard_absent"
        strata[key].append(row)
    for stratum in strata.values():
        stratum.sort(key=lambda r: r["stratum_rank"])

    counts = {k: len(v) for k, v in strata.items()}
    print(f"distinct successful rows: {counts}")
    print(f"quota required          : {QUOTA}")

    shortfall = {k: QUOTA[k] - counts[k] for k in QUOTA if counts[k] < QUOTA[k]}
    if shortfall:
        print(f"QUOTA FAILED — shortfall {shortfall}", file=sys.stderr)
        payload = {
            "schema": "hybrid_obstacle_full_collection_quota_failure",
            "distinct_successes": counts,
            "quota": QUOTA,
            "shortfall": shortfall,
            "decision": "FULL_COLLECTION_QUOTA_FAILED",
        }
        args.out_canonical.parent.mkdir(parents=True, exist_ok=True)
        args.out_canonical.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 2

    # ---- step 12: selection ------------------------------------------------
    selected: list[dict] = []
    excluded: list[dict] = []
    for key, quota in QUOTA.items():
        stratum = strata[key]
        for pos, row in enumerate(stratum):
            eid = row["episode_id"]
            a = audit_by_eid.get(eid, {})
            entry = {
                "episode_id": eid,
                "candidate_index": row["candidate_index"],
                "row_sha256": row["row_sha256"],
                "hazard_present": bool(row["hazard_present"]),
                "hazard_stratum": key,
                "predeclared_stratum_rank": row["stratum_rank"],
                "predeclared_canonical_rank": row["canonical_rank"],
                "manifest_split_label": row["split"],
                "selection_position": pos,
                "source_h5_sha256": a.get("hash_A_file_sha256"),
                "core_trajectory_sha256": a.get("hash_C_core_trajectory_sha256"),
                "task_state_sha256": a.get("hash_D_task_state_sha256"),
                "episode_spec_sha256": a.get("hash_E_episode_spec_sha256"),
                "source_relpath": f"rows/{eid}/trajectory.h5",
            }
            if pos < quota:
                entry["selection_reason"] = (
                    f"selected: successful {key} row at selection position {pos} "
                    f"< quota {quota}, ordered by predeclared stratum_rank "
                    f"{row['stratum_rank']}"
                )
                selected.append(entry)
            else:
                entry["exclusion_reason"] = (
                    f"excluded: successful {key} row at selection position {pos} "
                    f">= quota {quota} by predeclared stratum_rank"
                )
                excluded.append(entry)

    assert len(selected) == 100, len(selected)

    # ---- step 13: split ----------------------------------------------------
    for key in QUOTA:
        chosen = sorted(
            (e for e in selected if e["hazard_stratum"] == key),
            key=lambda e: e["predeclared_stratum_rank"],
        )
        n_train = SPLIT["train"][key]
        n_val = SPLIT["val"][key]
        for i, e in enumerate(chosen):
            if i < n_train:
                e["split"] = "train"
                e["split_rank"] = i
            elif i < n_train + n_val:
                e["split"] = "validation"
                e["split_rank"] = i - n_train
            else:  # pragma: no cover - quota guarantees this is unreachable
                raise AssertionError("selection larger than the committed split")

    selected.sort(key=lambda e: (not e["hazard_present"], e["predeclared_stratum_rank"]))
    for g, e in enumerate(selected):
        e["act_episode_index"] = g

    promoted = [e for e in selected if e["manifest_split_label"] == "reserve"]
    split_agree = [
        e
        for e in selected
        if e["manifest_split_label"] in ("train", "val")
        and e["split"] == ("validation" if e["manifest_split_label"] == "val" else "train")
    ]

    failed_rows = [
        {
            "episode_id": eid,
            "candidate_index": rows_by_eid[eid]["candidate_index"],
            "row_sha256": rows_by_eid[eid]["row_sha256"],
            "hazard_present": bool(rows_by_eid[eid]["hazard_present"]),
            "predeclared_stratum_rank": rows_by_eid[eid]["stratum_rank"],
            "outcome": o["status"],
            "retry_count": o.get("retry_count"),
            "failure_reason": str(o.get("failure_reason") or o.get("reason") or "")[:300],
        }
        for eid, o in sorted(failures.items(), key=lambda kv: rows_by_eid[kv[0]]["candidate_index"])
    ]

    selection_code_hash = sha256_file(Path(__file__).resolve())

    canonical = {
        "schema": SCHEMA,
        "label": "controlled_predeclared_canonical_subset",
        "manifest_version": manifest["manifest_version"],
        "master_seed": manifest["master_seed"],
        "source_manifest_sha256": manifest["manifest_sha256"],
        "source_collection_tree_sha256": source_manifest["tree_sha256"],
        "source_collection_file_count": source_manifest["file_count"],
        "source_collection_total_bytes": source_manifest["total_bytes"],
        "source_run_dir": str(args.run),
        "selection_rule": manifest["canonical_selection_rule"],
        "selection_rule_applied": (
            "first N successful rows per hazard stratum ordered by predeclared "
            "stratum_rank; no downstream metric consulted"
        ),
        "split_rule": manifest["split_rule"],
        "split_rule_applied": (
            "committed per-stratum rank boundaries applied to the selected set: "
            "positions [0,train_n) -> train, [train_n,train_n+val_n) -> validation"
        ),
        "selection_code_sha256": selection_code_hash,
        "composition": {
            "total": len(selected),
            "hazard_present": sum(1 for e in selected if e["hazard_present"]),
            "hazard_absent": sum(1 for e in selected if not e["hazard_present"]),
        },
        "distinct_successes": counts,
        "quota": QUOTA,
        "promoted_from_reserve_count": len(promoted),
        "manifest_split_label_agreement": len(split_agree),
        "selected": selected,
        "excluded_successful": excluded,
        "failed_rows": failed_rows,
    }
    canonical["manifest_sha256"] = canonical_hash(canonical)

    split_doc = {
        "schema": "hybrid_obstacle_canonical_split_v2",
        "canonical_manifest_sha256": canonical["manifest_sha256"],
        "source_collection_tree_sha256": source_manifest["tree_sha256"],
        "split_rule": manifest["split_rule"],
        "level": "trajectory",
        "counts": {
            "train": {
                "total": sum(1 for e in selected if e["split"] == "train"),
                "hazard_present": sum(
                    1 for e in selected if e["split"] == "train" and e["hazard_present"]
                ),
                "hazard_absent": sum(
                    1 for e in selected if e["split"] == "train" and not e["hazard_present"]
                ),
            },
            "validation": {
                "total": sum(1 for e in selected if e["split"] == "validation"),
                "hazard_present": sum(
                    1 for e in selected if e["split"] == "validation" and e["hazard_present"]
                ),
                "hazard_absent": sum(
                    1 for e in selected if e["split"] == "validation" and not e["hazard_present"]
                ),
            },
        },
        "episodes": [
            {
                "episode_id": e["episode_id"],
                "candidate_index": e["candidate_index"],
                "act_episode_index": e["act_episode_index"],
                "hazard_present": e["hazard_present"],
                "split": e["split"],
                "split_rank": e["split_rank"],
                "source_h5_sha256": e["source_h5_sha256"],
                "row_sha256": e["row_sha256"],
                "core_trajectory_sha256": e["core_trajectory_sha256"],
                "task_state_sha256": e["task_state_sha256"],
                "source_relpath": e["source_relpath"],
            }
            for e in selected
        ],
    }

    # ---- leakage audit ----------------------------------------------------
    def by_split(field: str, split: str) -> set:
        return {e[field] for e in selected if e["split"] == split and e.get(field)}

    leak = {}
    for field in ("episode_id", "source_h5_sha256", "core_trajectory_sha256", "task_state_sha256", "source_relpath"):
        overlap = by_split(field, "train") & by_split(field, "validation")
        leak[field] = sorted(overlap)
    ids = [e["episode_id"] for e in selected]
    files = [e["source_relpath"] for e in selected]
    leak["duplicate_episode_ids_within_selection"] = sorted(
        {i for i in ids if ids.count(i) > 1}
    )
    leak["duplicate_source_files_within_selection"] = sorted(
        {i for i in files if files.count(i) > 1}
    )
    split_doc["leakage_audit"] = leak
    split_doc["leakage_free"] = all(not v for v in leak.values())
    split_doc["split_manifest_sha256"] = canonical_hash(
        {k: v for k, v in split_doc.items() if k != "split_manifest_sha256"}
    )

    args.out_canonical.parent.mkdir(parents=True, exist_ok=True)
    args.out_canonical.write_text(json.dumps(canonical, indent=2, sort_keys=True) + "\n")
    args.out_split.parent.mkdir(parents=True, exist_ok=True)
    args.out_split.write_text(json.dumps(split_doc, indent=2, sort_keys=True) + "\n")

    print(f"selected                : {canonical['composition']}")
    print(f"excluded successful     : {len(excluded)}")
    print(f"failed rows             : {len(failed_rows)}")
    print(f"promoted from reserve   : {len(promoted)}")
    print(f"canonical manifest sha  : {canonical['manifest_sha256']}")
    print(f"split counts            : {split_doc['counts']}")
    print(f"split manifest sha      : {split_doc['split_manifest_sha256']}")
    print(f"leakage free            : {split_doc['leakage_free']}")
    print(f"wrote {args.out_canonical}")
    print(f"wrote {args.out_split}")
    return 0 if split_doc["leakage_free"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
