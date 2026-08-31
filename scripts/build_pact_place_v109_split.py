#!/usr/bin/env python3
"""V10.9 step 4: freeze the single 113/28 cell-stratified split.

One byte-identical split is used by both arms. The algorithm consults only
identity hashes -- ``attempt_id`` and the cell key under the frozen split seed.
It never looks at trajectory loss, length, clearance, contact, or any learned
outcome, so it cannot be tuned toward a result.

The output document satisfies ``fixed_split_data.load_split_manifest``:
schema ``hybrid_obstacle_canonical_split_v2`` with a self-hash recomputed over
the document with ``split_manifest_sha256`` removed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

sys.path.insert(0, str(ROOT / "submodules" / "act"))

from pact_place_v109_contract import (  # noqa: E402
    CONTRACT_VERSION_V109,
    N_ACCEPTED,
    SOLE_ROW_CELL,
    SPLIT_MASTER_SEED,
    TRAIN_COUNT,
    VALIDATION_COUNT,
    WORK_ROOT,
    freeze_split,
    quotas,
    write_immutable_text_create_only,
)

SPLIT_SCHEMA = "hybrid_obstacle_canonical_split_v2"
SPLIT_RULE = (
    "V10.9 cell-stratified split, master seed 2026082901. "
    "(1) the sole F3_aperture_side_stagger|right|neg5 row goes to training; "
    "(2) every other nonempty cell reserves one hash-ranked row for validation "
    "(23 rows); (3) the remaining five validation slots go by largest remainder "
    "across cells, capped so every cell keeps at least one training row, ties "
    "broken by SHA-256 of (split seed, cell key); (4) rows within a cell are "
    "ranked by SHA-256 of (split seed, attempt_id). No trajectory property is "
    "consulted."
)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path,
                        default=ROOT / WORK_ROOT / "source_manifest.json")
    parser.add_argument("--conversion-manifest", type=Path,
                        default=ROOT / WORK_ROOT / "conversion_manifest.json")
    parser.add_argument("--out", type=Path,
                        default=ROOT / WORK_ROOT / "split_manifest.json")
    args = parser.parse_args()

    source = json.loads(args.source_manifest.read_text())
    conversion = json.loads(args.conversion_manifest.read_text())
    if not source.get("verified"):
        raise SystemExit("source manifest is not verified")
    rows = source["rows"]
    by_id = {e["episode_id"]: e for e in conversion["episodes"]}
    if len(by_id) != N_ACCEPTED or {r["attempt_id"] for r in rows} != set(by_id):
        raise SystemExit("conversion manifest does not cover exactly the source rows")
    for row in rows:
        episode = by_id[row["attempt_id"]]
        if int(episode["act_episode_index"]) != int(row["act_episode_index"]):
            raise SystemExit(f"index drift for {row['attempt_id'][:16]}")

    frozen = freeze_split(rows)
    assignments = frozen["assignments"]

    episodes: list[dict[str, Any]] = []
    rank = {"train": 0, "validation": 0}
    for row in sorted(rows, key=lambda r: int(r["act_episode_index"])):
        label = assignments[row["attempt_id"]]
        episodes.append({
            "act_episode_index": int(row["act_episode_index"]),
            "episode_id": row["attempt_id"],
            "candidate_index": int(row["attempt_index"]),
            "hazard_present": True,
            "split": label,
            "split_rank": rank[label],
            "source_h5_sha256": row["trajectory_h5_sha256"],
            "cell": row["cell"],
            "family_id": row["family_id"],
            "intrusion_side": row["intrusion_side"],
            "pose_id": row["pose_id"],
        })
        rank[label] += 1

    train = [e for e in episodes if e["split"] == "train"]
    validation = [e for e in episodes if e["split"] == "validation"]
    if len(train) != TRAIN_COUNT or len(validation) != VALIDATION_COUNT:
        raise SystemExit(f"split is {len(train)}/{len(validation)}")
    if {e["act_episode_index"] for e in train} & {e["act_episode_index"] for e in validation}:
        raise SystemExit("train/validation overlap")
    if len({e["episode_id"] for e in episodes}) != N_ACCEPTED:
        raise SystemExit("an episode was assigned more than once")

    def tally(entries: list[dict[str, Any]], key: str) -> dict[str, int]:
        return dict(sorted(collections.Counter(e[key] for e in entries).items()))

    train_cells = set(tally(train, "cell"))
    validation_cells = set(tally(validation, "cell"))
    all_cells = set(quotas())
    if train_cells != all_cells:
        raise SystemExit(f"training covers {len(train_cells)} of 24 cells")
    if len(validation_cells) != 23 or validation_cells | {SOLE_ROW_CELL} != all_cells:
        raise SystemExit(f"validation covers {len(validation_cells)} cells, expected 23")

    document: dict[str, Any] = {
        "schema": SPLIT_SCHEMA,
        "experiment": "pact_place_v108_141_v109",
        "contract_version": CONTRACT_VERSION_V109,
        "split_master_seed": SPLIT_MASTER_SEED,
        "canonical_manifest_sha256": conversion["source_manifest_payload_sha256"],
        "source_collection_tree_sha256": conversion["converted_tree_file_sha256"],
        "source_collection_tree_hash_kind": "file",
        "conversion_manifest_payload_sha256": conversion["payload_sha256"],
        "split_rule": SPLIT_RULE,
        "counts": {
            "train": {"total": len(train), "hazard_present": len(train), "hazard_absent": 0},
            "validation": {"total": len(validation),
                           "hazard_present": len(validation), "hazard_absent": 0},
        },
        "stratification": {
            "cells_total": 24,
            "cells_in_train": len(train_cells),
            "cells_in_validation": len(validation_cells),
            "sole_row_cell": SOLE_ROW_CELL,
            "sole_row_cell_note":
                "one episode; mathematically cannot appear in both splits, assigned to training",
            "two_row_cell": "F3_aperture_side_stagger|right|pos5",
            "two_row_cell_note": "two episodes; splits 1/1",
            "extra_validation_slots_by_cell": frozen["extra_validation_by_cell"],
            "train_by_cell": tally(train, "cell"),
            "validation_by_cell": tally(validation, "cell"),
            "train_by_family": tally(train, "family_id"),
            "validation_by_family": tally(validation, "family_id"),
            "train_by_side": tally(train, "intrusion_side"),
            "validation_by_side": tally(validation, "intrusion_side"),
            "train_by_pose": tally(train, "pose_id"),
            "validation_by_pose": tally(validation, "pose_id"),
        },
        "underrepresentation_warning": {
            "note": "This dataset is NOT balanced. It is the V10.8 collection as "
                    "collected, stopped early by owner instruction at 141 of 152 "
                    "target successes. Any per-family reading of a downstream result "
                    "must account for this.",
            "accepted_by_family": source["underrepresentation"]["by_family"],
            "quota_by_family": {"each": 38},
            "f3_deficit": 38 - source["underrepresentation"]["by_family"][
                "F3_aperture_side_stagger"],
            "accepted_by_side": source["underrepresentation"]["by_side"],
            "cells_short": source["underrepresentation"]["cells_short"],
            "cells_over_quota": source["underrepresentation"]["cells_over_quota"],
        },
        "episodes": episodes,
    }
    document["split_manifest_sha256"] = canonical_hash(
        {k: v for k, v in document.items() if k != "split_manifest_sha256"}
    )
    # Written as text, not through write_immutable_create_only: that writer
    # appends its own ``payload_sha256`` to whatever it is handed, and
    # ``fixed_split_data.load_split_manifest`` recomputes the self-hash over
    # every key except ``split_manifest_sha256`` -- so an injected key makes the
    # manifest permanently unloadable. The training loader is not modified to
    # accommodate this; the manifest is written to match it.
    raw = write_immutable_text_create_only(
        args.out, json.dumps(document, indent=2, sort_keys=True) + "\n")
    written = {"raw_file_sha256": raw}

    from fixed_split_data import load_split_manifest  # noqa: PLC0415

    reloaded = load_split_manifest(str(args.out))
    if (len(reloaded["train"]), len(reloaded["val"])) != (TRAIN_COUNT, VALIDATION_COUNT):
        raise SystemExit("round trip through the training loader changed the split")
    print(json.dumps({
        "train": len(train),
        "validation": len(validation),
        "cells_in_train": len(train_cells),
        "cells_in_validation": len(validation_cells),
        "split_manifest_sha256": document["split_manifest_sha256"],
        "raw_file_sha256": written.get("raw_file_sha256"),
        "validation_by_family": document["stratification"]["validation_by_family"],
        "f3_deficit": document["underrepresentation_warning"]["f3_deficit"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
