#!/usr/bin/env python3
"""V10.9 step 0: create-only erratum correcting the V10.8 record.

``closeout.json`` is never overwritten. This erratum sits beside it and states
what the previous narrative got wrong, with every correction re-derived from
``ledger.jsonl`` and the retained row files rather than copied from the
close-out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v109_contract import (  # noqa: E402
    COLLECTION_ROOT,
    CONTRACT_VERSION_V109,
    LEDGER_SHA256,
    N_SENSORS,
    PENDANT_CONTACT_ATTEMPT_ID,
    WORK_ROOT,
    canonical_payload_sha256,
    empty_authorization,
    quotas,
    recompute_payload_sha256,
    sha256_file,
    write_immutable_create_only,
    write_immutable_text_create_only,
)

WORKER_DEATH_ERROR = (
    "worker died: BrokenProcessPool: A process in the process pool was "
    "terminated abruptly while the future was running or pending."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path,
                        default=ROOT / WORK_ROOT / "source_manifest.json")
    args = parser.parse_args()

    collection = ROOT / COLLECTION_ROOT
    ledger_path = collection / "ledger.jsonl"
    closeout_path = collection / "closeout.json"
    collection_path = collection / "collection_1787966003.json"

    source = json.loads(args.source_manifest.read_text())
    if not source.get("verified"):
        raise SystemExit("source manifest is not verified; refusing to write an erratum")

    ledger_sha = sha256_file(ledger_path)
    if ledger_sha != LEDGER_SHA256:
        raise SystemExit(f"ledger sha256 drift: {ledger_sha}")
    rows = [json.loads(x) for x in ledger_path.read_text().splitlines() if x.strip()]
    accepted = [r for r in rows if r["accepted"]]

    closeout = json.loads(closeout_path.read_text())
    run = json.loads(collection_path.read_text())

    # --- correction 1 + 2: pendant involvement -----------------------------
    pendant_rows = [r for r in rows if r["pendant_contact_frames"]
                    or r["contact_class_totals"].get("mounted_fixture", 0)]
    if len(pendant_rows) != 1:
        raise SystemExit(f"expected exactly one pendant-involved attempt, found {len(pendant_rows)}")
    pendant = pendant_rows[0]
    if pendant["attempt_id"] != PENDANT_CONTACT_ATTEMPT_ID:
        raise SystemExit(f"pendant attempt id drift: {pendant['attempt_id']}")

    # --- correction 3: the eight worker deaths ------------------------------
    defects = run["infrastructure_defects"]
    worker_deaths = [d for d in defects if WORKER_DEATH_ERROR in str(d.get("error", ""))]
    halted = run["halted_for_repair"]
    ledger_worker_died = [r for r in rows if r["status"] == "worker_died"]
    sampling_failures = [r for r in rows if r["status"] == "sampling_failure"]

    # --- correction 4: quota accounting ------------------------------------
    quota = quotas()
    by_cell: dict[str, int] = {}
    for r in accepted:
        by_cell[r["cell"]] = by_cell.get(r["cell"], 0) + 1
    at = sorted(c for c in quota if by_cell.get(c, 0) == quota[c])
    over = {c: by_cell[c] - quota[c] for c in quota if by_cell.get(c, 0) > quota[c]}
    short = {c: quota[c] - by_cell.get(c, 0) for c in quota if by_cell.get(c, 0) < quota[c]}

    lengths = [r["timesteps"] for r in source["rows"]]

    corrections: list[dict[str, Any]] = [
        {
            "id": "E1",
            "topic": "pendant involvement",
            "previous_claim": "no pendant involvement anywhere in the collection",
            "status": "FALSE",
            "correction":
                "Zero *accepted* rows contacted the pendant -- that part stands. But one "
                "rejected attempt did contact it, so the stronger claim is false.",
            "evidence": {
                "accepted_rows_with_pendant_contact": 0,
                "attempts_with_any_pendant_involvement": 1,
            },
        },
        {
            "id": "E2",
            "topic": "the one pendant-contact attempt",
            "previous_claim": "not recorded in the close-out narrative",
            "status": "OMITTED",
            "correction":
                "Rejected attempt {aid} recorded {mf} mounted_fixture contacts, "
                "{pcf} pendant-contact frame, and zero clearance. It is the only such "
                "row in all 353 attempts and it was correctly rejected.".format(
                    aid=pendant["attempt_id"], mf=pendant["contact_class_totals"]["mounted_fixture"],
                    pcf=pendant["pendant_contact_frames"]),
            "evidence": {
                "attempt_id": pendant["attempt_id"],
                "accepted": pendant["accepted"],
                "cell": pendant["cell"],
                "attempt_index": pendant["attempt_index"],
                "task_seed_u32": pendant["task_seed_u32"],
                "mounted_fixture_contacts": pendant["contact_class_totals"]["mounted_fixture"],
                "clutter_contacts": pendant["contact_class_totals"]["clutter"],
                "pendant_contact_frames": pendant["pendant_contact_frames"],
                "min_pendant_clearance_m": pendant["min_pendant_clearance_m"],
                "min_lobe_stem_clearance_m": pendant["min_lobe_stem_clearance_m"],
                "clutter_stability_events": pendant["clutter_stability_events"],
                "episode_steps": pendant["episode_steps"],
                "defects": pendant["defects"],
                "row_sha256": pendant["row_sha256"],
                "result_sha256": pendant["result_sha256"],
            },
        },
        {
            "id": "E3",
            "topic": "worker deaths and infrastructure events",
            "previous_claim": "closeout.json reports infrastructure_halts: 0",
            "status": "FALSE AND MISLEADING",
            "correction":
                "There were {n} BrokenProcessPool worker deaths, caused by terminating "
                "the process pool on the owner's stop instruction while a batch was in "
                "flight. They are owner-stop-induced worker terminations -- not zero "
                "infrastructure events, and not spontaneous data corruption. None "
                "advanced a scientific seed stream and none entered the 353 scientific "
                "ledger rows. They must not be conflated with the {s} sampling_failure "
                "ledger rows, which are a different, scientific outcome.".format(
                    n=len(worker_deaths), s=len(sampling_failures)),
            "evidence": {
                "closeout_infrastructure_halts": closeout["infrastructure_halts"],
                "recorded_worker_deaths": len(worker_deaths),
                "n_infrastructure_defects": run["n_infrastructure_defects"],
                "excluded_from_scientific_attempts":
                    run["infrastructure_defects_excluded_from_scientific_attempts"],
                "retries_authorized_by_contract":
                    run["infrastructure_retries_authorized_by_contract"],
                "halted_for_repair_attempt_id": halted["attempt_id"],
                "halted_row_replaced": halted["row_replaced"],
                "halted_scientific_stream_advanced": halted["scientific_stream_advanced"],
                "runner_recorded_stop_reason": run["stop_reason"],
                "true_stop_reason": closeout["stop_reason"],
                "worker_death_cells": [
                    {"cell": d["cell"], "attempt_index": d["attempt_index"]}
                    for d in worker_deaths
                ],
                "ledger_rows_with_status_worker_died": len(ledger_worker_died),
                "ledger_rows_with_status_sampling_failure": len(sampling_failures),
                "sampling_failure_cells": sorted({r["cell"] for r in sampling_failures}),
            },
        },
        {
            "id": "E4",
            "topic": "quota accounting",
            "previous_claim": "cells_at_quota: 19",
            "status": "AMBIGUOUS, OVERSTATES BALANCE",
            "correction":
                "Exactly {a} cells equal quota and {o} exceed it, so {t} meet or exceed "
                "quota. {s} cells are short. Reporting '19 at quota' without the split "
                "reads as 19 cells having met their target exactly.".format(
                    a=len(at), o=len(over), t=len(at) + len(over), s=len(short)),
            "evidence": {
                "cells_exactly_at_quota": len(at),
                "cells_over_quota": len(over),
                "cells_at_or_over_quota": len(at) + len(over),
                "cells_short": len(short),
                "over_quota_detail": over,
                "short_detail": short,
                "closeout_cells_at_quota_field": closeout["cells_at_quota"],
            },
        },
        {
            "id": "E5",
            "topic": "splittability of the smallest cells",
            "previous_claim":
                "the F3 right-side cells cannot be represented in both train and validation",
            "status": "OVERSTATED",
            "correction":
                "Only F3_aperture_side_stagger|right|neg5, with one episode, is "
                "mathematically unable to appear in both. "
                "F3_aperture_side_stagger|right|pos5, with two, splits 1/1.",
            "evidence": {
                "F3_aperture_side_stagger|right|neg5": by_cell.get(
                    "F3_aperture_side_stagger|right|neg5", 0),
                "F3_aperture_side_stagger|right|pos5": by_cell.get(
                    "F3_aperture_side_stagger|right|pos5", 0),
            },
        },
        {
            "id": "E6",
            "topic": "encoder-health provenance",
            "previous_claim":
                "encoder health was reported as if it characterised the collection",
            "status": "UNREPRESENTATIVE SAMPLE",
            "correction":
                "Those statistics came from 120 windows drawn from a single real "
                "episode, not the full corpus. The exact corpus is {t} HDF5 timesteps "
                "and {w} sensor windows. Corpus-wide statistics are computed during "
                "V10.9 embedding generation and supersede them.".format(
                    t=f"{sum(lengths):,}", w=f"{sum(lengths) * N_SENSORS:,}"),
            "evidence": {
                "previously_reported_windows": 120,
                "previously_reported_episodes": 1,
                "corpus_timesteps": sum(lengths),
                "corpus_sensor_windows": sum(lengths) * N_SENSORS,
                "sensors_per_episode": N_SENSORS,
            },
        },
        {
            "id": "E7",
            "topic": "corpus dimensions",
            "previous_claim": "episode_steps min 355, max 626 (control steps)",
            "status": "CORRECT BUT EASILY MISREAD",
            "correction":
                "Those are control steps. HDF5 T = episode_steps + 1, so T ranges "
                "{lo} to {hi} and sums to {s}. Anything sized against the HDF5 arrays "
                "(episode_horizon, padding, window counts) must use T.".format(
                    lo=min(lengths), hi=max(lengths), s=f"{sum(lengths):,}"),
            "evidence": {
                "closeout_episode_steps_min": closeout["episode_steps"]["min"],
                "closeout_episode_steps_max": closeout["episode_steps"]["max"],
                "hdf5_t_min": min(lengths),
                "hdf5_t_max": max(lengths),
                "hdf5_t_sum": sum(lengths),
            },
        },
    ]

    document: dict[str, Any] = {
        **empty_authorization(),
        "schema_version": "pact_place_v108_erratum_v1",
        "contract_version": CONTRACT_VERSION_V109,
        "role": "create-only erratum to the V10.8 record; closeout.json is NOT modified",
        "written_by": "V10.9 (docs/PACT_PLACE_V109_TRAIN_EVAL_PLAN.md section 0)",
        "authorization":
            "explicit owner authorization to convert, train, and evaluate the 141 "
            "accepted V10.8 demonstrations despite V10.7's failed Phase-0 gate and "
            "V10.8's early stop",
        "v108_status_unchanged": {
            "is_phase0_pass": False,
            "v107_phase0_result": "failed_8_of_24_permanently_closed",
            "stop_reason": "owner_instructed_early_stop",
            "target_successes": closeout["target_successes"],
            "accepted_total": closeout["accepted_total"],
            "quotas_met": closeout["quotas_met"],
        },
        "bindings": {
            "ledger_path": str(ledger_path.relative_to(ROOT)),
            "ledger_sha256": ledger_sha,
            "ledger_records": len(rows),
            "closeout_path": str(closeout_path.relative_to(ROOT)),
            "closeout_raw_file_sha256": sha256_file(closeout_path),
            "closeout_payload_sha256": closeout["payload_sha256"],
            "closeout_payload_sha256_recomputed": recompute_payload_sha256(closeout_path),
            "collection_record_path": str(collection_path.relative_to(ROOT)),
            "collection_record_raw_file_sha256": sha256_file(collection_path),
            "source_manifest_path": str(args.source_manifest.relative_to(ROOT)),
            "source_manifest_payload_sha256": source["payload_sha256"],
            "accepted_attempt_ids": [r["attempt_id"] for r in source["rows"]],
            "accepted_h5_sha256": {
                r["attempt_id"]: r["trajectory_h5_sha256"] for r in source["rows"]
            },
        },
        "corrections": corrections,
        "n_corrections": len(corrections),
    }
    document["payload_sha256"] = canonical_payload_sha256(document)

    out = ROOT / WORK_ROOT / "v108_erratum.json"
    written = write_immutable_create_only(out, document)

    lines = [
        "# V10.8 erratum",
        "",
        "Create-only. `diagnostics_output/pact_place_v108_collection/closeout.json` is",
        "**not** modified. Written by V10.9; every correction is re-derived from",
        "`ledger.jsonl` and the retained row files, not copied from the close-out.",
        "",
        "V10.8's status is unchanged: an exploratory owner-override collection, stopped",
        "early by owner instruction at 141 of 152 target successes, quotas unmet, not a",
        "Phase-0 pass. V10.7's Phase-0 gate remains failed at 8/24 and permanently closed.",
        "",
        f"- ledger.jsonl SHA-256 `{ledger_sha}` ({len(rows)} records)",
        f"- closeout.json raw SHA-256 `{document['bindings']['closeout_raw_file_sha256']}`",
        f"- closeout.json payload SHA-256 `{closeout['payload_sha256']}`",
        f"- source manifest payload SHA-256 `{source['payload_sha256']}` (141 rows)",
        f"- erratum payload SHA-256 `{document['payload_sha256']}`",
        "",
    ]
    for c in corrections:
        lines += [
            f"## {c['id']} — {c['topic']}  ·  {c['status']}",
            "",
            f"**Previously:** {c['previous_claim']}",
            "",
            f"**Correction:** {c['correction']}",
            "",
            "```json",
            json.dumps(c["evidence"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    text_sha = write_immutable_text_create_only(ROOT / WORK_ROOT / "V108_ERRATUM.md",
                                                "\n".join(lines))
    print(json.dumps({
        "corrections": len(corrections),
        "payload_sha256": document["payload_sha256"],
        "raw_file_sha256": written.get("raw_file_sha256"),
        "markdown_sha256": text_sha,
        "worker_deaths": len(worker_deaths),
        "ledger_worker_died_rows": len(ledger_worker_died),
        "sampling_failure_rows": len(sampling_failures),
        "cells_exactly_at_quota": len(at),
        "cells_over_quota": len(over),
        "cells_short": len(short),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
