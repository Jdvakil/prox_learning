#!/usr/bin/env python3
"""Revalidate the eight smoke-reference rows against the full 160-row run.

Handoff step 9. The comparison itself is delegated to the already-committed
``scripts/hybrid_obstacle_manifest_v2_audit.py``, so the frozen tolerances,
exact-match field list and discrete-event field list from the seeding audit are
used verbatim. No tolerance is created or relaxed here.

That audit compares two run directories row-for-row and requires the two episode
ID sets to be equal for ``all_invariant``. The full run holds 160 rows and the
retained smoke reference holds 8, so this driver first materialises a read-only
*view* of the full run containing symlinks to only the eight smoke row
directories, then invokes the committed audit on (reference, view).

Nothing in the full collection or the retained smoke artifacts is modified.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Locate the repo whether this file sits in scripts/ or is run from elsewhere."""
    marker = Path("configs") / "hybrid_obstacle_candidate_manifest_v2.json"
    for cand in (Path(__file__).resolve().parents[1], Path.cwd(), Path("/root/prox_learning_hybrid_safety")):
        if (cand / marker).is_file():
            return cand
    raise SystemExit("cannot locate the prox_learning_hybrid_safety repo root")


ROOT = _repo_root()
COMMITTED_AUDIT = ROOT / "scripts" / "hybrid_obstacle_manifest_v2_audit.py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path, help="full 160-row run dir")
    ap.add_argument("--reference", required=True, type=Path, help="validated 4-worker smoke run dir")
    ap.add_argument("--smoke-subset", required=True, type=Path)
    ap.add_argument("--view", required=True, type=Path, help="scratch dir for the 8-row view")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--decision-json", type=Path, default=None,
                    help="seeding final_decision.json, for committed smoke hashes")
    args = ap.parse_args()

    subset = json.loads(args.smoke_subset.read_text())
    episode_ids = list(subset["episode_ids"])
    if len(episode_ids) != 8:
        raise SystemExit(f"expected 8 smoke rows, subset declares {len(episode_ids)}")

    ref_rows = args.reference / "rows"
    retained = {eid: (ref_rows / eid / "trajectory.h5").is_file() for eid in episode_ids}
    reference_available = all(retained.values())

    # ---- build the read-only 8-row view of the full run --------------------
    view_rows = args.view / "rows"
    if args.view.exists():
        raise SystemExit(f"view dir already exists: {args.view}")
    view_rows.mkdir(parents=True)
    missing = []
    for eid in episode_ids:
        src = args.run / "rows" / eid
        if not (src / "outcome.json").is_file():
            missing.append(eid)
            continue
        (view_rows / eid).symlink_to(src.resolve())
    summary = args.run / "collection_summary.json"
    if summary.is_file():
        (args.view / "collection_summary.json").symlink_to(summary.resolve())
    if missing:
        raise SystemExit(f"smoke rows absent from the full run: {missing}")

    # ---- delegate to the committed audit ----------------------------------
    cmd = [
        sys.executable,
        str(COMMITTED_AUDIT),
        "--run", f"SMOKEREF={args.reference}",
        "--run", f"FULL8={args.view}",
        "--expected-rows", "8",
        "--out", str(args.out),
    ]
    print("committed audit:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)

    if not args.out.is_file():
        raise SystemExit("committed audit produced no report")
    report = json.loads(args.out.read_text())

    comparison = next(
        (c for c in report["comparisons"] if set(c["pair"].split("_vs_")) == {"SMOKEREF", "FULL8"}),
        None,
    )
    if comparison is None:
        raise SystemExit("no SMOKEREF/FULL8 comparison in the report")

    # ---- committed smoke hashes from the seeding decision ------------------
    committed_hashes = None
    if args.decision_json and args.decision_json.is_file():
        decision = json.loads(args.decision_json.read_text())

        def find_hashes(obj, path=""):
            out = {}
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v):
                        out[f"{path}/{k}"] = v
                    else:
                        out.update(find_hashes(v, f"{path}/{k}"))
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    out.update(find_hashes(v, f"{path}[{i}]"))
            return out

        allh = find_hashes(decision)
        committed_hashes = {k: v for k, v in allh.items() if any(e[:16] in k for e in episode_ids)}
        committed_hashes["_total_hashes_in_decision_json"] = len(allh)

    result = {
        "schema": "hybrid_obstacle_smoke8_reference_revalidation",
        "reference_run": str(args.reference),
        "reference_h5s_retained": reference_available,
        "reference_h5_present_per_row": retained,
        "full_run": str(args.run),
        "view": str(args.view),
        "smoke_subset_sha256": subset["subset_sha256"],
        "episode_ids": episode_ids,
        "tolerance_source": "frozen in scripts/hybrid_obstacle_manifest_v2_audit.py (seeding audit)",
        "frozen_tolerances": report["frozen_tolerances"],
        "exact_match_fields": report["exact_match_fields"],
        "discrete_event_fields": report["discrete_event_fields"],
        "episodes_compared": comparison["episodes_compared"],
        "episode_id_sets_match": comparison["episode_id_sets_match"],
        "all_invariant": comparison["all_invariant"],
        "all_bit_identical": comparison["all_bit_identical"],
        "per_episode": [
            {
                "episode_id": e["episode_id"],
                "candidate_index": e["candidate_index"],
                "invariant": e["invariant"],
                "bit_identical": e["bit_identical"],
                "h5_schema_match": e["h5_schema_match"],
                "sensor_order_match": e["sensor_order_match"],
                "field_mismatches": e["field_mismatches"],
                "discrete_event_mismatches": e["discrete_event_mismatches"],
                "worker_ids": e["worker_ids"],
            }
            for e in comparison["episodes"]
        ],
        "committed_smoke_hashes_from_decision_json": committed_hashes,
        "committed_audit_exit_code": proc.returncode,
        "committed_audit_report": str(args.out),
    }
    result["ok"] = bool(result["all_invariant"])

    out_summary = args.out.with_name(args.out.stem + "_summary.json")
    out_summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"\nreference H5s retained  {reference_available}")
    print(f"episodes compared       {result['episodes_compared']}/8")
    print(f"episode id sets match   {result['episode_id_sets_match']}")
    print(f"all invariant           {result['all_invariant']}")
    print(f"all bit-identical       {result['all_bit_identical']}")
    for e in result["per_episode"]:
        flag = "bit-identical" if e["bit_identical"] else ("invariant" if e["invariant"] else "MISMATCH")
        print(f"  [{e['candidate_index']:3d}] {e['episode_id'][:12]} {flag}"
              f" workers {e['worker_ids']['left']}->{e['worker_ids']['right']}")
        for m in e["field_mismatches"]:
            print(f"        field {m['field']}: {str(m['left'])[:60]} != {str(m['right'])[:60]}")
    print(f"wrote {out_summary}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
