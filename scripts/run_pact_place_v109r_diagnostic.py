#!/usr/bin/env python3
"""V10.9R step 6: F2 development diagnostic, original decoder vs repaired.

Ten F2 rows, both arms, both decoders: 40 rollouts. Trajectories and raw chunks
are retained so the comparison rests on recorded evidence rather than on the
summary each decoder reports about itself.

These ten rows are a **development** diagnostic. They were already inspected, so
they cannot serve as evidence for the repair; the frozen evaluation in step 7
runs on seeds disjoint from these and from everything before them.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v109_contract import (  # noqa: E402
    ENCODER_PATH,
    ENCODER_SHA256,
    EVAL_ROOT as V109_EVAL_ROOT,
    TRAINING_ROOT,
    canonical_payload_sha256,
    empty_authorization,
    sha256_file,
    sha256_payload,
    write_immutable_create_only,
)
from pact_place_v109_eval_contract import EVAL_NUM_QUERIES, load_manifest  # noqa: E402
from pact_place_v109r_contract import CONTRACT_VERSION_V109R, DIAGNOSTIC_ROOT  # noqa: E402

EVALUATOR = ROOT / "submodules" / "act" / "eval_pact_place_v109r_row.py"
ARMS = ("ACT", "PACT")
DECODERS = ("legacy", "event")
F2 = "F2_outer_panel_stagger"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def arm_binding(arm: str) -> dict[str, Any]:
    directory = Path(TRAINING_ROOT) / f"{arm.lower()}_seed3101"
    checkpoint, stats = directory / "policy_best.ckpt", directory / "dataset_stats.pkl"
    return {"arm": arm, "checkpoint_dir": str(directory),
            "checkpoint_sha256": sha256_file(checkpoint),
            "stats_sha256": sha256_file(stats), "checkpoint_seed": 3101}


def run_one(payload: dict[str, Any]) -> dict[str, Any]:
    row_dir = Path(payload["row_dir"])
    if (row_dir / "result.json").is_file():
        return {**payload["schedule"], "status": "already_present",
                "returncode": 0, "elapsed_s": 0.0}
    row_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    env.setdefault("PACT_CONTACT_AUDIT_SUMMARY_ONLY", "1")
    env["PACT_V109R_DECODER"] = payload["schedule"]["decoder"]
    env["PACT_V109_TRAJECTORY_H5_ONLY"] = "1"
    env.pop("DISPLAY", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "submodules" / "molmospaces"), str(ROOT / "scripts"),
         env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    started = time.monotonic()
    with (row_dir / "rollout.log").open("w") as stream:
        completed = subprocess.run(payload["command"], cwd=ROOT / "submodules" / "act",
                                   env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   check=False)
    return {**payload["schedule"], "returncode": completed.returncode,
            "elapsed_s": round(time.monotonic() - started, 1),
            "status": "complete" if (row_dir / "result.json").is_file() else "no_result",
            "row_dir": str(row_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-root", type=str, default=DIAGNOSTIC_ROOT)
    args = parser.parse_args()

    manifest_path = ROOT / V109_EVAL_ROOT / "eval_manifest.json"
    manifest = load_manifest(manifest_path)
    rows = [r for r in manifest["rows"] if r["family_id"] == F2]
    if len(rows) != 10:
        raise SystemExit(f"expected 10 F2 rows, found {len(rows)}")
    bindings = {arm: arm_binding(arm) for arm in ARMS}
    out_root = ROOT / args.out_root

    payloads = []
    for decoder in DECODERS:
        for arm in ARMS:
            for row in rows:
                schedule = {
                    "arm": arm, "decoder": decoder,
                    "episode_id": row["episode_id"],
                    "candidate_index": int(row["candidate_index"]),
                    "cell": row["cell"], "row_sha256": row["row_sha256"],
                    "checkpoint_sha256": bindings[arm]["checkpoint_sha256"],
                    "stats_sha256": bindings[arm]["stats_sha256"],
                    "rollout_id": f"v109r_diag_{decoder}_{arm.lower()}_"
                                  f"{int(row['candidate_index']):03d}",
                }
                schedule["schedule_row_sha256"] = sha256_payload(schedule)
                row_dir = (out_root / decoder / arm.lower() /
                           f"{int(row['candidate_index']):03d}_{row['episode_id'][:16]}")
                command = [
                    sys.executable, str(EVALUATOR),
                    "--arm", arm, "--episode-id", row["episode_id"],
                    "--manifest", str(manifest_path.resolve()),
                    "--checkpoint-dir", bindings[arm]["checkpoint_dir"],
                    "--checkpoint-sha256", bindings[arm]["checkpoint_sha256"],
                    "--checkpoint-seed", "3101",
                    "--schedule-row-sha256", schedule["schedule_row_sha256"],
                    "--rollout-id", schedule["rollout_id"],
                    "--stats-sha256", bindings[arm]["stats_sha256"],
                    "--output-dir", str(row_dir.resolve()),
                    "--save-video",
                ]
                if arm == "PACT":
                    command += ["--surface-encoder", ENCODER_PATH,
                                "--surface-encoder-sha256", ENCODER_SHA256]
                payloads.append({"schedule": schedule, "row_dir": str(row_dir),
                                 "command": command})

    print(f"F2 diagnostic: {len(payloads)} rollouts on {args.workers} workers",
          flush=True)
    started_utc, monotonic = utc_now(), time.monotonic()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, p): p for p in payloads}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                results.append(future.result())
            except BaseException as error:  # noqa: BLE001
                p = futures[future]
                results.append({**p["schedule"], "status": "worker_died",
                                "returncode": -1,
                                "error": f"{type(error).__name__}: {error}"[:300],
                                "elapsed_s": 0.0})
            elapsed = time.monotonic() - monotonic
            print(f"  {done}/{len(payloads)}  {elapsed/60:.1f} min, "
                  f"eta {(len(payloads)-done)*elapsed/done/60:.1f} min", flush=True)

    failures = [r for r in results if r["status"] not in ("complete", "already_present")]
    document = {
        **empty_authorization(),
        "schema_version": "pact_place_v109r_diagnostic_run_v1",
        "contract_version": CONTRACT_VERSION_V109R,
        "role": "F2 development diagnostic: original vs repaired decoder",
        "is_development_only": True,
        "not_evidence_for_the_repair":
            "these ten rows were already inspected; the frozen evaluation uses "
            "disjoint seeds",
        "manifest_sha256": manifest["manifest_sha256"],
        "arms": bindings, "decoders": list(DECODERS),
        "rollouts": len(results),
        "complete": sum(1 for r in results if r["status"] in ("complete", "already_present")),
        "failures": failures,
        "started_utc": started_utc, "finished_utc": utc_now(),
        "elapsed_hours": round((time.monotonic() - monotonic) / 3600, 3),
        "results": sorted(results, key=lambda r: (r["decoder"], r["arm"],
                                                  r["candidate_index"])),
    }
    document["payload_sha256"] = canonical_payload_sha256(document)
    write_immutable_create_only(out_root / "diagnostic_run.json", document)
    print(json.dumps({"complete": document["complete"], "of": len(results),
                      "failures": len(failures),
                      "elapsed_hours": document["elapsed_hours"]}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
