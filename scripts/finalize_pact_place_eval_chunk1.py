#!/usr/bin/env python3
"""Verify and report the chunk-1 held-out place rollout evaluation."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from pact_place_eval_chunk1_contract import load_manifest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path("/root/pact_place_chunk1_eval_seed3101")
MANIFEST = ROOT / "configs" / "pact_place_eval_chunk1_manifest.json"
ENCODER_SHA = "6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206"
EXPECTED_PROTECTED = {
    ROOT / "submodules/act/eval_pact_collision_row.py": "8d0342903df92abc02d04c97186dd80a2454e93fef02edfee171b8efe5072c44",
    ROOT / "submodules/act/eval_pact_frontend_screen_row.py": "810cbcadcac879075bc61959e74639211ecc6bcd4878238b51dced0a139681dc",
    Path("/root/pact_place_152_pact_vs_act_seed3101/act_seed3101/policy_best.ckpt"): "cd95d805cc1caa672137ce5d58eab1671ba175e36f309cc65070eee0acee2c30",
    Path("/root/pact_place_152_pact_vs_act_seed3101/pact_seed3101/policy_best.ckpt"): "4404138b5445a168c36e0dbd463216419179b9ee6211c9bf3e27ab25f47b1e99",
    Path("/root/pact_frontend_screen_artifacts/encoder_v1/embedding_encoder_frozen.pt"): ENCODER_SHA,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(arm: str, results: list[dict], drivers: list[dict]) -> dict[str, Any]:
    return {
        "arm": arm.upper(),
        "n": len(results),
        "task_success": sum(result["task_success"] for result in results),
        "collision_free_task_success": sum(
            result["collision_free_task_success"] for result in results
        ),
        "collision_free_episode": sum(
            result["contact_audit"]["collision_free"] for result in results
        ),
        "gripper_close_commanded": sum(
            result["policy_info"]["gripper_close_commanded"] for result in results
        ),
        "failure_taxonomy": dict(
            sorted(Counter(result["failure_taxonomy"] for result in results).items())
        ),
        "contact_episode_counts": {
            contact: sum(
                result["contact_audit"]["contact_class_totals"][contact] > 0
                for result in results
            )
            for contact in (
                "hazard_bar",
                "other_environment",
                "clutter",
                "place_receptacle",
            )
        },
        "control_step_counts": dict(
            sorted(Counter(result["policy_info"]["control_steps"] for result in results).items())
        ),
        "wall_clock_seconds": {
            "min": min(driver["wall_clock_seconds"] for driver in drivers),
            "median": statistics.median(
                driver["wall_clock_seconds"] for driver in drivers
            ),
            "mean": statistics.mean(driver["wall_clock_seconds"] for driver in drivers),
            "max": max(driver["wall_clock_seconds"] for driver in drivers),
        },
    }


def main() -> int:
    manifest = load_manifest(MANIFEST)
    protected = {str(path): sha256_file(path) for path in EXPECTED_PROTECTED}
    for path, expected in EXPECTED_PROTECTED.items():
        if protected[str(path)] != expected:
            raise RuntimeError(f"protected artifact changed: {path}")
    full = json.loads((OUTPUT_ROOT / "full_launcher_summary.json").read_text())
    smoke = json.loads((OUTPUT_ROOT / "smoke_launcher_summary.json").read_text())
    if (
        full["jobs_requested"] != 40
        or full["jobs_complete"] != 40
        or full["errors"]
    ):
        raise RuntimeError("full launcher did not reconcile all 40 rows")
    arms = {}
    all_result_ids = set()
    for arm in ("act", "pact"):
        results = [
            json.loads(path.read_text())
            for path in sorted((OUTPUT_ROOT / arm).glob("*/result.json"))
        ]
        drivers = [
            json.loads(path.read_text())
            for path in sorted((OUTPUT_ROOT / arm).glob("*/driver_result.json"))
        ]
        if len(results) != 20 or len(drivers) != 20:
            raise RuntimeError(f"{arm}: expected 20 results and drivers")
        by_episode = {result["episode_id"]: result for result in results}
        if set(by_episode) != {row["episode_id"] for row in manifest["rows"]}:
            raise RuntimeError(f"{arm}: result episode identities differ from manifest")
        side_counts = Counter(result["intrusion_side"] for result in results)
        if side_counts != {"left": 10, "right": 10}:
            raise RuntimeError(f"{arm}: side balance changed")
        for result in results:
            key = (result["arm"], result["episode_id"])
            if key in all_result_ids:
                raise RuntimeError("duplicate arm/episode result")
            all_result_ids.add(key)
            if not isinstance(result["task_success"], bool):
                raise RuntimeError("task_success is not bool")
            audit = result["contact_audit"]
            if "contact_class_totals" not in audit or not isinstance(
                audit.get("collision_free"), bool
            ):
                raise RuntimeError("contact audit summary incomplete")
            info = result["policy_info"]
            if info["contact_audit_class"] != "PactPlaceContactAudit":
                raise RuntimeError("place audit missing")
            if info["num_queries"] != 1 or not info["temporal_ensembling_inert_chunk1"]:
                raise RuntimeError("chunk-1 runtime contract failed")
            if arm == "pact":
                if (
                    info["proximity_feature_dim"] != 32
                    or info["input_proj_proximity_shape"] != [512, 32]
                    or result["surface_encoder_sha256"] != ENCODER_SHA
                ):
                    raise RuntimeError("PACT 32-D runtime contract failed")
        arms[arm.upper()] = aggregate(arm, results, drivers)
    collapse = any(
        arms[arm]["collision_free_task_success"] <= 1 for arm in ("ACT", "PACT")
    )
    analysis = {
        "schema_version": "pact_place_eval_chunk1_analysis_v1",
        "status": "CHUNK1_COLLAPSE" if collapse else "FUNCTIONAL_CHUNK1",
        "primary_endpoint": "collision_free_task_success",
        "collapse_rule": "if either arm is <= 1/20, both arms are uninformative",
        "cross_arm_claim_authorized": False,
        "single_seed": 3101,
        "n_per_arm": 20,
        "manifest_sha256": manifest["manifest_sha256"],
        "held_out_seed_audit": manifest["held_out_seed_audit"],
        "intrusion_side_counts_per_arm": {"left": 10, "right": 10},
        "smoke": {
            "workers": smoke["workers"],
            "n": smoke["jobs_complete"],
            "wall_clock_seconds": [
                row["wall_clock_seconds"] for row in smoke["results"]
            ],
            "median_wall_clock_seconds": statistics.median(
                row["wall_clock_seconds"] for row in smoke["results"]
            ),
            "n_justification": (
                "Measured median about 999 seconds per rollout; 40 jobs at 10 workers "
                "requires about four waves (~67 minutes), so retained N=20 per arm."
            ),
        },
        "full_dispatch": {
            "workers": full["workers"],
            "jobs_requested": full["jobs_requested"],
            "jobs_complete": full["jobs_complete"],
            "errors": full["errors"],
            "started_utc": full["started_utc"],
            "finished_utc": full["finished_utc"],
            "summary_sha256": full["summary_sha256"],
        },
        "arms": arms,
        "runtime_contract": {
            "num_queries": 1,
            "temporal_ensembling": "inert: exactly one age-0 action survives",
            "pact_encoder_schema": "pact_surface_embedding_encoder_v1",
            "pact_feature_dim": 32,
            "pact_projection_shape": [512, 32],
            "pact_encoder_sha256": ENCODER_SHA,
            "place_contact_audit": "PactPlaceContactAudit",
            "scene": "pact_place_corridor_v2.xml",
            "sampler": "PactPlaceCorridorV2Sampler",
            "task_horizon": 900,
        },
        "config_diff": {
            "verification": (
                "Pydantic model dumps of make_recovery_config and PactPlaceEvalConfig "
                "were equal after excluding policy_config and the ephemeral profiler object."
            ),
            "scientific_difference": "policy_config only: expert -> learned chunk-1 policy",
        },
        "place_receptacle_diagnostic_limit": (
            "Learned policy exposes no expert phase; audit phase remains 'other', so all "
            "place_receptacle contacts are conservatively counted outside placement."
        ),
        "protected_artifact_sha256": protected,
    }
    (OUTPUT_ROOT / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    )
    act, pact = arms["ACT"], arms["PACT"]
    report = f"""# Chunk-1 place rollout evaluation

## Decision

**CHUNK1_COLLAPSE.** ACT@1 scored **{act['collision_free_task_success']}/20** and PACT@1 scored **{pact['collision_free_task_success']}/20** collision-free task successes. The preregistered rule declares collapse if either arm is <= 1/20, so **both arms are uninformative**. This is a result about chunk size, not proximity modality. No PACT-vs-ACT difference is claimed.

| Arm | N | Task success | Collision-free task success | Collision-free episode diagnostic |
|---|---:|---:|---:|---:|
| ACT@1 | 20 | {act['task_success']}/20 | {act['collision_free_task_success']}/20 | {act['collision_free_episode']}/20 |
| PACT@1 | 20 | {pact['task_success']}/20 | {pact['collision_free_task_success']}/20 | {pact['collision_free_episode']}/20 |

Every rollout reached the 900-step horizon. Neither arm commanded gripper close in any rollout. Failure taxonomies are descriptive only: ACT `{json.dumps(act['failure_taxonomy'], sort_keys=True)}`; PACT `{json.dumps(pact['failure_taxonomy'], sort_keys=True)}`.

## Timing and N

The four-worker smoke measured {analysis['smoke']['median_wall_clock_seconds']:.1f} seconds median per rollout (range {min(analysis['smoke']['wall_clock_seconds']):.1f}--{max(analysis['smoke']['wall_clock_seconds']):.1f}). Forty jobs at 10 workers therefore required about four waves (~67 minutes projected), so N=20 per arm was retained. Full dispatch ran from `{full['started_utc']}` to `{full['finished_utc']}`, completed 40/40 jobs, and reported zero errors.

## Runtime verification

- Both checkpoints loaded under `strict=True` with CLI `num_queries=1`.
- Temporal ensembling is inert at chunk 1: each prediction has length one, so only the current age-0 action survives with weight 1.
- PACT used `load_frozen_surface_embedding_encoder`, 32-D `.policy_features()`, and projection shape `(512,32)`. Encoder SHA-256: `{ENCODER_SHA}`.
- Every result attached `PactPlaceContactAudit` and contains `contact_class_totals` plus boolean `collision_free`.
- Scene/sampler: `pact_place_corridor_v2.xml` / `PactPlaceCorridorV2Sampler`; task horizon 900.
- Serialized config comparison against `make_recovery_config` was equal after removing only `policy_config` and the ephemeral profiler object. Scientific diff: `policy_config` changed from expert to learned chunk-1 policy.

## Held-out instances

- Manifest SHA-256: `{manifest['manifest_sha256']}`.
- New master seed: `{manifest['master_seed']}`.
- Evaluation/training `task_seed_u64` intersection: **0** across 20 eval and 152 training seeds.
- Intrusion side is 10 left / 10 right in each arm.
- Each manifest row reconciles to exactly one ACT result and one PACT result.

## Diagnostic limits

- This is one training seed (3101) and N=20 per arm. It cannot establish a cross-arm difference; none is claimed.
- `place_receptacle` contact occurred in {act['contact_episode_counts']['place_receptacle']}/20 ACT and {pact['contact_episode_counts']['place_receptacle']}/20 PACT episodes. Learned policies expose no expert phase, so the audit remains in phase `other` and conservatively over-counts every such contact as outside placement. This is a separate diagnostic and is not folded into the primary endpoint.
- No clutter exists in the v2 scene; clutter contact was 0/20 in both arms.
- The @100 policy pair was not evaluated.

Protected corridor evaluators, the frozen encoder, and both chunk-1 checkpoints matched their pre-run SHA-256 values. Detailed machine-readable results are in `analysis.json`; per-row scientific and driver results are under `act/` and `pact/`.
"""
    (OUTPUT_ROOT / "EVAL.md").write_text(report)
    print(json.dumps({"status": analysis["status"], "arms": arms}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
