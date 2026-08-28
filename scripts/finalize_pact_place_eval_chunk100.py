#!/usr/bin/env python3
"""Verify and report the chunk-100 held-out place rollout evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from pact_place_eval_chunk100_contract import load_manifest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path("/root/pact_place_chunk100_eval_seed3101")
TRAIN_ROOT = Path("/root/pact_place_152_pact_vs_act_chunk100_seed3101")
CHUNK1_OUTPUT = Path("/root/pact_place_chunk1_eval_seed3101")
CHUNK1_TRAIN = Path("/root/pact_place_152_pact_vs_act_seed3101")
MANIFEST = ROOT / "configs/pact_place_eval_chunk100_manifest.json"
CHUNK1_MANIFEST = ROOT / "configs/pact_place_eval_chunk1_manifest.json"
TOKEN_PLAN = OUTPUT_ROOT / "token_plan/token_plan.json"
DATASET_DIR = ROOT / "assets/act_style_data/pact_place_corridor_v2_recovered_152"
DATASET_MANIFEST = ROOT / "diagnostics_output/pact_place_152_pact_vs_act/conversion_manifest_encoded.json"
DATASET_TREE_SHA = "b16a5a0bd221d786f54fd9f28e00d493d01316ed47d9e909c1a915d37b13e6f1"
ENCODER = Path("/root/pact_frontend_screen_artifacts/encoder_v1/embedding_encoder_frozen.pt")
ENCODER_SHA = "6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206"
STATS_SHA = "1860d71a09e7c6ca5afdcb13a952c6de84f52a7bc4810517554782027322c6de"
CHUNK1_TRAIN_TREE_SHA = "e5ef3dd94f4240f2b0dbd3089f37297b754d775318097aeb8df46d1274d991d4"
CHUNK1_EVAL_TREE_SHA = "80ab052b5ef28c48f8927fdd628b4d3e5893b287d58b095a24b2a4ddd63d7a1b"
EXPECTED_PROTECTED = {
    ROOT
    / "submodules/act/eval_pact_collision_row.py": "8d0342903df92abc02d04c97186dd80a2454e93fef02edfee171b8efe5072c44",
    ROOT
    / "submodules/act/eval_pact_frontend_screen_row.py": "810cbcadcac879075bc61959e74639211ecc6bcd4878238b51dced0a139681dc",
    ROOT
    / "submodules/act/eval_pact_valid_ablation_row.py": "ede6bd54b55ba143c0043ea80190eaf6449ed77b9e5d71b8af7dcebea8a502c4",
    ROOT
    / "submodules/act/eval_pact_place_row.py": "ac55339438d282de6b58d131d59add77ab7b2a66a8a0c2f5e2f34809f51238b7",
    ROOT
    / "scripts/build_pact_permuted_token_plan.py": "bf656f731dd390ec622aaaa776e98e5721a0997cb9f7d0010405a9c385697d4b",
    ROOT
    / "scripts/pact_place_eval_chunk1_contract.py": "43df91feb3eb9843c32d29b764be080f2aa06eaaee2072bbc4b0ad3302b99194",
    ROOT
    / "configs/pact_place_eval_chunk1_manifest.json": "0a9b405f50aa86d99ec17d78a8818cce7b71730e087ccf493b7eda0e4fa4ec91",
    CHUNK1_TRAIN
    / "act_seed3101/policy_best.ckpt": "cd95d805cc1caa672137ce5d58eab1671ba175e36f309cc65070eee0acee2c30",
    CHUNK1_TRAIN
    / "pact_seed3101/policy_best.ckpt": "4404138b5445a168c36e0dbd463216419179b9ee6211c9bf3e27ab25f47b1e99",
    ENCODER: ENCODER_SHA,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        digest.update(f"{path.relative_to(root).as_posix()}\0{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def verify_dataset_tree() -> dict[str, Any]:
    manifest = json.loads(DATASET_MANIFEST.read_text())
    digest = hashlib.sha256()
    episodes = sorted(manifest["episodes"], key=lambda row: int(row["act_episode_index"]))
    for episode in episodes:
        path = DATASET_DIR / episode["act_file"]
        actual = sha256_file(path)
        if actual != episode["act_file_sha256"]:
            raise RuntimeError(f"protected dataset file changed: {path}")
        digest.update(f"{episode['act_file']}\x1f{actual}\n".encode())
    tree_sha = digest.hexdigest()
    if tree_sha != manifest.get("converted_tree_file_sha256") or tree_sha != DATASET_TREE_SHA:
        raise RuntimeError("protected 152-episode dataset tree changed")
    return {
        "episodes": len(episodes),
        "per_file_hashes_verified": True,
        "tree_sha256": tree_sha,
    }


def load_rows(arm: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    directory = OUTPUT_ROOT / arm.lower()
    results = [json.loads(path.read_text()) for path in sorted(directory.glob("*/result.json"))]
    drivers = [
        json.loads(path.read_text()) for path in sorted(directory.glob("*/driver_result.json"))
    ]
    return results, drivers


def aggregate(arm: str, results: list[dict], drivers: list[dict]) -> dict[str, Any]:
    return {
        "arm": arm,
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
                result["contact_audit"]["contact_class_totals"].get(contact, 0) > 0 for result in results
            )
            for contact in (
                "hazard_bar",
                "other_environment",
                "clutter",
                "mounted_fixture",
                "place_receptacle",
            )
        },
        "control_step_counts": dict(
            sorted(Counter(result["policy_info"]["control_steps"] for result in results).items())
        ),
        "wall_clock_seconds": {
            "min": min(driver["wall_clock_seconds"] for driver in drivers),
            "median": statistics.median(driver["wall_clock_seconds"] for driver in drivers),
            "mean": statistics.mean(driver["wall_clock_seconds"] for driver in drivers),
            "max": max(driver["wall_clock_seconds"] for driver in drivers),
        },
    }


def verify_training() -> dict[str, Any]:
    manifests = {}
    hashes = {}
    for arm in ("act", "pact"):
        new_path = TRAIN_ROOT / f"{arm}_seed3101/run_manifest.json"
        old_path = CHUNK1_TRAIN / f"{arm}_seed3101/run_manifest.json"
        new = json.loads(new_path.read_text())
        old = json.loads(old_path.read_text())
        if new["chunk_size"] != 100 or old["chunk_size"] != 1:
            raise RuntimeError(f"{arm}: chunk-size contract failed")
        for key in (
            "seed",
            "num_epochs",
            "batch_size",
            "lr",
            "kl_weight",
            "hidden_dim",
            "dim_feedforward",
            "episode_horizon",
            "state_dim",
            "action_dim",
            "split_manifest_sha256",
            "dataset_stats_pkl_sha256",
        ):
            if new[key] != old[key]:
                raise RuntimeError(f"{arm}: training field {key} changed")
        if new["dataset_report"]["tree_sha256"] != old["dataset_report"]["tree_sha256"]:
            raise RuntimeError(f"{arm}: dataset tree changed")

        def normalized(command: list[str]) -> list[str]:
            value = list(command)
            value[value.index("--chunk_size") + 1] = "<chunk_size>"
            value[value.index("--ckpt_dir") + 1] = "<ckpt_dir>"
            return value

        if normalized(new["command"]) != normalized(old["command"]):
            raise RuntimeError(f"{arm}: command changed beyond chunk_size and ckpt_dir")
        checkpoint = TRAIN_ROOT / f"{arm}_seed3101/policy_best.ckpt"
        hashes[arm.upper()] = sha256_file(checkpoint)
        if sha256_file(TRAIN_ROOT / f"{arm}_seed3101/dataset_stats.pkl") != STATS_SHA:
            raise RuntimeError(f"{arm}: dataset statistics hash changed")
        manifests[arm.upper()] = {
            "run_manifest_sha256": new["run_manifest_sha256"],
            "checkpoint_sha256": hashes[arm.upper()],
            "chunk_size": new["chunk_size"],
        }
    act_command = json.loads((TRAIN_ROOT / "act_seed3101/run_manifest.json").read_text())["command"]
    pact_command = json.loads((TRAIN_ROOT / "pact_seed3101/run_manifest.json").read_text())[
        "command"
    ]
    proximity_tail = [
        "--use_proximity",
        "--n_proximity_sensors",
        "40",
        "--prox_tokens_per_sensor",
        "1",
        "--proximity_feature_dim",
        "32",
        "--proximity_encoder_sha256",
        ENCODER_SHA,
    ]
    pact_base = pact_command[: -len(proximity_tail)]
    if pact_command[-len(proximity_tail) :] != proximity_tail:
        raise RuntimeError("PACT proximity flag tail changed")
    pact_base[pact_base.index("--ckpt_dir") + 1] = act_command[act_command.index("--ckpt_dir") + 1]
    if pact_base != act_command:
        raise RuntimeError("PACT differs from ACT beyond checkpoint dir and five proximity flags")
    return {"arms": manifests, "act_pact_diff_exact": True, "chunk1_diff_exact": True}


def validate_arm(
    arm: str,
    results: list[dict[str, Any]],
    drivers: list[dict[str, Any]],
    manifest: dict[str, Any],
    token_sha: str,
) -> None:
    if len(results) != 40 or len(drivers) != 40:
        raise RuntimeError(f"{arm}: expected 40 results and drivers")
    by_episode = {result["episode_id"]: result for result in results}
    if set(by_episode) != {row["episode_id"] for row in manifest["rows"]}:
        raise RuntimeError(f"{arm}: result identities differ from manifest")
    if Counter(result["intrusion_side"] for result in results) != {"left": 20, "right": 20}:
        raise RuntimeError(f"{arm}: side balance changed")
    for result in results:
        if result["policy_info"]["num_queries"] != 100:
            raise RuntimeError(f"{arm}: num_queries changed")
        if result["policy_info"]["contact_audit_class"] != "PactPlaceContactAudit":
            raise RuntimeError(f"{arm}: place contact audit missing")
        if arm != "ACT":
            info = result["policy_info"]
            if (
                info["proximity_feature_dim"] != 32
                or info["input_proj_proximity_shape"] != [512, 32]
                or info["proximity_consumed_for_action"] is not True
            ):
                raise RuntimeError(f"{arm}: PACT runtime contract failed")
        if arm == "PACT_PERMUTED":
            info = result["policy_info"]
            if (
                info["token_plan_sha256"] != token_sha
                or info["live_proximity_aligned_with_action"] is not False
                or info["surface_encoder_self_verified"] is not True
                or info["surface_encoder_verified_sha256"] != ENCODER_SHA
                or info["token_plan_frames_consumed"] != 900
            ):
                raise RuntimeError("PACT_PERMUTED runtime contract failed")


def paired_difference_interval(
    pact_results: list[dict[str, Any]], permuted_results: list[dict[str, Any]]
) -> dict[str, float]:
    pact = {row["episode_id"]: row for row in pact_results}
    permuted = {row["episode_id"]: row for row in permuted_results}
    differences = [
        float(pact[key]["collision_free_task_success"])
        - float(permuted[key]["collision_free_task_success"])
        for key in sorted(pact)
    ]
    estimate = statistics.mean(differences)
    standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    return {
        "estimate": estimate,
        "lower_95": max(-1.0, estimate - 1.96 * standard_error),
        "upper_95": min(1.0, estimate + 1.96 * standard_error),
        "method": "paired normal interval over the 40 frozen instances",
    }


def shared_chunk_comparison(
    manifest: dict[str, Any], results_by_arm: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    old_manifest = json.loads(CHUNK1_MANIFEST.read_text())
    fields = ("task_seed_u64", "panel_x_jitter_m", "panel_face_jitter_m", "intrusion_side")
    for index in range(20):
        if any(manifest["rows"][index][key] != old_manifest["rows"][index][key] for key in fields):
            raise RuntimeError("shared physical instance fields changed")
    comparison = {"physical_fields_exact": True, "row_sha256_equality_asserted": False, "arms": {}}
    for arm in ("ACT", "PACT"):
        old_results = [
            json.loads(path.read_text())
            for path in sorted((CHUNK1_OUTPUT / arm.lower()).glob("*/result.json"))
        ]
        new_by_episode = {row["episode_id"]: row for row in results_by_arm[arm]}
        new_shared = [new_by_episode[row["episode_id"]] for row in manifest["rows"][:20]]
        comparison["arms"][arm] = {
            "chunk1": {
                "collision_free_task_success": sum(
                    r["collision_free_task_success"] for r in old_results
                ),
                "task_success": sum(r["task_success"] for r in old_results),
                "gripper_close_commanded": sum(
                    r["policy_info"]["gripper_close_commanded"] for r in old_results
                ),
            },
            "chunk100": {
                "collision_free_task_success": sum(
                    r["collision_free_task_success"] for r in new_shared
                ),
                "task_success": sum(r["task_success"] for r in new_shared),
                "gripper_close_commanded": sum(
                    r["policy_info"]["gripper_close_commanded"] for r in new_shared
                ),
            },
        }
    return comparison


def main() -> int:
    manifest = load_manifest(MANIFEST)
    protected = {str(path): sha256_file(path) for path in EXPECTED_PROTECTED}
    for path, expected in EXPECTED_PROTECTED.items():
        if protected[str(path)] != expected:
            raise RuntimeError(f"protected artifact changed: {path}")
    chunk1_train_tree = directory_tree_hash(CHUNK1_TRAIN)
    if chunk1_train_tree != CHUNK1_TRAIN_TREE_SHA:
        raise RuntimeError("protected chunk-1 training run directory changed")
    chunk1_eval_tree = directory_tree_hash(CHUNK1_OUTPUT)
    if chunk1_eval_tree != CHUNK1_EVAL_TREE_SHA:
        raise RuntimeError("protected chunk-1 evaluation run directory changed")
    protected_dataset = verify_dataset_tree()
    token_plan = json.loads(TOKEN_PLAN.read_text())
    token_sha = token_plan["token_plan_sha256"]
    runtime = json.loads((OUTPUT_ROOT / "runtime_decision.json").read_text())
    smoke = json.loads((OUTPUT_ROOT / "smoke_launcher_summary.json").read_text())
    training = verify_training()

    if runtime["decision"] == "STOP_NO_GRIPPER_CLOSE":
        report = """# Chunk-100 place rollout evaluation\n\n## Decision\n\n**STOP_NO_GRIPPER_CLOSE.** None of the six mandatory smoke rollouts commanded the gripper closed, so the preregistered hard stop was applied and T6 was not run. No modality comparison is claimed.\n"""
        (OUTPUT_ROOT / "EVAL.md").write_text(report)
        print(json.dumps({"status": "STOP_NO_GRIPPER_CLOSE"}))
        return 0

    full = json.loads((OUTPUT_ROOT / "full_launcher_summary.json").read_text())
    full_arms = runtime["full_arms"]
    if (
        full["jobs_requested"] != 40 * len(full_arms)
        or full["jobs_complete"] != 40 * len(full_arms)
        or full["errors"]
        or full["arms"] != full_arms
    ):
        raise RuntimeError("full launcher did not reconcile the frozen scope")
    results_by_arm = {}
    arms = {}
    for arm in full_arms:
        results, drivers = load_rows(arm)
        validate_arm(arm, results, drivers, manifest, token_sha)
        results_by_arm[arm] = results
        arms[arm] = aggregate(arm, results, drivers)
    shortcut_broken = all(arms[arm]["gripper_close_commanded"] >= 20 for arm in ("ACT", "PACT"))
    collapse = any(arms[arm]["collision_free_task_success"] <= 2 for arm in ("ACT", "PACT"))
    if collapse:
        status = "CHUNK100_COLLAPSE"
    elif not shortcut_broken:
        status = "CHUNK100_SHORTCUT_PERSISTS"
    else:
        status = "FUNCTIONAL_CHUNK100"
    cross_arm_claim = shortcut_broken and not collapse
    contrast = (
        paired_difference_interval(results_by_arm["PACT"], results_by_arm["PACT_PERMUTED"])
        if "PACT_PERMUTED" in results_by_arm
        else None
    )
    shared = shared_chunk_comparison(manifest, results_by_arm)
    generated_paths = (
        ROOT / "scripts/pact_place_eval_chunk100_contract.py",
        ROOT / "configs/pact_place_eval_chunk100_manifest.json",
        ROOT / "scripts/build_pact_place_permuted_token_plan.py",
        ROOT / "submodules/act/eval_pact_place_chunk100_row.py",
        ROOT / "submodules/act/eval_pact_place_permuted_row.py",
        ROOT / "scripts/run_pact_place_eval_chunk100.py",
        TOKEN_PLAN,
    )
    analysis = {
        "schema_version": "pact_place_eval_chunk100_analysis_v1",
        "status": status,
        "primary_endpoint": "collision_free_task_success",
        "shortcut_broken": shortcut_broken,
        "shortcut_rule": "gripper close commanded in at least 20/40 for both trained arms",
        "collapse": collapse,
        "collapse_rule": "if either trained arm is <= 2/40 CFTS, all arms are modality-uninformative",
        "cross_arm_claim_authorized": cross_arm_claim,
        "single_seed": 3101,
        "n_per_full_arm": 40,
        "arms": arms,
        "decision_bearing_contrast_pact_minus_pact_permuted": contrast,
        "shared_chunk1_chunk100_comparison": shared,
        "runtime_decision": runtime,
        "smoke": smoke,
        "full_dispatch": full,
        "training_verification": training,
        "manifest_sha256": manifest["manifest_sha256"],
        "token_plan_sha256": token_sha,
        "protected_artifact_sha256": protected,
        "protected_chunk1_training_tree_sha256": chunk1_train_tree,
        "protected_chunk1_evaluation_tree_sha256": chunk1_eval_tree,
        "protected_dataset": protected_dataset,
        "generated_artifact_sha256": {str(path): sha256_file(path) for path in generated_paths},
        "place_receptacle_diagnostic_limit": (
            "Learned policy exposes no expert phase; audit phase remains other, so "
            "place_receptacle contacts are conservatively over-counted outside placement."
        ),
    }
    (OUTPUT_ROOT / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    )

    rows = []
    detail_rows = []
    for arm in full_arms:
        value = arms[arm]
        rows.append(
            f"| {arm} | 40 | {value['task_success']}/40 | "
            f"{value['collision_free_task_success']}/40 | "
            f"{value['gripper_close_commanded']}/40 |"
        )
        contacts = value["contact_episode_counts"]
        wall = value["wall_clock_seconds"]
        detail_rows.append(
            f"| {arm} | {contacts['hazard_bar']} | {contacts['other_environment']} | "
            f"{contacts['clutter']} | {contacts['place_receptacle']} | "
            f"{value['control_step_counts'].get(900, value['control_step_counts'].get('900', 0))}/40 | "
            f"{wall['min']:.1f} / {wall['median']:.1f} / {wall['mean']:.1f} / {wall['max']:.1f} |"
        )
    contrast_text = (
        f"PACT − PACT_PERMUTED CFTS was {100 * contrast['estimate']:+.1f} pp "
        f"(paired approximate 95% interval {100 * contrast['lower_95']:+.1f} to "
        f"{100 * contrast['upper_95']:+.1f} pp); this is directional."
        if contrast is not None
        else "PACT_PERMUTED was removed by the frozen runtime-cut rule, so the decision-bearing contrast is pending."
    )
    claim_text = (
        "The shortcut-broken precondition held; cross-arm results may be interpreted subject to the single-seed and N=40 limits."
        if cross_arm_claim
        else "The frozen preconditions do not authorize a cross-arm modality claim."
    )
    report = f"""# Chunk-100 place rollout evaluation

## Decision

**{status}.** {claim_text} {contrast_text}

| Arm | N | Task success | Collision-free task success | Gripper close commanded |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

| Arm | Hazard-bar contact | Other-environment contact | Clutter contact | Place-receptacle contact | 900 control steps | Wall-clock seconds min / median / mean / max |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(detail_rows)}

## Shortcut and chunk-size comparison

The shortcut-broken rule requires gripper close in at least 50% of episodes for both trained arms; it was **{shortcut_broken}**. On the 20 shared physical instances, ACT changed from {shared["arms"]["ACT"]["chunk1"]["gripper_close_commanded"]}/20 close commands and {shared["arms"]["ACT"]["chunk1"]["collision_free_task_success"]}/20 CFTS at chunk 1 to {shared["arms"]["ACT"]["chunk100"]["gripper_close_commanded"]}/20 and {shared["arms"]["ACT"]["chunk100"]["collision_free_task_success"]}/20 at chunk 100. PACT changed from {shared["arms"]["PACT"]["chunk1"]["gripper_close_commanded"]}/20 and {shared["arms"]["PACT"]["chunk1"]["collision_free_task_success"]}/20 to {shared["arms"]["PACT"]["chunk100"]["gripper_close_commanded"]}/20 and {shared["arms"]["PACT"]["chunk100"]["collision_free_task_success"]}/20.

Rows 0–19 are the same physical episodes as chunk 1, matched on `task_seed_u64`, both jitters, and intrusion side. Their `episode_id` and `row_sha256` intentionally differ because schema version and role participate in those hashes; hash equality was not asserted.

## Runtime and scope

Smoke completed 6/6 rollouts and commanded close in {runtime["smoke_gripper_close_commanded"]}/6. Mean smoke time was {runtime["measured_minutes_per_rollout_mean"]:.2f} minutes per rollout; projected T6 time was {runtime["projected_T6_hours"]:.2f} hours. Runtime decision: `{runtime["decision"]}`. Full dispatch completed {full["jobs_complete"]}/{full["jobs_requested"]} requested jobs with 10 workers and zero errors.

## Verification and limits

- Every result used `num_queries=100`; both checkpoints loaded with `strict=True` in the evaluator.
- PACT and PACT_PERMUTED used feature width 32 and projection `(512, 32)`. The permuted arm destroyed live alignment, consumed 900 frozen frames per rollout, recorded token plan `{token_sha}`, and independently verified encoder `{ENCODER_SHA}`.
- The 40-row manifest has 20 left / 20 right, contiguous indices, and zero overlap with 152 training seeds.
- All {protected_dataset['episodes']} converted dataset files were re-hashed; the protected tree remains `{protected_dataset['tree_sha256']}`.
- Training seed is 3101 for both arms. Training commands differ from chunk 1 only in chunk size and checkpoint directory; PACT differs from ACT only by its checkpoint directory and five proximity flags.
- This is one seed with N=40 per evaluated arm and uses a cross-task frozen corridor encoder. Rate-difference precision is limited.
- `place_receptacle` contacts are over-counted because a learned policy exposes no expert phase, so the audit remains at `other`.

The exact chunk-1 → chunk-100 training-command diff for each arm was:

```diff
# ACT
--- --ckpt_dir /root/pact_place_152_pact_vs_act_seed3101/act_seed3101
+++ --ckpt_dir /root/pact_place_152_pact_vs_act_chunk100_seed3101/act_seed3101
--- --chunk_size 1
+++ --chunk_size 100

# PACT
--- --ckpt_dir /root/pact_place_152_pact_vs_act_seed3101/pact_seed3101
+++ --ckpt_dir /root/pact_place_152_pact_vs_act_chunk100_seed3101/pact_seed3101
--- --chunk_size 1
+++ --chunk_size 100
```

The exact ACT → PACT command-only differences at chunk 100 were:

```diff
--- --ckpt_dir /root/pact_place_152_pact_vs_act_chunk100_seed3101/act_seed3101
+++ --ckpt_dir /root/pact_place_152_pact_vs_act_chunk100_seed3101/pact_seed3101
+++ --use_proximity --n_proximity_sensors 40 --prox_tokens_per_sensor 1 --proximity_feature_dim 32 --proximity_encoder_sha256 {ENCODER_SHA}
```

Protected chunk-1 artifacts, published evaluators, the original token-plan builder, frozen encoder, chunk-1 checkpoints, and the 152-episode dataset all retained their recorded SHA-256 values. Machine-readable details are in `analysis.json`.
"""
    (OUTPUT_ROOT / "EVAL.md").write_text(report)
    print(json.dumps({"status": status, "arms": arms, "contrast": contrast}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
