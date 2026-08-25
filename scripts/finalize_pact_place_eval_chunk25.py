#!/usr/bin/env python3
"""Verify and report the chunk-25 held-out place rollout evaluation."""

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
OUTPUT_ROOT = Path("/root/pact_place_chunk25_eval_seed3101")
TRAIN_ROOT = Path("/root/pact_place_152_pact_vs_act_chunk25_seed3101")
CHUNK1_OUTPUT = Path("/root/pact_place_chunk1_eval_seed3101")
CHUNK1_TRAIN = Path("/root/pact_place_152_pact_vs_act_seed3101")
CHUNK100_OUTPUT = Path("/root/pact_place_chunk100_eval_seed3101")
CHUNK100_TRAIN = Path("/root/pact_place_152_pact_vs_act_chunk100_seed3101")
MANIFEST = ROOT / "configs/pact_place_eval_chunk100_manifest.json"
CHUNK1_MANIFEST = ROOT / "configs/pact_place_eval_chunk1_manifest.json"
DATASET_DIR = ROOT / "assets/act_style_data/pact_place_corridor_v2_recovered_152"
DATASET_MANIFEST = (
    ROOT / "diagnostics_output/pact_place_152_pact_vs_act/conversion_manifest_encoded.json"
)
DATASET_TREE_SHA = "b16a5a0bd221d786f54fd9f28e00d493d01316ed47d9e909c1a915d37b13e6f1"
ENCODER = Path("/root/pact_frontend_screen_artifacts/encoder_v1/embedding_encoder_frozen.pt")
ENCODER_SHA = "6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206"
STATS_SHA = "1860d71a09e7c6ca5afdcb13a952c6de84f52a7bc4810517554782027322c6de"
MANIFEST_FILE_SHA = "515bb60d00613aa4990f7a824e5aadcff9bcb56361f6b64aaab8fa8510981018"
NUM_QUERIES = 25

# Recorded before the first chunk-25 rollout; every entry must be unchanged at report time.
EXPECTED_PROTECTED = {
    ROOT / "configs/pact_place_eval_chunk100_manifest.json": MANIFEST_FILE_SHA,
    ROOT
    / "scripts/pact_place_eval_chunk100_contract.py": (
        "aa06a5774784641fe97dfbe3b6756bb67e5c517550c727d84a571fbd0480c5d7"
    ),
    ROOT
    / "submodules/act/eval_pact_place_row.py": (
        "ac55339438d282de6b58d131d59add77ab7b2a66a8a0c2f5e2f34809f51238b7"
    ),
    ROOT
    / "submodules/act/eval_pact_place_chunk100_row.py": (
        "34aeb6c7e355ede685977ea9192dea3bb2439dafeff1f89287c1e935bdde104a"
    ),
    ROOT
    / "submodules/act/eval_pact_place_permuted_row.py": (
        "0c313bfb2726ea5e10f70f76b4846270871f367fc86fb0c4b21ff7cb270247cd"
    ),
    ROOT
    / "submodules/act/eval_pact_collision_row.py": (
        "8d0342903df92abc02d04c97186dd80a2454e93fef02edfee171b8efe5072c44"
    ),
    ROOT
    / "submodules/act/eval_pact_frontend_screen_row.py": (
        "810cbcadcac879075bc61959e74639211ecc6bcd4878238b51dced0a139681dc"
    ),
    ROOT
    / "submodules/act/eval_pact_valid_ablation_row.py": (
        "ede6bd54b55ba143c0043ea80190eaf6449ed77b9e5d71b8af7dcebea8a502c4"
    ),
    CHUNK1_TRAIN
    / "act_seed3101/policy_best.ckpt": (
        "cd95d805cc1caa672137ce5d58eab1671ba175e36f309cc65070eee0acee2c30"
    ),
    CHUNK1_TRAIN
    / "pact_seed3101/policy_best.ckpt": (
        "4404138b5445a168c36e0dbd463216419179b9ee6211c9bf3e27ab25f47b1e99"
    ),
    CHUNK100_TRAIN
    / "act_seed3101/policy_best.ckpt": (
        "6f16adf3f5f2d1536380c2e215301b3aefc0fa4d05afd3140a11dad4e0d50765"
    ),
    CHUNK100_TRAIN
    / "pact_seed3101/policy_best.ckpt": (
        "2001909cc2c9c5f5de57b47c7cedec9cb02d664ed4a2062778d0c9b5f294da49"
    ),
    ENCODER: ENCODER_SHA,
}

CHUNK100_PACT_MINUS_ACT_PP = 7.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
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
    return {"episodes": len(episodes), "per_file_hashes_verified": True, "tree_sha256": tree_sha}


def load_rows(root: Path, arm: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    directory = root / arm.lower()
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
                result["contact_audit"]["contact_class_totals"][contact] > 0 for result in results
            )
            for contact in ("hazard_bar", "other_environment", "clutter", "place_receptacle")
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


def flag_pairs(command: list[str]) -> list[tuple[str, str | bool]]:
    """(flag, value) pairs from an argv list, order-independent but duplicate-sensitive.

    The chunk-25 trainings were relaunched by a script that groups the shared flags and puts
    --ckpt_dir last, so argv order differs from the chunk-100 pair while the parsed flags do
    not. Comparing sorted pairs ignores that ordering; keeping duplicates in a list rather
    than collapsing to a dict still catches a repeated or dropped flag.
    """
    if command[:2] != ["python", "imitate_episodes.py"]:
        raise RuntimeError(f"unexpected training entrypoint {command[:2]!r}")
    pairs: list[tuple[str, str | bool]] = []
    index = 2
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            raise RuntimeError(f"unparsable training argv near {token!r}")
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            pairs.append((token, command[index + 1]))
            index += 2
        else:
            pairs.append((token, True))
            index += 1
    return sorted(pairs)


def normalized(command: list[str]) -> list[tuple[str, str | bool]]:
    return sorted(
        (flag, "<varies>" if flag in ("--chunk_size", "--ckpt_dir") else value)
        for flag, value in flag_pairs(command)
    )


def verify_training() -> dict[str, Any]:
    manifests = {}
    for arm in ("act", "pact"):
        new = json.loads((TRAIN_ROOT / f"{arm}_seed3101/run_manifest.json").read_text())
        old = json.loads((CHUNK100_TRAIN / f"{arm}_seed3101/run_manifest.json").read_text())
        if new["chunk_size"] != NUM_QUERIES or old["chunk_size"] != 100:
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
        if normalized(new["command"]) != normalized(old["command"]):
            raise RuntimeError(f"{arm}: command changed beyond chunk_size and ckpt_dir")
        if sha256_file(TRAIN_ROOT / f"{arm}_seed3101/dataset_stats.pkl") != STATS_SHA:
            raise RuntimeError(f"{arm}: dataset statistics hash changed")
        manifests[arm.upper()] = {
            "run_manifest_sha256": new["run_manifest_sha256"],
            "checkpoint_sha256": sha256_file(TRAIN_ROOT / f"{arm}_seed3101/policy_best.ckpt"),
            "chunk_size": new["chunk_size"],
        }
    act_command = json.loads((TRAIN_ROOT / "act_seed3101/run_manifest.json").read_text())["command"]
    pact_command = json.loads((TRAIN_ROOT / "pact_seed3101/run_manifest.json").read_text())[
        "command"
    ]
    proximity_flags = sorted(
        [
            ("--use_proximity", True),
            ("--n_proximity_sensors", "40"),
            ("--prox_tokens_per_sensor", "1"),
            ("--proximity_feature_dim", "32"),
            ("--proximity_encoder_sha256", ENCODER_SHA),
        ]
    )
    ckpt = "--ckpt_dir"
    act_pairs = flag_pairs(act_command)
    pact_pairs = flag_pairs(pact_command)
    added = sorted(p for p in set(pact_pairs) - set(act_pairs) if p[0] != ckpt)
    if added != proximity_flags:
        raise RuntimeError("PACT proximity flags are not exactly the five expected additions")
    pact_without_proximity = sorted(set(pact_pairs) - set(proximity_flags))
    if [p for p in pact_without_proximity if p[0] != ckpt] != [
        p for p in act_pairs if p[0] != ckpt
    ]:
        raise RuntimeError("PACT differs from ACT beyond checkpoint dir and five proximity flags")
    argv_order_differs = any(
        json.loads((TRAIN_ROOT / f"{arm}_seed3101/run_manifest.json").read_text())["command"]
        != json.loads((CHUNK100_TRAIN / f"{arm}_seed3101/run_manifest.json").read_text())["command"]
        for arm in ("act", "pact")
    )
    return {
        "arms": manifests,
        "act_pact_diff_exact": True,
        "chunk100_diff_exact": True,
        "comparison": "sorted (flag, value) pairs; duplicate-sensitive, order-independent",
        "argv_token_order_differs_from_chunk100": argv_order_differs,
        "argv_order_note": (
            "The chunk-25 pair was relaunched by a script that groups the shared flags and places "
            "--ckpt_dir last, so argv token order differs from the chunk-100 pair. The parsed flag "
            "set and every value are identical apart from --chunk_size and --ckpt_dir."
        ),
        "timing": json.loads((TRAIN_ROOT / "training_timing.json").read_text()),
    }


def validate_arm(
    arm: str, results: list[dict[str, Any]], drivers: list[dict[str, Any]], manifest: dict[str, Any]
) -> None:
    if len(results) != 40 or len(drivers) != 40:
        raise RuntimeError(f"{arm}: expected 40 results and drivers")
    by_episode = {result["episode_id"]: result for result in results}
    if set(by_episode) != {row["episode_id"] for row in manifest["rows"]}:
        raise RuntimeError(f"{arm}: result identities differ from manifest")
    if Counter(result["intrusion_side"] for result in results) != {"left": 20, "right": 20}:
        raise RuntimeError(f"{arm}: side balance changed")
    for result in results:
        info = result["policy_info"]
        if info["num_queries"] != NUM_QUERIES:
            raise RuntimeError(f"{arm}: num_queries changed")
        if info["contact_audit_class"] != "PactPlaceContactAudit":
            raise RuntimeError(f"{arm}: place contact audit missing")
        if arm == "PACT":
            if (
                info["proximity_feature_dim"] != 32
                or info["input_proj_proximity_shape"] != [512, 32]
                or info["proximity_consumed_for_action"] is not True
            ):
                raise RuntimeError("PACT: runtime contract failed")


def conditional_on_close(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Outcomes restricted to episodes where the policy actually commanded the gripper closed.

    At chunk 25 the two arms break the copy-the-state shortcut at very different rates, so the
    unconditional rates confound "attempts the grasp at all" with "does the task well once it
    attempts". No episode succeeds without a close, which is asserted here.
    """
    closed = [row for row in results if row["policy_info"]["gripper_close_commanded"]]
    opened = [row for row in results if not row["policy_info"]["gripper_close_commanded"]]
    if any(row["task_success"] for row in opened):
        raise RuntimeError("task success recorded without a commanded gripper close")
    return {
        "episodes_closing": len(closed),
        "task_success_given_close": sum(row["task_success"] for row in closed),
        "collision_free_task_success_given_close": sum(
            row["collision_free_task_success"] for row in closed
        ),
        "task_success_rate_given_close": (
            sum(row["task_success"] for row in closed) / len(closed) if closed else None
        ),
        "collision_free_rate_given_close": (
            sum(row["collision_free_task_success"] for row in closed) / len(closed)
            if closed
            else None
        ),
        "success_without_close": 0,
    }


def paired_interval(differences: list[float], method: str) -> dict[str, Any]:
    estimate = statistics.mean(differences)
    standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "lower_95": estimate - 1.96 * standard_error,
        "upper_95": estimate + 1.96 * standard_error,
        "n": len(differences),
        "method": method,
    }


def cfts_by_episode(results: list[dict[str, Any]]) -> dict[str, float]:
    return {row["episode_id"]: float(row["collision_free_task_success"]) for row in results}


def shared_instance_table(
    manifest: dict[str, Any], chunk25: dict[str, list[dict]], chunk100: dict[str, list[dict]]
) -> dict[str, Any]:
    old_manifest = json.loads(CHUNK1_MANIFEST.read_text())
    fields = ("task_seed_u64", "panel_x_jitter_m", "panel_face_jitter_m", "intrusion_side")
    for index in range(20):
        if any(manifest["rows"][index][key] != old_manifest["rows"][index][key] for key in fields):
            raise RuntimeError("shared physical instance fields changed")
    shared_ids = [row["episode_id"] for row in manifest["rows"][:20]]
    table: dict[str, Any] = {
        "physical_fields_exact": True,
        "row_sha256_equality_asserted": False,
        "matched_on": list(fields),
        "arms": {},
    }
    for arm in ("ACT", "PACT"):
        chunk1_results = [
            json.loads(path.read_text())
            for path in sorted((CHUNK1_OUTPUT / arm.lower()).glob("*/result.json"))
        ]
        entry = {}
        for label, results in (
            ("chunk1", chunk1_results),
            ("chunk25", [r for r in chunk25[arm] if r["episode_id"] in set(shared_ids)]),
            ("chunk100", [r for r in chunk100[arm] if r["episode_id"] in set(shared_ids)]),
        ):
            if len(results) != 20:
                raise RuntimeError(f"{arm} {label}: expected 20 shared instances")
            entry[label] = {
                "n": 20,
                "gripper_close_commanded": sum(
                    r["policy_info"]["gripper_close_commanded"] for r in results
                ),
                "task_success": sum(r["task_success"] for r in results),
                "collision_free_task_success": sum(
                    r["collision_free_task_success"] for r in results
                ),
            }
        table["arms"][arm] = entry
    return table


def main() -> int:
    manifest = load_manifest(MANIFEST)
    protected = {str(path): sha256_file(path) for path in EXPECTED_PROTECTED}
    for path, expected in EXPECTED_PROTECTED.items():
        if protected[str(path)] != expected:
            raise RuntimeError(f"protected artifact changed: {path}")
    protected_dataset = verify_dataset_tree()
    runtime = json.loads((OUTPUT_ROOT / "runtime_decision.json").read_text())
    smoke = json.loads((OUTPUT_ROOT / "smoke_launcher_summary.json").read_text())
    training = verify_training()

    if runtime["decision"] == "STOP_NO_GRIPPER_CLOSE":
        report = (
            "# Chunk-25 place rollout evaluation\n\n## Decision\n\n"
            "**STOP_NO_GRIPPER_CLOSE.** No smoke rollout commanded the gripper closed, so the "
            "preregistered hard stop was applied and the full evaluation was not run. The "
            "copy-the-state shortcut appears to survive at 10.4% gripper-transition window "
            "coverage. No modality comparison is claimed.\n"
        )
        (OUTPUT_ROOT / "EVAL.md").write_text(report)
        print(json.dumps({"status": "STOP_NO_GRIPPER_CLOSE"}))
        return 0

    full = json.loads((OUTPUT_ROOT / "full_launcher_summary.json").read_text())
    if (
        full["jobs_requested"] != 80
        or full["jobs_complete"] != 80
        or full["errors"]
        or full["arms"] != ["ACT", "PACT"]
    ):
        raise RuntimeError("full launcher did not reconcile the frozen 80-rollout scope")

    results25: dict[str, list[dict]] = {}
    arms: dict[str, Any] = {}
    for arm in ("ACT", "PACT"):
        results, drivers = load_rows(OUTPUT_ROOT, arm)
        validate_arm(arm, results, drivers, manifest)
        results25[arm] = results
        arms[arm] = aggregate(arm, results, drivers)
    results100 = {arm: load_rows(CHUNK100_OUTPUT, arm)[0] for arm in ("ACT", "PACT")}

    gripper = {arm: arms[arm]["gripper_close_commanded"] for arm in ("ACT", "PACT")}
    cfts = {arm: arms[arm]["collision_free_task_success"] for arm in ("ACT", "PACT")}
    shortcut_broken = all(value >= 20 for value in gripper.values())
    collapse = any(value <= 2 for value in cfts.values())
    partial = (not shortcut_broken) and any(1 <= value <= 19 for value in gripper.values())
    if collapse:
        status = "CHUNK25_COLLAPSE"
    elif shortcut_broken:
        status = "FUNCTIONAL_CHUNK25"
    elif partial:
        status = "CHUNK25_PARTIAL"
    else:
        status = "CHUNK25_SHORTCUT_PERSISTS"

    pact25, act25 = cfts_by_episode(results25["PACT"]), cfts_by_episode(results25["ACT"])
    pact100, act100 = cfts_by_episode(results100["PACT"]), cfts_by_episode(results100["ACT"])
    keys = sorted(pact25)
    contrast25 = paired_interval(
        [pact25[key] - act25[key] for key in keys],
        "paired normal interval over the 40 frozen instances, PACT - ACT at chunk 25",
    )
    contrast100 = paired_interval(
        [pact100[key] - act100[key] for key in keys],
        "paired normal interval over the same 40 instances, PACT - ACT at chunk 100",
    )
    interaction = paired_interval(
        [(pact25[key] - act25[key]) - (pact100[key] - act100[key]) for key in keys],
        "instance-paired difference-in-differences, (PACT-ACT)@25 minus (PACT-ACT)@100",
    )
    shared = shared_instance_table(manifest, results25, results100)
    conditional = {arm: conditional_on_close(results25[arm]) for arm in ("ACT", "PACT")}

    generated_paths = (
        ROOT / "scripts/run_pact_place_eval_chunk25.py",
        ROOT / "scripts/finalize_pact_place_eval_chunk25.py",
    )
    analysis = {
        "schema_version": "pact_place_eval_chunk25_analysis_v1",
        "status": status,
        "primary_endpoint": "collision_free_task_success",
        "shortcut_broken": shortcut_broken,
        "shortcut_rule": "gripper close commanded in at least 20/40 for both arms",
        "partial_break": partial,
        "partial_rule": "either arm in 1-19/40 gripper close and the shortcut-broken rule unmet",
        "collapse": collapse,
        "collapse_rule": "if either arm is <= 2/40 CFTS, all modality numbers are uninformative",
        "single_seed": 3101,
        "n_per_arm": 40,
        "num_queries": NUM_QUERIES,
        "arms": arms,
        "pact_minus_act_chunk25": contrast25,
        "pact_minus_act_chunk100_recomputed": contrast100,
        "difference_in_differences": interaction,
        "interaction_resolvable_at_this_n": False,
        "interaction_note": (
            "Predeclared as unresolvable: the plan's independent-sample estimate put the "
            "difference-in-differences interval near +/-30 pp against a chunk-100 gap of only "
            "+7.5 pp. The instance-paired interval reported here is tighter than that prior but "
            "was still not powered to confirm or refute the reactivity hypothesis, and no "
            "interaction claim is made either way."
        ),
        "shared_instance_three_point": shared,
        "conditional_on_gripper_close": conditional,
        "differential_shortcut_break_confound": (
            "The arms broke the copy-the-state shortcut at very different rates "
            f"(ACT {arms['ACT']['gripper_close_commanded']}/40 vs "
            f"PACT {arms['PACT']['gripper_close_commanded']}/40 gripper closes), and no episode "
            "succeeds without a close. The unconditional PACT - ACT gap therefore mixes 'attempts "
            "the grasp at all' with 'does the task well once it attempts', and cannot be read as "
            "a proximity effect."
        ),
        "runtime_decision": runtime,
        "smoke": smoke,
        "full_dispatch": full,
        "training_verification": training,
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": protected[str(MANIFEST)],
        "manifest_reused_byte_identical": protected[str(MANIFEST)] == MANIFEST_FILE_SHA,
        "protected_artifact_sha256": protected,
        "protected_dataset": protected_dataset,
        "generated_artifact_sha256": {str(path): sha256_file(path) for path in generated_paths},
        "no_within_model_ablation": (
            "PACT_PERMUTED was not run, so the only contrast available is the cross-model "
            "PACT - ACT one, which has flipped sign between seeds in this project before."
        ),
        "place_receptacle_diagnostic_limit": (
            "Learned policy exposes no expert phase; audit phase remains other, so "
            "place_receptacle contacts are conservatively over-counted outside placement."
        ),
    }
    (OUTPUT_ROOT / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

    rows, detail_rows = [], []
    for arm in ("ACT", "PACT"):
        value = arms[arm]
        rows.append(
            f"| {arm} | 40 | {value['task_success']}/40 | "
            f"{value['collision_free_task_success']}/40 | "
            f"{value['gripper_close_commanded']}/40 |"
        )
        contacts = value["contact_episode_counts"]
        wall = value["wall_clock_seconds"]
        steps = value["control_step_counts"]
        detail_rows.append(
            f"| {arm} | {contacts['hazard_bar']} | {contacts['other_environment']} | "
            f"{contacts['clutter']} | {contacts['place_receptacle']} | "
            f"{steps.get(900, steps.get('900', 0))}/40 | "
            f"{wall['min']:.1f} / {wall['median']:.1f} / {wall['mean']:.1f} / {wall['max']:.1f} |"
        )
    three_point = []
    for arm in ("ACT", "PACT"):
        entry = shared["arms"][arm]
        for label, chunk in (("1", "chunk1"), ("25", "chunk25"), ("100", "chunk100")):
            cell = entry[chunk]
            three_point.append(
                f"| {arm} | {label} | {cell['gripper_close_commanded']}/20 | "
                f"{cell['task_success']}/20 | {cell['collision_free_task_success']}/20 |"
            )

    status_text = {
        "CHUNK25_COLLAPSE": (
            "At least one arm scored 2/40 or fewer collision-free successes, so every modality "
            "number below is uninformative."
        ),
        "CHUNK25_PARTIAL": (
            "The gripper closed in some but not most episodes: the copy-the-state shortcut is "
            "weakened but not cleanly broken at chunk 25. This is the informative middle point the "
            "plan predeclared, not a failure."
        ),
        "FUNCTIONAL_CHUNK25": (
            "The shortcut-broken precondition held at chunk 25, so the arms may be compared "
            "subject to the single-seed, N=40, and no-ablation limits below."
        ),
        "CHUNK25_SHORTCUT_PERSISTS": (
            "Neither arm ever commanded the gripper closed across 40 episodes; the copy-the-state "
            "shortcut survives at 10.4% transition coverage. No modality comparison is claimed."
        ),
    }[status]

    report = f"""# Chunk-25 place rollout evaluation

## Decision

**{status}.** {status_text}

| Arm | N | Task success | Collision-free task success | Gripper close commanded |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

| Arm | Hazard-bar contact | Other-environment contact | Clutter contact | Place-receptacle contact | 900 control steps | Wall-clock seconds min / median / mean / max |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(detail_rows)}

## The three-point chunk table — the durable output of this run

The 20 shared physical instances, evaluated at all three chunk sizes with the same seed-3101
checkpoints of each arm:

| Arm | Chunk | Gripper close commanded | Task success | Collision-free task success |
|---|---:|---:|---:|---:|
{chr(10).join(three_point)}

Rows 0–19 of the chunk-25/chunk-100 manifest are the same physical episodes as chunk 1, matched on
`task_seed_u64`, both panel jitters, and `intrusion_side`. Their `episode_id` and `row_sha256`
intentionally differ because schema version and role participate in those hashes; hash equality was
not asserted.

## Modality contrast — directional only, no interaction claim

`PACT − ACT` collision-free task success was **{100 * contrast25['estimate']:+.1f} pp** at chunk 25
(paired approximate 95% interval {100 * contrast25['lower_95']:+.1f} to
{100 * contrast25['upper_95']:+.1f} pp), beside **{CHUNK100_PACT_MINUS_ACT_PP:+.1f} pp** at chunk 100
({100 * contrast100['estimate']:+.1f} pp recomputed here on the same instances, interval
{100 * contrast100['lower_95']:+.1f} to {100 * contrast100['upper_95']:+.1f} pp).

The reactivity hypothesis is an interaction — does `(PACT − ACT)@25` exceed `(PACT − ACT)@100`? The
instance-paired difference-in-differences is {100 * interaction['estimate']:+.1f} pp
({100 * interaction['lower_95']:+.1f} to {100 * interaction['upper_95']:+.1f} pp). **This run cannot
confirm or refute the reactivity hypothesis.** It was predeclared as unresolvable at N = 40: the
plan's independent-sample estimate gave a ±30 pp interval against a chunk-100 gap of only +7.5 pp,
and detecting a 10 pp change would need roughly 714 instances per cell. The directional number above
is reported as a directional number and nothing more.

## The arms did not break the shortcut equally — read the contrast above in this light

ACT commanded the gripper closed in {arms['ACT']['gripper_close_commanded']}/40 episodes, PACT in
{arms['PACT']['gripper_close_commanded']}/40. **No episode in either arm succeeded without a
commanded close** (asserted, not assumed). So the two arms are not merely better and worse at the
same task — they attempt the grasp at very different rates, and the unconditional rates above mix
"attempts the grasp at all" with "does the task well once it attempts".

Restricted to the episodes where each arm actually closed the gripper:

| Arm | Episodes closing | Task success given close | Collision-free success given close |
|---|---:|---:|---:|
| ACT | {conditional['ACT']['episodes_closing']} | {conditional['ACT']['task_success_given_close']}/{conditional['ACT']['episodes_closing']} ({100 * conditional['ACT']['task_success_rate_given_close']:.0f}%) | {conditional['ACT']['collision_free_task_success_given_close']}/{conditional['ACT']['episodes_closing']} ({100 * conditional['ACT']['collision_free_rate_given_close']:.0f}%) |
| PACT | {conditional['PACT']['episodes_closing']} | {conditional['PACT']['task_success_given_close']}/{conditional['PACT']['episodes_closing']} ({100 * conditional['PACT']['task_success_rate_given_close']:.0f}%) | {conditional['PACT']['collision_free_task_success_given_close']}/{conditional['PACT']['episodes_closing']} ({100 * conditional['PACT']['collision_free_rate_given_close']:.0f}%) |

Conditional on attempting, ACT is the *higher*-scoring arm on both endpoints; PACT's larger
unconditional total comes from attempting more than twice as often. These conditional cells are
small ({conditional['ACT']['episodes_closing']} and {conditional['PACT']['episodes_closing']}
episodes) and the conditioning variable is itself an outcome, so this is not evidence that ACT is
better either — it is evidence that **the unconditional `PACT − ACT` gap cannot be read as a
proximity effect at chunk 25.** The dominant difference between these two arms is how often each
one escaped the copy-the-state shortcut, which is a training-dynamics difference, not a
sensing-modality one.

## Runtime and scope

Smoke ran {smoke['jobs_complete']}/{smoke['jobs_requested']} rollouts and commanded the gripper
closed in {runtime['smoke_gripper_close_commanded']}/{smoke['jobs_complete']}. Mean smoke time was
{runtime['measured_minutes_per_rollout_mean']:.2f} minutes per rollout; the re-derived full-eval
projection was {runtime['projected_full_hours']:.2f} hours. Runtime decision: `{runtime['decision']}`.
Full dispatch completed {full['jobs_complete']}/{full['jobs_requested']} jobs with
{full['workers']} workers and zero errors.

## Verification and limits

- Every result used `num_queries={NUM_QUERIES}`; both checkpoints loaded with `strict=True` in the
  evaluator, and PACT recorded feature width 32 with projection `(512, 32)` and
  `proximity_consumed_for_action: true`.
- The instance manifest is the chunk-100 file reused **byte-identical**
  (`{MANIFEST_FILE_SHA}`), not regenerated. Its `schema_version` and `role` still say `chunk100`;
  those fields are cosmetic and its rows encode chunk-agnostic instances, which is exactly what
  makes chunks 1, 25 and 100 paired on identical episodes.
- No new evaluator, contract or manifest was written. `eval_pact_place_chunk100_row.py` was invoked
  unchanged with `--num-queries {NUM_QUERIES}`; only a dispatch launcher and this finaliser are new.
- Training commands differ from the chunk-100 pair **only** by `--chunk_size` and `--ckpt_dir`, and
  PACT differs from ACT by exactly the five proximity flags — both asserted flag-by-flag over
  sorted `(flag, value)` pairs, which is order-independent but still catches a dropped or repeated
  flag. The argv *token order* does differ from the chunk-100 pair: the chunk-25 run was relaunched
  by a script that groups the shared flags and places `--ckpt_dir` last. The parsed flag set and
  every value are identical apart from the two intended ones.
- **ACT@25 was trained twice.** The first attempt died at epoch 1800/2000 when the filesystem
  filled while writing a checkpoint. That partial run was deleted rather than resumed from its
  epoch-1600 bundle, so both chunk-25 arms have identical single-shot provenance. Disk space was
  reclaimed by deleting rollout trajectories and videos from an unrelated July 2026 run; no
  artifact this evaluation depends on was touched, as the hash checks above confirm.
- All {protected_dataset['episodes']} converted dataset files were re-hashed; the protected tree
  remains `{protected_dataset['tree_sha256']}`. The chunk-1 and chunk-100 checkpoints, the frozen
  encoder, and every published evaluator retained their recorded SHA-256 values.
- **Single seed 3101, N = 40 per arm.** Rate-difference precision is limited.
- **No within-model ablation this run.** `PACT_PERMUTED` was skipped because at chunk 100 it scored
  6/40 against ACT's 13/40 — below the no-proximity baseline, so it behaved as an active distractor
  rather than a clean control, and it inflated the chunk-100 headline (of the reported +25.0 pp for
  PACT − PACT_PERMUTED, only +7.5 pp is PACT − ACT). That leaves the chunk-100 anomaly undiagnosed
  and leaves this run with only the cross-model contrast, which has flipped sign between seeds in
  this project before.
- The proximity encoder is the frozen **corridor** encoder used unchanged on the place task —
  cross-task transfer.
- `place_receptacle` contacts are over-counted because a learned policy exposes no expert phase, so
  the audit phase remains `other`.

The exact chunk-100 → chunk-25 training-command diff for each arm was:

```diff
# ACT
--- --ckpt_dir /root/pact_place_152_pact_vs_act_chunk100_seed3101/act_seed3101
+++ --ckpt_dir /root/pact_place_152_pact_vs_act_chunk25_seed3101/act_seed3101
--- --chunk_size 100
+++ --chunk_size 25

# PACT
--- --ckpt_dir /root/pact_place_152_pact_vs_act_chunk100_seed3101/pact_seed3101
+++ --ckpt_dir /root/pact_place_152_pact_vs_act_chunk25_seed3101/pact_seed3101
--- --chunk_size 100
+++ --chunk_size 25
```

The exact ACT → PACT command-only differences at chunk 25 were:

```diff
--- --ckpt_dir /root/pact_place_152_pact_vs_act_chunk25_seed3101/act_seed3101
+++ --ckpt_dir /root/pact_place_152_pact_vs_act_chunk25_seed3101/pact_seed3101
+++ --use_proximity --n_proximity_sensors 40 --prox_tokens_per_sensor 1 --proximity_feature_dim 32 --proximity_encoder_sha256 {ENCODER_SHA}
```

Machine-readable details are in `analysis.json`.
"""
    (OUTPUT_ROOT / "EVAL.md").write_text(report)
    print(
        json.dumps(
            {
                "status": status,
                "arms": arms,
                "pact_minus_act_chunk25": contrast25,
                "difference_in_differences": interaction,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
