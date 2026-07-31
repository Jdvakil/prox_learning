from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pact_frontend_screen_contract import (
    INSTANCE_COUNT,
    build_manifest,
    sha256_payload,
    validate_manifest,
    validate_recovery_event,
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analysis = _load(
    "analyze_pact_frontend_screen",
    ROOT / "scripts/analyze_pact_frontend_screen.py",
)
schedule_builder = _load(
    "build_pact_frontend_screen_schedule",
    ROOT / "scripts/build_pact_frontend_screen_schedule.py",
)
supervisor_module = _load(
    "run_pact_frontend_screen_supervisor",
    ROOT / "scripts/run_pact_frontend_screen_supervisor.py",
)


def _instances(a_only: int, b_only: int, neither: int = 0):
    instances = []
    for _ in range(a_only):
        instances.append(
            {
                "PACT": {"collision_free_task_success": True},
                "PACT_ZERO": {"collision_free_task_success": False},
            }
        )
    for _ in range(b_only):
        instances.append(
            {
                "PACT": {"collision_free_task_success": False},
                "PACT_ZERO": {"collision_free_task_success": True},
            }
        )
    for _ in range(neither):
        instances.append(
            {
                "PACT": {"collision_free_task_success": False},
                "PACT_ZERO": {"collision_free_task_success": False},
            }
        )
    return instances


def test_screen_manifest_is_balanced_and_disjoint():
    excluded = {"prior-a", "prior-b"}
    manifest = build_manifest(
        source_hashes={"test": "0" * 64},
        sensor_names=[f"sensor_{index:02d}" for index in range(40)],
        excluded_episode_ids=excluded,
        excluded_manifests={"r1": "1" * 64, "r2": "2" * 64},
    )
    validate_manifest(manifest, excluded_episode_ids=excluded)
    assert len(manifest["rows"]) == INSTANCE_COUNT
    assert {
        side: sum(
            row["intrusion_side"] == side for row in manifest["rows"]
        )
        for side in ("left", "right")
    } == {"left": 20, "right": 20}
    assert not (
        {row["episode_id"] for row in manifest["rows"]} & excluded
    )


def test_mcnemar_reports_directional_discordant_pairs():
    result = analysis.discordant_pairs(
        _instances(6, 1, 33), arm_a="PACT", arm_b="PACT_ZERO"
    )
    assert result["arm_a_success_arm_b_failure"] == 6
    assert result["arm_a_failure_arm_b_success"] == 1
    assert result["discordant_pairs"] == 7
    assert 0.0 <= result["p_value_exact_two_sided"] <= 1.0


def test_screen_decision_rule_cannot_emit_confirmatory_tokens():
    present = {
        "difference": 0.10,
        "ci_95": [0.025, 0.20],
    }
    weak = {"difference": 0.075, "ci_95": [-0.05, 0.20]}
    none = {"difference": 0.025, "ci_95": [-0.10, 0.15]}
    assert (
        analysis.choose_decision(
            reconciled=True, pact_minus_zero=present
        )
        == "FRONTEND_SCREEN_SIGNAL_PRESENT"
    )
    assert (
        analysis.choose_decision(
            reconciled=True, pact_minus_zero=weak
        )
        == "FRONTEND_SCREEN_WEAK_SIGNAL"
    )
    assert (
        analysis.choose_decision(
            reconciled=True, pact_minus_zero=none
        )
        == "FRONTEND_SCREEN_NO_SIGNAL"
    )
    assert (
        analysis.choose_decision(
            reconciled=False, pact_minus_zero=None
        )
        == "FRONTEND_SCREEN_INCONCLUSIVE"
    )


def test_preregistration_self_hash_and_frozen_screen_rule():
    path = (
        ROOT
        / "configs"
        / "pact_frontend_screen_preregistration_v1.json"
    )
    document = json.loads(path.read_text())
    observed = document.pop("preregistration_sha256")
    expected = hashlib.sha256(
        json.dumps(
            document, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    assert observed == expected
    assert document["design"]["rollouts"] == 120
    assert document["design"]["workers"] == 8
    assert document["frontend"]["primary_variant"].startswith(
        "embedding32"
    )
    assert document["decision_rule"][
        "FRONTEND_SCREEN_SIGNAL_PRESENT"
    ] == "PACT_minus_PACT_ZERO_ge_0.10_and_paired_CI_lower_gt_0"


def test_dataset_hash_amendment_is_outcome_blind_and_self_hashed():
    path = (
        ROOT
        / "configs"
        / "pact_frontend_screen_dataset_hash_amendment_v1.json"
    )
    document = json.loads(path.read_text())
    observed = document.pop("amendment_sha256")
    expected = hashlib.sha256(
        json.dumps(
            document, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    assert observed == expected
    assert not any(
        document["outcome_blind_timing"].values()
    )
    assert document["rematerializer_sha256_new"] == hashlib.sha256(
        (
            ROOT / "scripts/prepare_pact_embedding_dataset.py"
        ).read_bytes()
    ).hexdigest()


def test_screen_arm_orders_are_balanced_without_changing_n():
    orders = schedule_builder.arm_orders(40)
    assert len(orders) == 40
    assert all(set(order) == {"ACT", "PACT", "PACT_ZERO"} for order in orders)
    for position in range(3):
        counts = {
            arm: sum(order[position] == arm for order in orders)
            for arm in ("ACT", "PACT", "PACT_ZERO")
        }
        assert max(counts.values()) - min(counts.values()) <= 1


def test_screen_runtime_sources_bind_detachment_and_group_recovery():
    supervisor = (
        ROOT
        / "scripts"
        / "run_pact_frontend_screen_supervisor.py"
    ).read_text()
    launcher = (
        ROOT
        / "scripts"
        / "launch_pact_frontend_screen_detached.py"
    ).read_text()
    proof = (
        ROOT
        / "scripts"
        / "prove_pact_frontend_screen_detachment.py"
    ).read_text()
    assert "WORKERS = 8" in supervisor
    assert "pact_frontend_screen_group_recovery_v1" in supervisor
    assert "eval_pact_frontend_screen_row.py" in supervisor
    assert '"/usr/bin/setsid"' in launcher
    assert '"/usr/bin/nohup"' in launcher
    assert "/root/prox_learning_pact_remediation/assets" in launcher
    assert "os.kill(launching_shell.pid, signal.SIGKILL)" in proof


def test_screen_group_recovery_authorizes_every_recorded_row(tmp_path):
    rows = [
        {
            "rollout_id": f"rollout-{index}",
            "schedule_row_sha256": str(index) * 64,
            "attempt_index": 0,
            "result_present": False,
        }
        for index in (1, 2)
    ]
    event = {
        "schema_version": (
            "pact_frontend_screen_group_recovery_v1"
        ),
        "qualifying_indiscriminate_termination": True,
        "all_inflight_rows_rerun": True,
        "result_absent_for_all": True,
        "active_cohort_size": 2,
        "rows": rows,
    }
    event["recovery_event_sha256"] = sha256_payload(event)
    path = tmp_path / "event.json"
    path.write_text(json.dumps(event))
    for row in rows:
        assert (
            validate_recovery_event(
                path,
                rollout_id=row["rollout_id"],
                schedule_row_sha256=row[
                    "schedule_row_sha256"
                ],
                attempt_index=1,
            )["active_cohort_size"]
            == 2
        )


def test_supervisor_command_uses_screen_evaluator_and_recovery(tmp_path):
    supervisor = object.__new__(
        supervisor_module.ScreenSupervisor
    )
    supervisor.manifest_path = tmp_path / "manifest.json"
    supervisor.output_root = tmp_path / "output"
    row = {
        "arm": "PACT",
        "instance_episode_id": "episode",
        "checkpoint_path": str(tmp_path / "policy_best.ckpt"),
        "checkpoint_sha256": "a" * 64,
        "checkpoint_seed": 3101,
        "dataset_stats_sha256": "b" * 64,
        "schedule_row_sha256": "c" * 64,
        "rollout_id": "rollout",
        "surface_encoder_path": str(tmp_path / "encoder.pt"),
        "surface_encoder_sha256": "d" * 64,
        "output_relpath": "rows/000",
    }
    recovery = tmp_path / "recovery.json"
    command = supervisor._command(
        row, attempt_index=2, recovery_event=recovery
    )
    assert command[1].endswith(
        "eval_pact_frontend_screen_row.py"
    )
    assert command[command.index("--attempt-index") + 1] == "2"
    assert (
        command[
            command.index("--inflight-recovery-event") + 1
        ]
        == str(recovery)
    )


def test_frozen_analysis_reconciles_exact_pairs_and_emits_screen_token(
    tmp_path,
):
    rows = []
    for instance in range(40):
        episode_id = f"episode-{instance:02d}"
        outcomes = {
            "ACT": instance < 4,
            "PACT": instance < 8,
            "PACT_ZERO": instance < 2,
        }
        for arm in ("ACT", "PACT", "PACT_ZERO"):
            index = len(rows)
            rollout_id = f"rollout-{index:03d}"
            row = {
                "schedule_index": index,
                "instance_episode_id": episode_id,
                "arm": arm,
                "checkpoint_seed": 3101,
                "checkpoint_sha256": f"{index:064x}",
                "schedule_row_sha256": f"{index + 1:064x}",
                "rollout_id": rollout_id,
                "output_relpath": f"rows/{index:03d}",
            }
            rows.append(row)
            row_dir = tmp_path / row["output_relpath"]
            row_dir.mkdir(parents=True)
            (row_dir / "driver_result.json").write_text(
                json.dumps({"status": "complete"})
            )
            success = outcomes[arm]
            result = {
                "status": "complete",
                "rollout_id": rollout_id,
                "schedule_row_sha256": row[
                    "schedule_row_sha256"
                ],
                "episode_id": episode_id,
                "arm": arm,
                "checkpoint_sha256": row["checkpoint_sha256"],
                "checkpoint_seed": 3101,
                "task_success": success,
                "collision_free_task_success": success,
                "contact_audit": {
                    "contact_class_totals": {
                        "grasp_target": int(success),
                        "hazard_bar": 0,
                        "other_environment": 0,
                    }
                },
                "failure_taxonomy": (
                    "collision_free_task_success"
                    if success
                    else "no_gripper_close"
                ),
            }
            (row_dir / "result.json").write_text(
                json.dumps(result)
            )
    schedule = {
        "schedule_sha256": "schedule",
        "instances": 40,
        "rollouts": 120,
        "workers": 8,
        "fresh_subprocess_per_rollout": True,
        "detectable_effect_statement": "screen",
        "bootstrap_replicates": 20000,
        "bootstrap_seed": 2026073103,
        "rows": rows,
    }
    report, decision = analysis.analyze(schedule, tmp_path)
    assert report["reconciliation"]["valid"] == 120
    assert report["mcnemar_exact_primary"][
        "arm_a_success_arm_b_failure"
    ] == 6
    assert report["mcnemar_exact_primary"][
        "arm_a_failure_arm_b_success"
    ] == 0
    assert (
        decision["decision"]
        == "FRONTEND_SCREEN_SIGNAL_PRESENT"
    )
