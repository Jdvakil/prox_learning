from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_pact_blind_rgb as analysis
import build_pact_blind_rgb_config as config_builder
import build_pact_blind_rgb_schedule as schedule_builder
import pact_blind_rgb_contract as contract


ACT = ROOT / "submodules/act"
MODULE_PATH = ACT / "pact_blur.py"
FRONTEND_PATH = ACT / "eval_pact_frontend_screen_row.py"
COLLISION_PATH = ACT / "eval_pact_collision_row.py"
SOURCE_MANIFEST = ROOT / "configs/pact_blur_sweep_v1.json"
BLUR_SCHEDULE = ROOT / "diagnostics_output/pact_blur_sweep/schedule.json"
REGISTRY = ROOT / "diagnostics_output/pact_contact_endpoint/policy_training.json"
TOKEN_PLAN = ROOT / "diagnostics_output/pact_contact_endpoint/token_plan.json"
ANALYSIS_SCRIPT = ROOT / "scripts/analyze_pact_blind_rgb.py"


def load_primitive():
    spec = importlib.util.spec_from_file_location("pact_blur", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_manifest() -> dict:
    return config_builder.build(
        config_builder.load_blur_manifest(SOURCE_MANIFEST), SOURCE_MANIFEST
    )


def build_schedule() -> dict:
    return schedule_builder.build(
        manifest=build_manifest(),
        registry=json.loads(REGISTRY.read_text()),
        token_plan=json.loads(TOKEN_PLAN.read_text()),
        token_plan_path=TOKEN_PLAN,
        blur_schedule=json.loads(BLUR_SCHEDULE.read_text()),
        analysis_script=ANALYSIS_SCRIPT,
    )


def test_mean_fill_is_exact_constant_without_mutating_source() -> None:
    module = load_primitive()
    image = torch.rand(2, 1, 3, 11, 13)
    original = image.clone()
    blinded = module.mean_fill_images(image)
    expected = module.IMAGENET_MEAN.to(image).view(1, 1, 3, 1, 1).expand_as(image)
    assert torch.equal(blinded, expected)
    assert torch.equal(image, original)
    assert blinded.shape == image.shape
    assert blinded.dtype == image.dtype
    assert blinded.device == image.device


def test_live_frontend_applies_blind_before_proximity_and_cli_rejects_mix() -> None:
    frontend = FRONTEND_PATH.read_text()
    collision = COLLISION_PATH.read_text()
    assert "if self.pc.blind_rgb:" in frontend
    assert "pact_blur.mean_fill_images(image_tensor)" in frontend
    assert frontend.index("pact_blur.mean_fill_images") < frontend.index(
        "raw = self._raw_proximity(observation)"
    )
    assert 'parser.add_argument("--blind-rgb", action="store_true")' in collision
    assert "args.blind_rgb and args.blur_sigma != 0.0" in collision
    assert '"blind_rgb": bool(args.blind_rgb)' in collision
    assert "blind_rgb=bool(args.blind_rgb)" in collision


def test_config_reuses_exact_25_blur_instances_without_outcome_selection() -> None:
    document = build_manifest()
    contract.validate_manifest(document)
    source = json.loads(SOURCE_MANIFEST.read_text())
    assert document["source_blur_manifest"]["policy_outcomes_read_for_selection"] is False
    assert [row["episode_id"] for row in document["rows"]] == [
        row["episode_id"] for row in source["rows"]
    ]
    assert len(document["rows"]) == 25
    assert document["predeclared_expected_outcome"]["hazard_contact"].startswith("PACT_BLIND")


def test_schedule_is_balanced_450_and_smoke_exercises_blind_permuted() -> None:
    schedule = build_schedule()
    assert schedule["rollouts"] == len(schedule["rows"]) == 450
    assert schedule["workers"] == 12
    expected = Counter(
        (condition, seed, arm)
        for condition in ("sighted", "blind")
        for seed in (3101, 3102, 3103)
        for arm in ("ACT", "PACT", "PACT_PERMUTED")
        for _ in range(25)
    )
    assert Counter(
        (row["vision_condition"], row["checkpoint_seed"], row["arm"])
        for row in schedule["rows"]
    ) == expected
    for instance in range(25):
        rows = [row for row in schedule["rows"] if row["instance_index"] == instance]
        assert len(rows) == 18
        assert len({row["instance_episode_id"] for row in rows}) == 1
    smoke = schedule["rows"][0]
    assert (smoke["vision_condition"], smoke["checkpoint_seed"], smoke["arm"]) == (
        "blind",
        3101,
        "PACT_PERMUTED",
    )


def decision_fixture(*, contact: bool, task: bool, collapsed: bool) -> tuple[dict, dict]:
    schedule = {
        "collapse_floor_collision_free_success": 0.10,
        "collapse_floor_manipulation_success": 0.05,
    }
    contact_ci = [-100.0, -1.0] if contact else [-100.0, 10.0]
    analysis_doc = {
        "reconciliation": {"reconciled": True},
        "blind_arm_contrasts": {
            label: {
                "pooled": {
                    "hazard_bar_contact_frames": {
                        "instance_cluster_bootstrap_ci_95": contact_ci
                    },
                    "collision_free_task_success": {
                        "instance_cluster_bootstrap_ci_95": [0.01, 0.2]
                        if task
                        else [-0.1, 0.1]
                    },
                }
            }
            for label in ("PACT_minus_ACT", "PACT_minus_PACT_PERMUTED")
        },
        "absolute_performance": {
            "blind": {
                arm: {
                    "pooled": {
                        "collision_free_task_success": {"rate": 0.01 if collapsed else 0.2},
                        "task_success": {"rate": 0.01 if collapsed else 0.2},
                    }
                }
                for arm in analysis.ARMS
            }
        },
    }
    return schedule, analysis_doc


def test_decision_precedence_preserves_measurable_contact_under_task_collapse() -> None:
    schedule, document = decision_fixture(contact=True, task=False, collapsed=True)
    assert analysis.choose_decision(schedule, document)[0] == "PROXIMITY_STANDALONE_CONTACT_BENEFIT"
    schedule, document = decision_fixture(contact=True, task=True, collapsed=False)
    assert analysis.choose_decision(schedule, document)[0] == "PROXIMITY_STANDALONE_TASK_BENEFIT"
    schedule, document = decision_fixture(contact=False, task=False, collapsed=True)
    assert analysis.choose_decision(schedule, document)[0] == "BLIND_UNINFORMATIVE_COLLAPSE"
    schedule, document = decision_fixture(contact=False, task=False, collapsed=False)
    assert analysis.choose_decision(schedule, document)[0] == "NO_STANDALONE_BENEFIT"


def test_analysis_rejects_silently_dropped_blind_flag() -> None:
    row = build_schedule()["rows"][0]
    result = {
        "status": "complete",
        "rollout_id": row["rollout_id"],
        "schedule_row_sha256": row["schedule_row_sha256"],
        "episode_id": row["instance_episode_id"],
        "arm": row["arm"],
        "checkpoint_seed": row["checkpoint_seed"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "blind_rgb": False,
        "blur_sigma": 0.0,
    }
    try:
        analysis.validate_result(result, row)
    except ValueError as error:
        assert "blind_rgb" in str(error)
    else:
        raise AssertionError("mismatched blind flag was accepted")


def test_compactor_retains_blind_intervention_audit_fields() -> None:
    source = (ROOT / "scripts/compact_pact_contact_storage.py").read_text()
    assert source.count('"blind_rgb"') >= 2
    assert '"blur_diagnostic"' in source
    assert '"first_raw_proximity_sha256"' in source
    assert '"model_output_trace_sha256"' in source
