"""Tests for the RAW_HEAD_RESIDUAL_V1 qualification protocol.

Handoff step 16. Covers the controller definition, the shadow-equivalence design,
the frozen manifests and schedule, the retained MSAA/camera contract, and the
cluster-aware analysis schema.

No simulator environment is constructed, so this suite is fast and needs no GPU.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path("/root/prox_learning_hybrid_safety")
ACT = ROOT / "submodules/act"
MS = ROOT / "submodules/molmospaces"
DIAG = ROOT / "diagnostics_output/hybrid_obstacle_raw_head_qualification"
sys.path.insert(0, str(ACT))
sys.path.insert(0, str(ROOT / "scripts"))

DEV = ROOT / "configs/hybrid_obstacle_controller_development4_v1.json"
CONF = ROOT / "configs/hybrid_obstacle_confirmatory41_v1.json"
SCHED = ROOT / "configs/hybrid_obstacle_rawhead_schedule_v1.json"
SPLIT = ROOT / "configs/hybrid_obstacle_canonical_split_v2.json"
STACK = ROOT / "configs/hybrid_safety_stack_v1.json"
CKPT_MANIFEST = ROOT / "diagnostics_output/hybrid_obstacle_act_baseline/checkpoint_manifest.json"
RAWHEAD_EVAL = ACT / "eval_act_obstacle_rawhead.py"

EXPECT_CAMERA_CONFIG = "7e90b4db37b0037344e9a55b35e1d4d98b9e2025edab32e7132a7a434799cfa6"
EXPECT_SENSOR_ORDER = "c31df8c36b0011b0eaf5b2eb5ce66d2514b5d6662ba9d7684ff021cd17cec858"
EXPECT_MOLMOSPACES = "678f2eb4a0ac0d9e3d14e555aaac0e099089b9a5"


def sha(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canon(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def dev():
    return json.loads(DEV.read_text())


@pytest.fixture(scope="module")
def conf():
    return json.loads(CONF.read_text())


@pytest.fixture(scope="module")
def sched():
    return json.loads(SCHED.read_text())


@pytest.fixture(scope="module")
def rawhead_src():
    return RAWHEAD_EVAL.read_text()


# --------------------------------------------------------------------------- #
# controller definition
# --------------------------------------------------------------------------- #
def test_raw_head_mode_has_no_reference_tensor(rawhead_src):
    """RAW_HEAD must force the baseline to zeros, never load a reference episode."""
    assert "np.zeros(7, dtype=np.float32)" in rawhead_src
    assert 'self.condition == "ACT_PLUS_RAW_HEAD"' in rawhead_src
    # the only reference-H5 path is gated behind the privileged oracle condition
    tree = ast.parse(rawhead_src)
    src_lines = rawhead_src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "baseline_reference_h5":
            line = src_lines[node.lineno - 1]
            assert "oracle" in line.lower(), f"reference H5 assigned outside oracle path: {line}"


def test_scaling_applied_exactly_once():
    """SafetyHead multiplies by label_scale; the controller divides by it."""
    from hybrid_safety_residual import ResidualSafetyController
    meta = json.loads((ROOT / "assets/safety/cvae_v3/meta.json").read_text())
    scale = meta["label_scale"]
    ctrl = ResidualSafetyController(label_scale=scale, dt=0.066)
    raw = np.full(7, scale, dtype=np.float32)          # i.e. model output of 1.0
    step = ctrl.step(raw, np.zeros(7, dtype=np.float32))
    # (raw - 0) / scale == 1.0 exactly: one multiply, one divide
    assert np.allclose(step.subtracted_dq, np.ones(7), atol=1e-6)
    assert np.allclose(step.subtracted_dq_physical, raw, atol=1e-4)


def test_first_live_is_not_selectable(rawhead_src):
    import eval_act_obstacle_rawhead as m
    assert "first_live" not in [c.lower() for c in m.CONDITIONS]
    assert m.CONDITIONS == ("ACT_ONLY", "ACT_PLUS_RAW_HEAD", "ORACLE_PARKED_REFERENCE")
    assert '"first_live_skin_used": False' in rawhead_src


def test_oracle_mode_is_marked_privileged(rawhead_src, sched):
    import eval_act_obstacle_rawhead as m
    assert m.PRIVILEGED_CONDITIONS == frozenset({"ORACLE_PARKED_REFERENCE"})
    assert "ACT_PLUS_RAW_HEAD" not in m.PRIVILEGED_CONDITIONS
    assert sched["oracle_entry"]["privileged"] is True
    assert "privileged" in sched["oracle_entry"]["privileged_note"].lower()
    assert sched["oracle_entry"]["condition"] == "ORACLE_PARKED_REFERENCE"


def test_oracle_is_never_the_default(rawhead_src, sched):
    import eval_act_obstacle_rawhead as m
    assert m.RawHeadReplayPolicy.condition == "ACT_ONLY"
    assert all(e["condition"] != "ORACLE_PARKED_REFERENCE" for e in sched["entries"])
    assert all(e["privileged"] is False for e in sched["entries"])


def test_controller_id_and_status(rawhead_src):
    import eval_act_obstacle_rawhead as m
    assert m.CONTROLLER_ID == "RAW_HEAD_RESIDUAL_V1"
    assert "UNDER_QUALIFICATION" in m.CONTROLLER_STATUS
    assert "PROVEN" not in m.CONTROLLER_STATUS.upper()


# --------------------------------------------------------------------------- #
# shadow equivalence
# --------------------------------------------------------------------------- #
def test_shadow_zero_is_exact_for_the_frozen_controller():
    """A zero drive through the frozen dynamics must leave the action untouched."""
    from hybrid_safety_residual import ResidualSafetyController, apply_arm_residual
    ctrl = ResidualSafetyController(label_scale=11.359346389770508, dt=0.066)
    nominal = {"arm": np.linspace(-1, 1, 7).astype(np.float32),
               "gripper": np.array([255.0], np.float32)}
    for _ in range(250):
        step = ctrl.step(np.zeros(7, np.float32), np.zeros(7, np.float32))
        out = apply_arm_residual(nominal, step.correction)
        assert float(np.max(np.abs(out["arm"] - nominal["arm"]))) <= 1e-8
        assert np.array_equal(out["gripper"], nominal["gripper"])


def test_shadow_uses_a_throwaway_controller_not_the_real_one(rawhead_src):
    """The shadow path must never step the executing controller."""
    assert "_new_controller" in rawhead_src
    assert "_shadow_zero_controller" in rawhead_src
    assert "_shadow_raw_controller" in rawhead_src
    tree = ast.parse(rawhead_src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "get_action")
    stepped = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "step" and isinstance(node.func.value, ast.Attribute):
            stepped.add(node.func.value.attr)
    assert "_controller" not in stepped, (
        f"get_action steps the executing controller directly: {stepped}")
    assert stepped <= {"_shadow_zero_controller", "_shadow_raw_controller"}


def test_shadow_raw_head_is_not_executed(rawhead_src):
    assert '"executed": False' in rawhead_src
    assert "shadow_raw_head_action" in rawhead_src


# --------------------------------------------------------------------------- #
# residual semantics
# --------------------------------------------------------------------------- #
def test_residual_is_applied_after_temporal_aggregation(rawhead_src):
    """The adapter aggregates, then the residual is added; temp_agg stays on."""
    assert "pc.temp_agg_off = False" in rawhead_src
    adapter = (ACT / "eval_act_obstacle_safety.py").read_text()
    i_nom = adapter.index("nominal_action = self.model_output_to_action")
    i_res = adapter.index("executed_action = apply_arm_residual")
    assert i_nom < i_res, "residual must be applied after aggregation"


def test_residual_is_arm_only_and_gripper_bitwise_preserved():
    from hybrid_safety_residual import apply_arm_residual
    nominal = {"arm": np.zeros(7, np.float32), "gripper": np.array([137.0], np.float32)}
    out = apply_arm_residual(nominal, np.full(7, 0.2, np.float32))
    assert np.allclose(out["arm"], 0.2)
    assert np.array_equal(out["gripper"], nominal["gripper"])
    assert out["gripper"].dtype == nominal["gripper"].dtype


def test_controller_reset_clears_all_state():
    from hybrid_safety_residual import ResidualSafetyController
    ctrl = ResidualSafetyController(label_scale=1.0, dt=0.066)
    for _ in range(40):
        ctrl.step(np.full(7, 10.0, np.float32), np.zeros(7, np.float32))
    assert float(np.max(np.abs(ctrl.correction))) > 0
    ctrl.reset()
    assert float(np.max(np.abs(ctrl.correction))) == 0.0
    assert float(np.max(np.abs(ctrl.filtered_safety_dq))) == 0.0


def test_clipping_and_saturation_are_logged(rawhead_src):
    for key in ("correction_saturated", "correction_max_abs",
                "saturation_fraction_of_limit", "saturated_steps",
                "saturation_fraction_of_timesteps"):
        assert key in rawhead_src, f"{key} must be logged"


def test_frozen_constants_untouched():
    from hybrid_safety_residual import (
        DEFAULT_DECAY,
        DEFAULT_EMA,
        DEFAULT_GAIN,
        DEFAULT_MAX_DEVIATION,
    )
    assert (DEFAULT_GAIN, DEFAULT_DECAY, DEFAULT_EMA, DEFAULT_MAX_DEVIATION) == (
        4.0, 2.2, 0.75, 0.35)


# --------------------------------------------------------------------------- #
# manifests
# --------------------------------------------------------------------------- #
def test_development4_composition_and_self_hash(dev):
    d = dict(dev)
    stored = d.pop("manifest_sha256")
    assert canon(d) == stored
    assert dev["composition"] == {"total": 4, "hazard_present": 3, "hazard_absent": 1}
    assert len(dev["rows"]) == 4
    assert sum(r["hazard_present"] for r in dev["rows"]) == 3
    assert dev["role"] == "DEVELOPMENT_ONLY"


def test_confirmatory41_composition_and_self_hash(conf):
    d = dict(conf)
    stored = d.pop("manifest_sha256")
    assert canon(d) == stored
    assert conf["composition"] == {"total": 41, "hazard_present": 32, "hazard_absent": 9}
    assert len(conf["rows"]) == 41
    assert sum(r["hazard_present"] for r in conf["rows"]) == 32
    assert conf["role"] == "CONFIRMATORY_UNTOUCHED"
    assert conf["executed_in_this_task"] is False


def test_no_overlap_anywhere(dev, conf):
    split = json.loads(SPLIT.read_text())
    train_ids = {e["episode_id"] for e in split["episodes"] if e["split"] == "train"}
    val_ids = {e["episode_id"] for e in split["episodes"] if e["split"] == "validation"}
    dev_ids = {r["episode_id"] for r in dev["rows"]}
    conf_ids = {r["episode_id"] for r in conf["rows"]}
    assert not (dev_ids & conf_ids)
    assert not (dev_ids & train_ids)
    assert not (dev_ids & val_ids)
    assert not (conf_ids & train_ids)
    assert not (conf_ids & val_ids)
    assert len(dev_ids | conf_ids) == 45
    dev_h = {r["source_h5_sha256"] for r in dev["rows"]}
    conf_h = {r["source_h5_sha256"] for r in conf["rows"]}
    train_h = {e["source_h5_sha256"] for e in split["episodes"] if e["split"] == "train"}
    val_h = {e["source_h5_sha256"] for e in split["episodes"] if e["split"] == "validation"}
    assert not (dev_h & train_h) and not (dev_h & val_h)
    assert not (conf_h & train_h) and not (conf_h & val_h)


def test_manifest_rows_carry_replay_fields(dev, conf):
    required = {"candidate_index", "episode_id", "manifest_row_sha256", "hazard_present",
                "accepted_retry_index", "source_h5_sha256", "initial_state_sha256",
                "robot_initial_qpos", "target_uid", "obstacle_theta", "sensor_order_sha256"}
    for m in (dev, conf):
        for r in m["rows"]:
            assert required <= set(r), f"missing {required - set(r)}"
            assert r["sensor_order_sha256"] == EXPECT_SENSOR_ORDER


def test_evaluator_refuses_confirmatory_rows(rawhead_src):
    assert "refusing to execute a confirmatory row in this task" in rawhead_src
    assert 'CONFIRMATORY_UNTOUCHED' in rawhead_src


# --------------------------------------------------------------------------- #
# schedule
# --------------------------------------------------------------------------- #
def test_schedule_is_forty_primary_rollouts(sched):
    d = dict(sched)
    stored = d.pop("schedule_sha256")
    assert canon(d) == stored
    assert sched["primary_rollouts"] == 40 == sched["primary_rollout_budget"]
    assert len(sched["entries"]) == 40
    assert sched["repeats"] == 5
    assert sched["rows"] == 4
    assert sched["oracle_rollouts"] <= sched["oracle_rollout_budget"] == 1


def test_schedule_covers_every_row_condition_repeat(sched):
    seen = {(e["candidate_index"], e["condition"], e["repeat_index"]) for e in sched["entries"]}
    assert len(seen) == 40
    cands = {e["candidate_index"] for e in sched["entries"]}
    assert len(cands) == 4
    for c in cands:
        for cond in ("ACT_ONLY", "ACT_PLUS_RAW_HEAD"):
            reps = sorted(e["repeat_index"] for e in sched["entries"]
                          if e["candidate_index"] == c and e["condition"] == cond)
            assert reps == [0, 1, 2, 3, 4]


def test_condition_order_is_deterministically_balanced(sched):
    firsts = {}
    for e in sorted(sched["entries"], key=lambda x: x["execution_order"]):
        firsts.setdefault((e["candidate_index"], e["repeat_index"]), e["condition"])
    for (_c, rep), cond in firsts.items():
        expected = "ACT_ONLY" if rep % 2 == 0 else "ACT_PLUS_RAW_HEAD"
        assert cond == expected, f"repeat {rep} should start with {expected}"
    assert sched["order_balance"]["balanced"] is True


def test_schedule_output_dirs_are_unique(sched):
    dirs = [e["output_dir"] for e in sched["entries"]]
    assert len(set(dirs)) == len(dirs)


def test_schedule_initial_state_hashes_match_the_manifest(sched, dev):
    by_id = {r["episode_id"]: r["initial_state_sha256"] for r in dev["rows"]}
    for e in sched["entries"]:
        assert e["initial_state_sha256"] == by_id[e["episode_id"]]


# --------------------------------------------------------------------------- #
# contracts that must not move
# --------------------------------------------------------------------------- #
def test_msaa_remains_enabled_and_is_never_written(rawhead_src):
    assert "offsamples" in rawhead_src
    assert 'expected the retained contract value 4' in rawhead_src
    # the evaluator must only READ offsamples
    assert "offsamples =" not in rawhead_src.replace("offsamples = int(", "READ(")
    obs = json.loads((ROOT / "diagnostics_output/hybrid_obstacle_observation_reference"
                      "/final_decision.json").read_text())
    assert obs["blocker_1_wrist_determinism"]["offsamples"] == 4


def test_camera_and_sensor_contract_unchanged():
    assert sha(MS / "molmo_spaces/configs/camera_configs.py") == EXPECT_CAMERA_CONFIG
    stack = json.loads(STACK.read_text())
    assert stack["sensor_contract"]["sensor_order_hash"] == EXPECT_SENSOR_ORDER
    assert stack["sensor_contract"]["input_shape"] == [40, 8, 8]


def test_molmospaces_unmodified():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=MS, capture_output=True,
                          text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=MS, capture_output=True,
                            text=True, check=True).stdout.strip()
    assert head == EXPECT_MOLMOSPACES
    assert status == ""


def test_pinned_checkpoint_unchanged():
    pin = json.loads(CKPT_MANIFEST.read_text())
    ckpt = Path(pin["policy_best_ckpt"]["path"])
    assert sha(ckpt) == pin["policy_best_ckpt"]["sha256"]
    assert sha(ckpt.parent / "dataset_stats.pkl") == pin["dataset_stats_pkl_sha256"]
    assert pin["best_epoch"] == 1738


def test_audited_adapter_still_matches_its_base_commit():
    for name in ("eval_act_obstacle_safety.py", "hybrid_safety_residual.py"):
        at_base = subprocess.run(
            ["git", "show", f"3d25c69edd8d972afa59fec5c3edb9d13a357f92:{name}"],
            cwd=ACT, capture_output=True, check=True).stdout
        assert hashlib.sha256(at_base).hexdigest() == sha(ACT / name)


# --------------------------------------------------------------------------- #
# future analysis protocol
# --------------------------------------------------------------------------- #
def test_cluster_aware_analysis_schema_is_declared():
    p = DIAG / "future_statistical_protocol.json"
    if not p.is_file():
        pytest.skip("protocol not written yet")
    proto = json.loads(p.read_text())
    assert proto["resampling_unit"] == "manifest_row"
    assert proto["resampling_unit"] != "repeat"
    assert "mcnemar" in json.dumps(proto).lower()
    assert proto["mcnemar_as_sole_test"] is False
    for k in ("task_success", "collision_free_task_success", "hazard_collision_occurrence"):
        assert k in proto["binary_outcomes"]
    for k in ("minimum_hazard_clearance", "other_environment_collisions",
              "task_duration", "correction_magnitude"):
        assert k in proto["continuous_outcomes"]
    assert proto["strata"] == ["hazard_present", "hazard_absent"]
    assert proto["executed_in_this_task"] is False


def test_repeat_count_rule_is_predeclared():
    p = DIAG / "future_statistical_protocol.json"
    if not p.is_file():
        pytest.skip("protocol not written yet")
    rule = json.loads(p.read_text())["repeat_count_rule"]
    assert rule["0_unstable_rows"] == 3
    assert rule["1_or_2_unstable_rows"] == 5
    assert "STOCHASTICITY_TOO_HIGH_FOR_CONFIRMATORY_PROTOCOL" in rule["3_or_4_unstable_rows"]
