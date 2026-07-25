"""Tests for the observation-determinism and Safety-CVAE reference findings.

Handoff step 11. These lock in the audit results so they cannot silently regress:

* the exact artifact/checkpoint hashes the paired stack is pinned to;
* the controller constants and label scaling;
* the provenance labels, including that the documented canonical reference is a
  privileged per-frame counterfactual and that ``first_live_skin`` is not canonical;
* that an oracle mode can never be relabelled deployable;
* the recorded root cause of the render nondeterminism (MSAA ``offsamples``);
* that MolmoSpaces camera/image semantics were not modified by this task.

No simulator environment is constructed here, so the suite is fast and needs no GPU.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path("/root/prox_learning_hybrid_safety")
DIAG = ROOT / "diagnostics_output/hybrid_obstacle_observation_reference"
ACT = ROOT / "submodules/act"
MS = ROOT / "submodules/molmospaces"
STACK = ROOT / "configs/hybrid_safety_stack_v1.json"
CKPT_MANIFEST = ROOT / "diagnostics_output/hybrid_obstacle_act_baseline/checkpoint_manifest.json"

ADAPTER_BASE = "3d25c69edd8d972afa59fec5c3edb9d13a357f92"
EXPECT_MOLMOSPACES = "678f2eb4a0ac0d9e3d14e555aaac0e099089b9a5"
EXPECT_SENSOR_ORDER = "c31df8c36b0011b0eaf5b2eb5ce66d2514b5d6662ba9d7684ff021cd17cec858"
EXPECT_CAMERA_CONFIG = "7e90b4db37b0037344e9a55b35e1d4d98b9e2025edab32e7132a7a434799cfa6"
EXPECT_MODEL_XML = "50924661e0411f92ab529c790512b17b674e789434c592c3dbc6d2359164d4c6"


def sha(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canon(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def provenance():
    return json.loads((DIAG / "reference_provenance.json").read_text())


@pytest.fixture(scope="module")
def classification():
    return json.loads((DIAG / "reference_contract_classification.json").read_text())


@pytest.fixture(scope="module")
def probe_matrix():
    return json.loads((DIAG / "wrist_divergence_probe_matrix.json").read_text())


@pytest.fixture(scope="module")
def attribution():
    return json.loads((DIAG / "one_pair_attribution.json").read_text())


# --------------------------------------------------------------------------- #
# artifact and source verification
# --------------------------------------------------------------------------- #
def test_pinned_checkpoint_and_statistics_hashes():
    pin = json.loads(CKPT_MANIFEST.read_text())
    ckpt = Path(pin["policy_best_ckpt"]["path"])
    assert sha(ckpt) == pin["policy_best_ckpt"]["sha256"]
    assert sha(ckpt.parent / "dataset_stats.pkl") == pin["dataset_stats_pkl_sha256"]
    assert pin["best_epoch"] == 1738


def test_checkpoint_manifest_self_hash():
    pin = json.loads(CKPT_MANIFEST.read_text())
    stored = pin.pop("checkpoint_manifest_sha256")
    assert canon(pin) == stored


def test_safety_cvae_pins_and_sensor_order():
    stack = json.loads(STACK.read_text())
    sc = stack["sensor_contract"]
    assert len(sc["ordered_names"]) == 40
    assert sc["input_shape"] == [40, 8, 8]
    assert sc["sensor_order_hash"] == EXPECT_SENSOR_ORDER
    assert hashlib.sha256(
        json.dumps(sc["ordered_names"], separators=(",", ":")).encode()
    ).hexdigest() == EXPECT_SENSOR_ORDER
    safe = ROOT / "assets/safety/cvae_v3"
    assert sha(safe / "model.pt") == stack["pinned_hashes"]["safety_model_sha256"]
    assert sha(safe / "meta.json") == stack["pinned_hashes"]["safety_meta_sha256"]


def test_camera_and_robot_semantics_unchanged():
    """This task must not have altered camera config or robot model."""
    assert sha(MS / "molmo_spaces/configs/camera_configs.py") == EXPECT_CAMERA_CONFIG
    assert sha(ROOT / "assets/robots/franka_skin/model_hybrid.xml") == EXPECT_MODEL_XML


def test_molmospaces_is_unmodified():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=MS,
                          capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=MS,
                            capture_output=True, text=True, check=True).stdout.strip()
    assert head == EXPECT_MOLMOSPACES
    assert status == "", f"MolmoSpaces worktree is dirty: {status}"


def test_act_adapter_files_still_match_the_base_commit():
    for name in ("eval_act_obstacle_safety.py", "hybrid_safety_residual.py",
                 "run_paired_hybrid_safety_eval.py", "tests/test_hybrid_safety_residual.py"):
        at_base = subprocess.run(["git", "show", f"{ADAPTER_BASE}:{name}"], cwd=ACT,
                                 capture_output=True, check=True).stdout
        assert hashlib.sha256(at_base).hexdigest() == sha(ACT / name)


# --------------------------------------------------------------------------- #
# controller constants and scaling
# --------------------------------------------------------------------------- #
def test_controller_constants_and_scaling_unchanged():
    import sys

    sys.path.insert(0, str(ACT))
    from hybrid_safety_residual import (
        DEFAULT_DECAY,
        DEFAULT_EMA,
        DEFAULT_GAIN,
        DEFAULT_MAX_DEVIATION,
    )
    assert (DEFAULT_GAIN, DEFAULT_DECAY, DEFAULT_EMA, DEFAULT_MAX_DEVIATION) == (
        4.0, 2.2, 0.75, 0.35)
    meta = json.loads((ROOT / "assets/safety/cvae_v3/meta.json").read_text())
    assert meta["label_scale"] == pytest.approx(11.359346389770508)
    assert [meta["n_in"], meta["n_out"]] == [40 * 8 * 8, 7]


# --------------------------------------------------------------------------- #
# reference provenance
# --------------------------------------------------------------------------- #
def test_first_live_skin_is_not_canonical(provenance):
    b = next(m for m in provenance["modes"] if m["id"] == "B")
    assert b["used_by_committed_demo"] is False
    assert b["merely_a_new_evaluation_choice"] is True
    assert b["used_to_train_the_safety_cvae"] is False
    assert b["frame_aligned_with_current_posture"] is False
    assert "not canonical" in provenance["headline_finding"].lower()


def test_documented_canonical_reference_is_the_parked_counterfactual(provenance):
    d = next(m for m in provenance["modes"] if m["id"] == "D")
    assert d["used_by_committed_demo"] is True
    assert d["requires_privileged_simulation_state"] is True
    assert d["frame_aligned_with_current_posture"] is True
    assert d["deployable_outside_simulation"] is False
    cites = " ".join(c["text"] for c in provenance["citations"])
    assert "obstacle parked" in cites
    assert "per-frame" in cites


def test_safety_cvae_was_trained_without_subtraction(provenance):
    trained = [m["id"] for m in provenance["modes"] if m["used_to_train_the_safety_cvae"]]
    assert trained == ["A"], (
        "only the raw no-subtraction mode matches Safety-CVAE training inputs")


def test_every_mode_is_classified(classification):
    labels = {c["controller"]: c["label"] for c in classification["classification"]}
    assert len(labels) == 5
    allowed = {"DEPLOYABLE", "SIMULATION_ORACLE", "HISTORICAL_DEMO_ONLY",
               "NOVEL_UNVALIDATED", "UNSUPPORTED"}
    assert set(labels.values()) <= allowed


def test_oracle_mode_cannot_be_mislabeled_deployable(classification, provenance):
    """Any mode requiring privileged state must not be labelled DEPLOYABLE."""
    privileged_ids = {m["id"] for m in provenance["modes"]
                      if m["requires_privileged_simulation_state"]}
    assert privileged_ids, "expected at least one privileged mode"
    deployable = [c for c in classification["classification"] if c["label"] == "DEPLOYABLE"]
    for c in deployable:
        assert "parked" not in c["controller"].lower()
        assert "counterfactual" not in c["controller"].lower()
        assert "recorded" not in c["controller"].lower()
    oracle = next(c for c in classification["classification"]
                  if c["label"] == "SIMULATION_ORACLE")
    assert "privileged" in oracle["caveat"].lower() or "privileged" in oracle["why"].lower()


def test_next_task_options_are_stated_and_no_choice_is_made(classification):
    opts = {o["option"] for o in classification["options_for_the_next_approved_task"]}
    assert opts == {"A", "B", "C", "D"}
    assert "withheld" in classification["recommendation_withheld"].lower() or \
           "not chosen" in classification["recommendation_withheld"].lower()


# --------------------------------------------------------------------------- #
# render nondeterminism root cause
# --------------------------------------------------------------------------- #
def test_root_cause_is_recorded_as_multisampling(probe_matrix):
    assert "offsamples" in probe_matrix["exact_causal_defect"]
    l = probe_matrix["probe_L_multisample_hypothesis"]
    assert l["verdict"] == "CONFIRMED"
    assert l["offsamples"] == 4
    assert l["wrist_msaa_as_configured"]["all_identical"] is False
    assert l["wrist_msaa_disabled"]["all_identical"] is True


def test_state_and_cache_causes_are_ruled_out(probe_matrix):
    loc = probe_matrix["first_divergence_localization"]
    for key in ("simulator_state_restoration", "derived_mujoco_kinematics",
                "robot_mounted_camera_pose_update", "observation_cache_reuse",
                "cache_key_collision", "renderer_context_initialization", "render_order"):
        assert loc[key].startswith("RULED OUT"), f"{key} should be ruled out"


def test_scene_camera_hypothesis_was_tested_and_refuted(probe_matrix):
    k = probe_matrix["probe_K_scene_camera_hypothesis"]
    assert k["verdict"] == "REFUTED"
    assert k["wrist_configured_camera_distinct"] > 1


def test_no_speculative_fix_was_applied(probe_matrix):
    fa = probe_matrix["fix_assessment"]
    assert fa["no_narrow_fix_applied"] is True
    assert "offsamples = 0" in fa["the_only_known_route_to_determinism"]
    assert "train/eval" in fa["why_that_route_is_not_taken"]


def test_probe_budget_respected(probe_matrix):
    used = probe_matrix["environment_constructions_used"]
    assert used["total"] <= used["budget"] == 20


def test_exo_correction_is_recorded(probe_matrix):
    """The earlier belief that exo was deterministic must be corrected, not left standing."""
    assert "incidental" in probe_matrix["correction_to_earlier_inference"]
    l = probe_matrix["probe_L_multisample_hypothesis"]
    assert l["exo_msaa_as_configured"]["all_identical"] is False


# --------------------------------------------------------------------------- #
# one-pair attribution
# --------------------------------------------------------------------------- #
def test_saturation_attributed_to_posture_drift_not_hazard(attribution):
    cc = attribution["candidate_causes"]
    assert cc["posture_drift_from_the_frozen_reference"]["primary"] is True
    assert cc["actual_hazard_proximity"]["supported"] is False
    assert cc["changing_self_returns"]["supported"] is False
    assert attribution["reference"]["reference_head_norm_constant"] is True


def test_known_self_return_sensors_reported_separately(attribution):
    ks = attribution["known_self_return_sensors"]
    assert ks["sensors"] == ["link5_front_sensor_1", "link5_front_sensor_2"]
    assert ks["frames_active"] == 0
    assert ks["dominate_raw_or_subtracted_output"] is False


def test_attribution_makes_no_effect_size_claim(attribution):
    assert "one episode" in attribution["caveat"].lower()
    assert attribution["explicitly_not_done"]
