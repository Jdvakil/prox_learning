"""V10.9 conversion, split, evaluation-contract and evaluator-binding tests."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces",
           ROOT / "submodules" / "act"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v109_contract import (  # noqa: E402
    CANONICAL_SENSOR_NAMES,
    PACT_ONLY_FLAGS,
    SENSOR_ORDER_SHA256,
    SOLE_ROW_CELL,
    TRAIN_COUNT,
    VALIDATION_COUNT,
    canonical_row_order,
    command_diff,
    freeze_split,
    training_command,
)
from pact_place_v109_eval_contract import (  # noqa: E402
    DOUBLES_PER_FAMILY,
    DOUBLES_PER_SIDE,
    EXPECTED_PER_FAMILY,
    EXPECTED_PER_POSE,
    EXPECTED_PER_SIDE,
    SCENE_BY_POSE,
    doubled_cells,
    instance_plan,
)

WORK = ROOT / "diagnostics_output" / "pact_place_v109_train_eval"
EVAL = ROOT / "diagnostics_output" / "pact_place_v109_eval"


# --------------------------------------------------------------------------- #
# sensor order
# --------------------------------------------------------------------------- #
def test_sensor_order_hash_matches_the_v5_dataset():
    digest = hashlib.sha256(
        json.dumps(list(CANONICAL_SENSOR_NAMES), separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == SENSOR_ORDER_SHA256


def test_sensor_order_is_not_alphabetical():
    """Alphabetical sorting would put link5_back before link5_front and silently
    relabel every sensor slot the transformer embeds positionally."""
    names = list(CANONICAL_SENSOR_NAMES)
    assert names != sorted(names)
    assert names.index("link5_front_sensor_0") < names.index("link5_back_sensor_0")


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def source_rows():
    path = WORK / "source_manifest.json"
    if not path.is_file():
        pytest.skip("source manifest not built yet")
    document = json.loads(path.read_text())
    assert document["verified"]
    return document["rows"]


def test_split_is_113_28_and_covers_every_cell(source_rows):
    frozen = freeze_split(source_rows)
    labels = frozen["assignments"]
    assert sum(1 for v in labels.values() if v == "train") == TRAIN_COUNT
    assert sum(1 for v in labels.values() if v == "validation") == VALIDATION_COUNT
    assert len(frozen["train_cells"]) == 24
    assert len(frozen["validation_cells"]) == 23
    assert SOLE_ROW_CELL not in frozen["validation_cells"]


def test_split_does_not_depend_on_input_order(source_rows):
    import random

    shuffled = list(source_rows)
    random.Random(20260829).shuffle(shuffled)
    assert freeze_split(shuffled)["assignments"] == freeze_split(source_rows)["assignments"]


def test_every_cell_keeps_at_least_one_training_row(source_rows):
    labels = freeze_split(source_rows)["assignments"]
    by_cell: dict[str, int] = {}
    for row in source_rows:
        if labels[row["attempt_id"]] == "train":
            by_cell[row["cell"]] = by_cell.get(row["cell"], 0) + 1
    assert len(by_cell) == 24
    assert min(by_cell.values()) >= 1


def test_two_row_cell_splits_one_and_one(source_rows):
    labels = freeze_split(source_rows)["assignments"]
    members = [r for r in source_rows
               if r["cell"] == "F3_aperture_side_stagger|right|pos5"]
    assert len(members) == 2
    assert sorted(labels[r["attempt_id"]] for r in members) == ["train", "validation"]


def test_canonical_row_order_is_stable_and_total(source_rows):
    import random

    shuffled = list(source_rows)
    random.Random(7).shuffle(shuffled)
    assert ([r["attempt_id"] for r in canonical_row_order(shuffled)]
            == [r["attempt_id"] for r in canonical_row_order(source_rows)])


# --------------------------------------------------------------------------- #
# training commands
# --------------------------------------------------------------------------- #
def _commands():
    shared = dict(dataset_dir="/d", split_manifest="/s", dataset_manifest="/m",
                  expect_split_sha256="a", expect_dataset_tree_sha256="b")
    return (training_command(arm="act", ckpt_dir="/ckpt/act", **shared),
            training_command(arm="pact", ckpt_dir="/ckpt/pact", **shared))


def test_pact_differs_only_by_ckpt_dir_and_five_flags():
    act, pact = _commands()
    diff = command_diff(act, pact)
    assert diff["identical_except_allowance"], diff["violations"]
    assert set(diff["only_in_pact"]) == set(PACT_ONLY_FLAGS)
    assert set(diff["differing_values"]) == {"--ckpt_dir"}


def test_command_diff_fails_closed_on_an_extra_flag():
    act, pact = _commands()
    diff = command_diff(act, [*pact, "--lr_backbone", "1e-4"])
    assert not diff["identical_except_allowance"]


def test_command_diff_fails_closed_on_a_changed_shared_value():
    act, pact = _commands()
    mutated = list(pact)
    mutated[mutated.index("--seed") + 1] = "999"
    assert not command_diff(act, mutated)["identical_except_allowance"]


# --------------------------------------------------------------------------- #
# evaluation contract
# --------------------------------------------------------------------------- #
def test_doubled_cells_satisfy_every_balance_constraint():
    chosen = doubled_cells()
    assert len(chosen) == 16
    families: dict[str, int] = {}
    sides: dict[str, int] = {}
    poses: dict[str, int] = {}
    for key in chosen:
        family, side, pose = key.split("|")
        families[family] = families.get(family, 0) + 1
        sides[side] = sides.get(side, 0) + 1
        poses[pose] = poses.get(pose, 0) + 1
    assert set(families.values()) == {DOUBLES_PER_FAMILY}
    assert set(sides.values()) == {DOUBLES_PER_SIDE}
    assert poses == {"center": 6, "neg5": 5, "pos5": 5}


def test_doubled_cell_selection_is_deterministic():
    assert doubled_cells() == doubled_cells()


def test_instance_plan_balance():
    plan = instance_plan()
    assert len(plan) == 40
    families: dict[str, int] = {}
    sides: dict[str, int] = {}
    poses: dict[str, int] = {}
    for family, side, pose, _ in plan:
        families[family] = families.get(family, 0) + 1
        sides[side] = sides.get(side, 0) + 1
        poses[pose] = poses.get(pose, 0) + 1
    assert set(families.values()) == {EXPECTED_PER_FAMILY}
    assert set(sides.values()) == {EXPECTED_PER_SIDE}
    assert poses == EXPECTED_PER_POSE


def test_certified_scene_hashes_match_the_files_on_disk():
    for pose, entry in SCENE_BY_POSE.items():
        path = ROOT / entry["relative"]
        assert path.is_file(), f"{pose} scene missing"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


# --------------------------------------------------------------------------- #
# evaluator binding: no V2 scene or sampler fallback
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def eval_manifest():
    path = EVAL / "eval_manifest.json"
    if not path.is_file():
        pytest.skip("evaluation manifest not built yet")
    from pact_place_v109_eval_contract import load_manifest

    return load_manifest(path)


def test_every_row_names_the_v106_sampler(eval_manifest):
    for row in eval_manifest["rows"] + eval_manifest["smoke"]["rows"]:
        assert row["task_sampler_class"] == "PactPlaceCorridorV106Sampler"


def test_every_row_binds_its_own_pose_specific_certified_scene(eval_manifest):
    for row in eval_manifest["rows"] + eval_manifest["smoke"]["rows"]:
        relative = row["pact_v109_scene_relative"]
        assert row["pose_id"] in Path(relative).name
        assert "pact_place_corridor_v2" not in relative
        path = ROOT / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() \
            == row["pact_v106_scene_sha256"]
        assert row["pact_v106_scene_sha256"] == SCENE_BY_POSE[row["pose_id"]]["sha256"]


def test_evaluation_seeds_are_disjoint_from_every_prior_stream(eval_manifest):
    assert eval_manifest["held_out_seed_audit"]["disjoint"]
    assert not eval_manifest["held_out_seed_audit"]["collisions"]


def test_evaluator_refuses_a_foreign_sampler():
    module = pytest.importorskip("eval_pact_place_v109_row")
    with pytest.raises(ValueError):
        module.task_sampler_class_for(
            {"task_sampler_class": "PactPlaceCorridorV2Sampler"}, object)
    with pytest.raises(ValueError):
        module.task_sampler_class_for({}, object)


def test_evaluator_refuses_an_unbound_scene():
    module = pytest.importorskip("eval_pact_place_v109_row")
    module._ACTIVE_SCENE = None
    with pytest.raises(RuntimeError):
        module._active_scene()


def test_v109_bindings_survive_delegation_to_the_legacy_evaluator():
    """``place.main()`` reassigns its own module globals onto ``legacy`` as its
    last act before delegating. Patching ``legacy`` directly is therefore
    silently undone -- and the environment config reverts to the V2 place
    scene. This reproduces that delegation block and asserts the V10.9 objects
    are what survives."""
    v109 = pytest.importorskip("eval_pact_place_v109_row")
    place = pytest.importorskip("eval_pact_place_row")
    legacy = pytest.importorskip("eval_pact_collision_row")

    v109.install_v109_bindings()
    legacy.PactCollisionInferencePolicy = place.PactPlaceInferencePolicy
    legacy.PactCollisionPolicyConfig = place.policy_config_factory
    legacy.FrankaSkinPACTCollisionCorridorConfig = place.PactPlaceEvalConfig
    legacy.PactContactAudit = place.PactPlaceContactAudit
    legacy.load_eval_manifest = place.load_manifest
    legacy.retry_seed_for = place.retry_seed
    legacy.task_sampler_class_for = place.task_sampler_class_for

    surviving = v109.bindings_are_v109()
    assert all(surviving.values()), surviving
    with pytest.raises(ValueError):
        legacy.task_sampler_class_for(
            {"task_sampler_class": "PactPlaceCorridorV2Sampler"}, object)


def test_smoke_rows_resolve_through_the_evaluator():
    """The manifest keeps smoke rows under ``smoke.rows`` so the analysis can
    never count one as an evaluation instance, but the evaluator resolves an
    episode by scanning ``rows``. Both must hold at once."""
    v109 = pytest.importorskip("eval_pact_place_v109_row")
    path = EVAL / "eval_manifest.json"
    if not path.is_file():
        pytest.skip("evaluation manifest not built yet")
    merged = v109.load_manifest_all_rows(path)
    stored = v109.load_manifest(path)
    assert len(merged["rows"]) == len(stored["rows"]) + stored["smoke"]["instances"]
    assert len({r["episode_id"] for r in merged["rows"]}) == len(merged["rows"])
    assert len(stored["rows"]) == stored["total_candidates"] == 40
    for row in stored["smoke"]["rows"]:
        assert any(r["episode_id"] == row["episode_id"] for r in merged["rows"])
