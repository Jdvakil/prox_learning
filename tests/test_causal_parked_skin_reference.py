"""Contract tests for the CAUSAL_PARKED_SKIN_REFERENCE_V1 offline learnability task.

These check the things that would silently invalidate the result rather than crash it:
a history window that reaches forward, a privileged field leaking into model input, a
SafetyHead that quietly trains, a threshold fitted on the partition it is scored on.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "submodules" / "act"))

from causal_parked_skin import data as cps_data
from causal_parked_skin import gates as cps_gates
from causal_parked_skin import metrics as cps_metrics

DIAG = ROOT / "diagnostics_output" / "causal_parked_skin_reference_v1"
DATASET_DECISION = (ROOT / "diagnostics_output" / "hybrid_obstacle_parked_skin_dataset"
                    / "final_decision.json")
MANIFEST = ROOT / "configs" / "hybrid_obstacle_parked_skin_supervision_v1.json"
STACK = ROOT / "configs" / "hybrid_safety_stack_v1.json"
SAFETY_DIR = ROOT / "assets" / "safety" / "cvae_v3"


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def manifest() -> dict:
    return _json(MANIFEST)


@pytest.fixture(scope="module")
def dataset_decision() -> dict:
    return _json(DATASET_DECISION)


@pytest.fixture(scope="module")
def sensor_names() -> list:
    return list(_json(STACK)["sensor_contract"]["ordered_names"])


@pytest.fixture(scope="module")
def sample_files(manifest) -> list:
    """A few real trajectories, one per source distribution."""
    import h5py

    picked = []
    for distribution in cps_data.SOURCE_MODES:
        entry = next((e for e in manifest["entries"]
                      if e["distribution"] == distribution), None)
        if entry is not None:
            picked.append(entry)
    loaded = []
    for entry in picked:
        with h5py.File(entry["output"], "r") as handle:
            loaded.append({
                "entry": entry,
                "attrs": {k: handle.attrs[k] for k in handle.attrs},
                "current": handle["deployable/current_closeness"][:],
                "parked": handle["privileged/parked_closeness"][:],
                "changed": handle["privileged/changed_pixel_mask"][:],
                "current_head": handle["privileged/current_head"][:],
                "parked_head": handle["privileged/parked_head"][:],
                "oracle_dq": handle["privileged/oracle_dq"][:],
                "oracle_active": handle["privileged/oracle_active"][:],
            })
    return loaded


# --------------------------------------------------------------- 1. dataset integrity
def test_frozen_dataset_tree_hash_matches_decision(manifest, dataset_decision):
    """Every file still hashes to what the freeze recorded."""
    files = []
    for entry in manifest["entries"]:
        digest = hashlib.sha256(Path(entry["output"]).read_bytes()).hexdigest()
        files.append({"distribution": entry["distribution"],
                      "episode_id": entry["episode_id"], "file_sha256": digest})
    tree = _canonical_hash(sorted(files, key=lambda f: (f["distribution"],
                                                        f["episode_id"])))
    assert tree == dataset_decision["dataset"]["tree_sha256"]
    assert len(files) == 364


def test_source_files_are_read_only(manifest):
    """Mode bits stay 444.

    Asserting the bits rather than attempting a write: this test runs as root, and root
    bypasses the permission check, so a successful write would prove nothing.
    """
    writable = [e["output"] for e in manifest["entries"][:40]
                if stat.S_IMODE(os.stat(e["output"]).st_mode) & 0o222]
    assert writable == []


def test_loader_opens_dataset_read_only():
    source = (ROOT / "causal_parked_skin" / "data.py").read_text()
    tree = ast.parse(source)
    modes = [node.args[1].value for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "File"
             and len(node.args) > 1 and isinstance(node.args[1], ast.Constant)]
    assert modes, "no h5py.File call found"
    assert set(modes) == {"r"}


# ------------------------------------------------------------ 2. partition independence
def test_partition_independence_report_clean():
    report = _json(DIAG / "partition_independence.json")
    assert report["total_crossings"] == 0
    assert report["valid"] is True
    assert set(report["identity_keys_checked"]) == {
        "episode_id", "source_h5_sha256", "manifest_row_id", "trajectory_id",
        "initial_state_sha256"}


def test_partition_frame_counts_match_frozen_dataset():
    report = _json(DIAG / "partition_independence.json")
    expected = {"reference_train": 43519, "reference_validation": 3910,
                "reference_calibration": 3821, "offline_reference_test": 9543}
    for name, frames in expected.items():
        assert report["composition"][name]["frames"] == frames


# ------------------------------------------------------------------ 3. causal history
def test_causal_history_is_exactly_four_left_padded_frames():
    idx = cps_data.causal_history_indices(6)
    assert idx.shape == (6, 4)
    assert idx[0].tolist() == [0, 0, 0, 0]
    assert idx[1].tolist() == [0, 0, 0, 1]
    assert idx[3].tolist() == [0, 1, 2, 3]
    assert idx[5].tolist() == [2, 3, 4, 5]


def test_history_never_reaches_a_future_frame():
    for length in (1, 2, 4, 7, 200):
        idx = cps_data.causal_history_indices(length)
        steps = np.arange(length)[:, None]
        assert (idx <= steps).all(), "a window reached forward in time"
        assert (idx >= 0).all()
        assert (idx[:, -1] == np.arange(length)).all(), "last frame must be t"


def test_history_offset_does_not_cross_trajectory_boundary():
    first = cps_data.causal_history_indices(5, offset=0)
    second = cps_data.causal_history_indices(5, offset=5)
    assert first.max() < 5
    assert second.min() >= 5


# ------------------------------------------------------------- 4. deployable-only input
def test_model_forward_consumes_only_deployable_tensors():
    from causal_parked_skin.model import CausalParkedSkinReferenceV1

    signature = ast.parse(
        (ROOT / "causal_parked_skin" / "model.py").read_text())
    forwards = [n for n in ast.walk(signature)
                if isinstance(n, ast.FunctionDef) and n.name == "forward"
                and any(a.arg == "history" for a in n.args.args)]
    assert forwards, "no forward(history, ...) found"
    names = {a.arg for a in forwards[0].args.args} - {"self"}
    assert names == {"history", "history_valid", "state"}
    assert CausalParkedSkinReferenceV1 is not None


def test_batch_keys_separate_deployable_from_privileged():
    assert set(cps_data.DEPLOYABLE_INPUTS).isdisjoint(cps_data.PRIVILEGED_TARGETS)
    for prohibited in ("parked", "oracle_dq", "changed", "parked_head"):
        assert prohibited not in cps_data.DEPLOYABLE_INPUTS


def test_input_contract_audit_declares_no_prohibited_inputs():
    audit = _json(DIAG / "input_contract_audit.json")
    assert audit["valid"] is True
    assert audit["prohibited_inputs_used"] == []
    assert audit["inputs_not_live_available"] == []
    inputs = set(audit["contract"]["model_input_fields"])
    assert "parked_closeness" not in inputs
    assert "oracle_dq" not in inputs
    assert "current_head" not in inputs


# --------------------------------------------------------- 5. output bounds/monotonicity
@pytest.fixture(scope="module")
def torch_model(sensor_names):
    import torch

    from causal_parked_skin.model import build_model

    link_ids, links = cps_data.sensor_link_ids(sensor_names)
    model = build_model("FULL_CAUSAL", hidden=64, blocks=1,
                        link_ids=torch.from_numpy(link_ids), link_count=len(links))
    return model


def test_output_bounds_hold_under_adversarial_logits(torch_model):
    import torch

    generator = torch.Generator().manual_seed(0)
    history = torch.rand(8, 4, 40, 8, 8, generator=generator)
    valid = torch.ones_like(history, dtype=torch.bool)
    state = torch.randn(8, 29, generator=generator)
    for bias in (-60.0, -5.0, 0.0, 5.0, 60.0):
        with torch.no_grad():
            torch_model.decoder.out.bias.fill_(bias)
            out = torch_model(history, valid, state)
        parked, current = out["parked"], out["current"]
        assert torch.isfinite(parked).all()
        assert (parked >= -1e-7).all(), "parked went below zero"
        assert (parked <= current + 1e-7).all(), "parked exceeded current"
        assert (parked <= 1.0 + 1e-7).all()


def test_predicted_delta_is_monotone_in_mask_logit():
    import torch

    from causal_parked_skin.model import constrained_parked

    current = torch.full((4, 40, 8, 8), 0.5)
    magnitude = torch.zeros_like(current)
    previous = None
    for logit in (-5.0, -1.0, 0.0, 1.0, 5.0):
        _, delta, _ = constrained_parked(current, torch.full_like(current, logit),
                                         magnitude)
        value = float(delta.mean())
        if previous is not None:
            assert value >= previous - 1e-9, "delta must not shrink as mask logit grows"
        previous = value


def test_zero_differential_baseline_predicts_current_exactly():
    import torch

    from causal_parked_skin.model import zero_differential

    history = torch.rand(6, 4, 40, 8, 8)
    out = zero_differential(history)
    assert torch.equal(out["parked"], history[:, -1])
    assert float(out["delta"].abs().max()) == 0.0


# ------------------------------------------------------- 6. hazard-absent exact control
def test_hazard_absent_rows_are_bitwise_exact_controls(manifest):
    import h5py

    entry = next(e for e in manifest["entries"] if not e["hazard_present"])
    with h5py.File(entry["output"], "r") as handle:
        current = handle["deployable/current_closeness"][:]
        parked = handle["privileged/parked_closeness"][:]
        changed = handle["privileged/changed_pixel_mask"][:]
        oracle = handle["privileged/oracle_dq"][:]
    assert np.array_equal(current, parked)
    assert not changed.any()
    assert float(np.abs(oracle).max()) == 0.0


def test_physical_inequality_holds_on_sampled_files(sample_files):
    for block in sample_files:
        assert (block["parked"] <= block["current"] + 1e-7).all()
        assert (block["parked"] >= 0.0).all()
        assert (block["current"] <= 1.0).all()


# --------------------------------------------------------------- 7. oracle-zero retention
def test_all_oracle_zero_frames_are_retained(dataset_decision):
    counts = dataset_decision["dataset"]["counts"]
    assert counts["oracle_zero_frames"] == 46382
    assert counts["total_frames"] == 60793
    natural = dataset_decision["dataset"]["natural_distribution_retained"]
    assert natural["zero_frames_subsampled"] is False
    assert natural["active_zero_balancing_at_generation"] is False


def test_sampler_does_not_remove_zero_frames_from_the_partition():
    report = _json(DIAG / "selection.json")
    sampler = next(iter(report["results"].values()))["sampler"]
    assert sampler["zero_frames_removed"] is False
    assert sampler["zero_frames_subsampled_from_dataset"] is False
    assert sampler["evaluation_uses_natural_distribution"] is True
    assert sampler["natural_active_prevalence"] != sampler["sampled_active_prevalence"]


# ------------------------------------------------------------ 8. deterministic sampler
class _FakePartition:
    def __init__(self, frames: int = 400) -> None:
        rng = np.random.default_rng(0)
        self.name = "fake"
        self.arrays = {
            "oracle_active": rng.random(frames) < 0.25,
            "hazard_present": rng.random(frames) < 0.7,
            "source_mode": rng.integers(0, 4, frames).astype(np.int8),
        }

    def __len__(self) -> int:
        return len(self.arrays["oracle_active"])

    def __getitem__(self, key):
        return self.arrays[key]


def test_stratified_sampler_is_deterministic_for_a_seed_and_epoch():
    partition = _FakePartition()
    a = cps_data.StratifiedBatchSampler(partition, 32, seed=7, batches_per_epoch=5)
    b = cps_data.StratifiedBatchSampler(partition, 32, seed=7, batches_per_epoch=5)
    for left, right in zip(a.epoch(3), b.epoch(3)):
        assert np.array_equal(left, right)


def test_stratified_sampler_differs_across_epochs_and_seeds():
    partition = _FakePartition()
    sampler = cps_data.StratifiedBatchSampler(partition, 32, seed=7, batches_per_epoch=5)
    other = cps_data.StratifiedBatchSampler(partition, 32, seed=8, batches_per_epoch=5)
    assert not np.array_equal(sampler.epoch(0)[0], sampler.epoch(1)[0])
    assert not np.array_equal(sampler.epoch(0)[0], other.epoch(0)[0])


def test_sampler_raises_rather_than_silently_unbalancing():
    partition = _FakePartition()
    with pytest.raises(ValueError):
        cps_data.StratifiedBatchSampler(partition, 32, active_fraction=0.0)
    with pytest.raises(ValueError):
        cps_data.StratifiedBatchSampler(partition, 32, active_fraction=1.0)


# ---------------------------------------------------------------- 9. frozen SafetyHead
@pytest.fixture(scope="module")
def frozen_head():
    from causal_parked_skin.model import FrozenSafetyHead

    return FrozenSafetyHead.load(SAFETY_DIR, device="cpu")


def test_safety_head_parameters_are_frozen(frozen_head):
    assert frozen_head.frozen()
    assert all(not p.requires_grad for p in frozen_head.decoder.parameters())


def test_safety_head_stays_in_eval_even_when_train_is_called(frozen_head):
    frozen_head.train(True)
    assert frozen_head.training is False
    assert frozen_head.decoder.training is False


def test_gradient_reaches_input_but_not_safety_head_parameters(frozen_head):
    import torch

    field = torch.rand(3, 40, 8, 8, requires_grad=True)
    frozen_head(field).sum().backward()
    assert field.grad is not None and torch.isfinite(field.grad).all()
    assert all(p.grad is None for p in frozen_head.decoder.parameters())


def test_safety_head_weights_unchanged_after_a_backward_pass(frozen_head):
    import torch

    before = [p.detach().clone() for p in frozen_head.decoder.parameters()]
    field = torch.rand(3, 40, 8, 8, requires_grad=True)
    frozen_head(field).pow(2).sum().backward()
    for original, current in zip(before, frozen_head.decoder.parameters()):
        assert torch.equal(original, current)


def test_label_scale_applied_exactly_once(frozen_head):
    import torch

    meta = _json(SAFETY_DIR / "meta.json")
    field = torch.rand(2, 40, 8, 8)
    flat = field.reshape(2, -1)
    latent = torch.zeros(2, meta["z_dim"])
    with torch.no_grad():
        raw = frozen_head.decoder(torch.cat([flat, latent], dim=-1))
        produced = frozen_head(field)
    expected = raw * meta["label_scale"]
    assert torch.allclose(produced, expected, atol=0, rtol=0)
    # applying it twice would be off by the scale factor, which is far from 1
    assert not torch.allclose(produced, raw * meta["label_scale"] ** 2)


def test_frozen_head_reproduces_stored_targets(sample_files, frozen_head):
    """head(current) - head(parked) must return the stored oracle differential."""
    import torch

    for block in sample_files:
        with torch.no_grad():
            current = frozen_head(torch.from_numpy(block["current"][:32]))
            parked = frozen_head(torch.from_numpy(block["parked"][:32]))
        assert np.abs(current.numpy() - block["current_head"][:32]).max() < 1e-4
        assert np.abs(parked.numpy() - block["parked_head"][:32]).max() < 1e-4
        assert np.abs((current - parked).numpy()
                      - block["oracle_dq"][:32]).max() < 1e-4


def test_current_minus_parked_equals_stored_oracle_dq(sample_files):
    for block in sample_files:
        reconstructed = block["current_head"] - block["parked_head"]
        assert np.abs(reconstructed - block["oracle_dq"]).max() == 0.0


# --------------------------------------------------------------------- 10. baselines
def test_three_learned_baselines_differ_in_what_they_consume(sensor_names):
    import torch

    from causal_parked_skin.model import build_model

    link_ids, links = cps_data.sensor_link_ids(sensor_names)
    built = {name: build_model(name, hidden=64, blocks=1,
                               link_ids=torch.from_numpy(link_ids), link_count=len(links))
             for name in ("FULL_CAUSAL", "CURRENT_FRAME_ONLY", "QPOS_ONLY")}
    assert built["FULL_CAUSAL"].history_frames == 4
    assert built["CURRENT_FRAME_ONLY"].history_frames == 1
    assert built["QPOS_ONLY"].use_proximity is False
    assert built["FULL_CAUSAL"].use_proximity is True
    for model in built.values():
        assert model.parameter_count() < 3_000_000


def test_qpos_only_logits_ignore_the_current_frame_too(sensor_names):
    """The state-only control must not see proximity at all, including frame t.

    Checking only the earlier frames would pass even if the decoder were fed the current
    field directly, which would make this baseline a slightly worse CURRENT_FRAME_ONLY
    instead of a control. The predicted parked field still varies with the current field
    through the shared output constraint, so the invariant has to be asserted on the
    logits rather than on the prediction.
    """
    import torch

    from causal_parked_skin.model import build_model

    link_ids, links = cps_data.sensor_link_ids(sensor_names)
    model = build_model("QPOS_ONLY", hidden=64, blocks=1,
                        link_ids=torch.from_numpy(link_ids), link_count=len(links)).eval()
    torch.nn.init.normal_(model.decoder.out.weight, std=0.1)
    generator = torch.Generator().manual_seed(17)
    history = torch.rand(4, 4, 40, 8, 8, generator=generator)
    valid = torch.ones_like(history, dtype=torch.bool)
    state = torch.randn(4, 29, generator=generator)
    other = torch.rand(4, 4, 40, 8, 8, generator=generator)
    with torch.no_grad():
        a = model(history, valid, state)["mask_logits"]
        b = model(other, valid, state)["mask_logits"]
    assert torch.equal(a, b), "state-only logits moved when the proximity field changed"


def test_qpos_only_still_responds_to_robot_state(sensor_names):
    import torch

    from causal_parked_skin.model import build_model

    link_ids, links = cps_data.sensor_link_ids(sensor_names)
    model = build_model("QPOS_ONLY", hidden=64, blocks=1,
                        link_ids=torch.from_numpy(link_ids), link_count=len(links)).eval()
    torch.nn.init.normal_(model.decoder.out.weight, std=0.1)
    generator = torch.Generator().manual_seed(19)
    history = torch.rand(4, 4, 40, 8, 8, generator=generator)
    valid = torch.ones_like(history, dtype=torch.bool)
    with torch.no_grad():
        a = model(history, valid, torch.randn(4, 29, generator=generator))["mask_logits"]
        b = model(history, valid, torch.randn(4, 29, generator=generator))["mask_logits"]
    assert not torch.equal(a, b), "state-only model ignores its only input"


def test_qpos_only_ignores_the_proximity_history(sensor_names):
    """Changing earlier frames must not move a state-only model's prediction."""
    import torch

    from causal_parked_skin.model import build_model

    link_ids, links = cps_data.sensor_link_ids(sensor_names)
    model = build_model("QPOS_ONLY", hidden=64, blocks=1,
                        link_ids=torch.from_numpy(link_ids), link_count=len(links)).eval()
    generator = torch.Generator().manual_seed(3)
    history = torch.rand(4, 4, 40, 8, 8, generator=generator)
    valid = torch.ones_like(history, dtype=torch.bool)
    state = torch.randn(4, 29, generator=generator)
    altered = history.clone()
    altered[:, :3] = torch.rand(4, 3, 40, 8, 8, generator=generator)
    with torch.no_grad():
        a = model(history, valid, state)["parked"]
        b = model(altered, valid, state)["parked"]
    assert torch.equal(a, b)


def test_full_causal_actually_uses_the_earlier_frames(sensor_names):
    import torch

    from causal_parked_skin.model import build_model

    link_ids, links = cps_data.sensor_link_ids(sensor_names)
    model = build_model("FULL_CAUSAL", hidden=64, blocks=1,
                        link_ids=torch.from_numpy(link_ids), link_count=len(links)).eval()
    # a zeroed decoder would make every input look identical; give it real weights
    torch.nn.init.normal_(model.decoder.out.weight, std=0.1)
    generator = torch.Generator().manual_seed(5)
    history = torch.rand(4, 4, 40, 8, 8, generator=generator)
    valid = torch.ones_like(history, dtype=torch.bool)
    state = torch.randn(4, 29, generator=generator)
    altered = history.clone()
    altered[:, :3] = torch.rand(4, 3, 40, 8, 8, generator=generator)
    with torch.no_grad():
        a = model(history, valid, state)["parked"]
        b = model(altered, valid, state)["parked"]
    assert not torch.equal(a, b)


# ------------------------------------------------------- 11. calibration / test locking
def test_calibration_threshold_is_computed_from_zero_frames_only():
    from causal_parked_skin_train_final import calibrate_threshold

    rng = np.random.default_rng(0)
    active = np.zeros(1000, dtype=bool)
    active[:200] = True
    norms = np.where(active, rng.uniform(1.0, 2.0, 1000), rng.uniform(0.0, 0.1, 1000))
    result = calibrate_threshold({"predicted_norm": norms, "oracle_active": active})
    assert result["selected_on"] == "reference_calibration"
    assert result["offline_test_used"] is False
    assert result["achieved_false_positive_rate_on_calibration"] <= 0.01 + 1e-9
    # the threshold must sit inside the zero-frame range, not be dragged up by actives
    assert result["threshold"] < 1.0


def test_calibration_threshold_ignores_active_frame_magnitudes():
    from causal_parked_skin_train_final import calibrate_threshold

    active = np.zeros(500, dtype=bool)
    active[:100] = True
    zeros = np.linspace(0.0, 0.5, 400)
    small = np.concatenate([np.full(100, 1.0), zeros])
    huge = np.concatenate([np.full(100, 500.0), zeros])
    a = calibrate_threshold({"predicted_norm": small, "oracle_active": active})
    b = calibrate_threshold({"predicted_norm": huge, "oracle_active": active})
    assert a["threshold"] == b["threshold"]


def test_selection_script_never_reads_offline_test():
    source = (ROOT / "scripts" / "causal_parked_skin_select.py").read_text()
    tree = ast.parse(source)
    loads = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "load_partition"]
    named = [a.value for call in loads for a in call.args
             if isinstance(a, ast.Constant)]
    assert "offline_reference_test" not in named
    assert set(named) <= {"reference_train", "reference_validation"}


def test_selection_report_declares_offline_test_untouched():
    report = _json(DIAG / "selection.json")
    assert report["offline_test_loaded"] is False
    assert report["selection_partition"] == "reference_validation"
    assert report["candidates_run"] <= report["candidate_budget"] == 6


def test_selection_varied_only_permitted_axes():
    report = _json(DIAG / "selection.json")
    assert set(report["varied_axes"]) <= {"loss_component_weights", "cross_sensor_blocks",
                                          "hidden_width"}
    configs = [r["config"] for r in report["results"].values()]
    for field in ("variant", "batch_size", "active_fraction", "max_epochs", "patience"):
        assert len({c[field] for c in configs}) == 1, f"{field} varied across candidates"


# ------------------------------------------------------------- 12. checkpoints / resume
def test_checkpoint_save_is_atomic_and_reloads(tmp_path):
    import torch

    from causal_parked_skin.engine import atomic_save

    payload = {"model": {"w": torch.arange(5.0)}, "epoch": 3}
    path = tmp_path / "ckpt.pt"
    digest = atomic_save(payload, path)
    assert path.exists()
    assert not list(tmp_path.glob("*.partial")), "temp file survived"
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    reloaded = torch.load(path, weights_only=False)
    assert torch.equal(reloaded["model"]["w"], payload["model"]["w"])
    assert reloaded["epoch"] == 3


def test_rng_bundle_round_trips():
    import torch

    from causal_parked_skin.engine import restore_rng, rng_bundle

    torch.manual_seed(11)
    np.random.seed(11)
    bundle = rng_bundle()
    first = (torch.randn(4), np.random.random(4))
    restore_rng(bundle)
    second = (torch.randn(4), np.random.random(4))
    assert torch.equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


# ------------------------------------------------------------------- 13. metric shapes
def test_average_precision_matches_hand_computed_case():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([1, 0, 1, 0])
    # precision at the two hits: 1/1 and 2/3
    assert cps_metrics.average_precision(scores, labels) == pytest.approx(
        (1.0 + 2.0 / 3.0) / 2.0)


def test_direction_cosine_is_one_for_parallel_vectors():
    a = np.array([[1.0, 2.0, 3.0, 0, 0, 0, 0]])
    assert cps_metrics.direction_cosine(a, a * 2.5)[0] == pytest.approx(1.0)
    assert cps_metrics.direction_cosine(a, -a)[0] == pytest.approx(-1.0)


def test_constraint_violations_counts_rather_than_clamps():
    current = np.full((2, 4), 0.5)
    parked = np.array([[0.4, 0.6, -0.1, 0.5], [0.5, 0.5, 0.5, 0.5]])
    counted = cps_metrics.constraint_violations(parked, current)
    assert counted["parked_above_current"] == 1
    assert counted["parked_below_zero"] == 1
    assert counted["total"] == 2


def test_seed_summary_reports_coefficient_of_variation():
    summary = cps_metrics.summarize_seeds([1.0, 1.1, 0.9])
    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(1.0)
    assert summary["coefficient_of_variation"] == pytest.approx(
        np.std([1.0, 1.1, 0.9]) / 1.0)


# --------------------------------------------------------------- 14. per-source metrics
def test_final_report_carries_every_source_mode_block():
    report = _json(DIAG / "final_training.json")
    for run in report["runs"].values():
        for partition, block in run["metrics"].items():
            assert set(block["by_source_mode"]) == set(cps_data.SOURCE_MODES), partition
            for entry in block["by_source_mode"].values():
                if entry["available"]:
                    assert entry["frames"] > 0
                else:
                    assert entry["frames"] == 0


def test_learner_induced_absence_from_test_is_recorded_not_hidden():
    """The learner-induced mode exists only in train; the report must say so."""
    partition_report = _json(DIAG / "partition_independence.json")
    modes = partition_report["composition"]["offline_reference_test"][
        "source_mode_trajectories"]
    assert "LEARNER_INDUCED_ON_POLICY" not in modes
    final = _json(DIAG / "final_training.json")
    block = next(iter(final["runs"].values()))["metrics"]["offline_reference_test"]
    assert block["by_source_mode"]["LEARNER_INDUCED_ON_POLICY"]["available"] is False


# ----------------------------------------------------------------- 15. gate arithmetic
def _fake_reports(*, mae=0.02, zero_mae=0.04, cosine=0.9, fp=0.01, ratio=0.1,
                  auprc_multiple=10.0, current_mae=0.03, violations=0,
                  deterministic=True, crossings=0):
    runs = {}
    for seed in (0, 1, 2):
        for variant, value in (("FULL_CAUSAL", mae), ("CURRENT_FRAME_ONLY", current_mae)):
            runs[f"{variant}__seed{seed}"] = {
                "variant": variant, "seed": seed,
                "metrics": {"offline_reference_test": {
                    "nonfinite_outputs": 0,
                    "constraint_violations": {"total": violations},
                    "head": {"differential_mae": value,
                             "median_direction_cosine_active": cosine,
                             "hazard_absent_rms": ratio, "hazard_absent_raw_head_rms": 1.0},
                    "activation": {"oracle_zero_false_positive_rate": fp},
                    "mask": {"auprc": auprc_multiple * 0.001, "prevalence": 0.001},
                    "by_source_mode": {
                        "EXPERT_RECONSTRUCTED": {"available": True,
                                                 "median_direction_cosine_active": cosine},
                        "LEARNER_INDUCED_ON_POLICY": {"available": False}},
                }}}
    final = {
        "runs": runs,
        "baselines": {"ZERO_DIFFERENTIAL": {"metrics": {"offline_reference_test": {
            "head": {"differential_mae": zero_mae}}}}},
        "safety_head": {"unchanged": True, "frozen": True},
        "checkpoint_reload_determinism": {
            k: {"bitwise_identical": deterministic} for k in runs},
        "offline_test_opened_after_all_training_and_calibration": True,
    }
    contract = {"valid": True, "prohibited_inputs_used": [],
                "inputs_not_live_available": []}
    partition = {"total_crossings": crossings}
    dataset = {"dataset": {"tree_sha256": "abc"}}
    return final, contract, partition, dataset


def test_gates_pass_and_return_ready_when_everything_clears():
    gates = cps_gates.compute_gates(*_fake_reports())
    assert gates["all_passed"] is True
    assert cps_gates.decide(gates) == cps_gates.DECISION_READY


def test_failing_the_zero_baseline_gate_returns_signal_insufficient():
    gates = cps_gates.compute_gates(*_fake_reports(mae=0.039, zero_mae=0.04))
    assert gates["generalization_passed"] is False
    assert cps_gates.decide(gates) == cps_gates.DECISION_INSUFFICIENT


def test_beating_baseline_but_failing_direction_returns_overfit():
    gates = cps_gates.compute_gates(*_fake_reports(cosine=0.4))
    assert cps_gates.decide(gates) == cps_gates.DECISION_OVERFIT


def test_partition_leakage_returns_data_contract_invalid():
    gates = cps_gates.compute_gates(*_fake_reports(crossings=3))
    assert cps_gates.decide(gates) == cps_gates.DECISION_DATA_INVALID


def test_constraint_violation_is_a_technical_failure():
    gates = cps_gates.compute_gates(*_fake_reports(violations=5))
    assert cps_gates.decide(gates) == cps_gates.DECISION_TRAINING_FAILED


def test_nondeterministic_reload_is_a_technical_failure():
    gates = cps_gates.compute_gates(*_fake_reports(deterministic=False))
    assert cps_gates.decide(gates) == cps_gates.DECISION_TRAINING_FAILED


def test_missing_checkpoints_short_circuit_to_training_failed():
    gates = cps_gates.compute_gates(*_fake_reports())
    assert cps_gates.decide(gates, training_produced_checkpoints=False) == \
        cps_gates.DECISION_TRAINING_FAILED


def test_one_bad_seed_fails_the_gate_even_when_the_mean_would_pass():
    final, contract, partition, dataset = _fake_reports(mae=0.02, zero_mae=0.04)
    final["runs"]["FULL_CAUSAL__seed2"]["metrics"][
        "offline_reference_test"]["head"]["differential_mae"] = 0.039
    gates = cps_gates.compute_gates(final, contract, partition, dataset)
    gate = next(g for g in gates["generalization"]
                if g["gate"] == "offline_mae_beats_zero_by_25pct")
    assert gate["passed"] is False


def test_decision_token_is_always_one_of_the_allowed_five():
    for kwargs in ({}, {"mae": 0.039}, {"cosine": 0.1}, {"crossings": 1},
                   {"violations": 2}, {"fp": 0.5}):
        gates = cps_gates.compute_gates(*_fake_reports(**kwargs))
        assert cps_gates.decide(gates) in cps_gates.ALLOWED_DECISIONS


def test_final_decision_token_matches_markdown_last_line():
    decision = _json(DIAG / "final_decision.json")
    markdown = (ROOT / "docs"
                / "CAUSAL_PARKED_SKIN_REFERENCE_V1_FINAL_DECISION.md").read_text()
    last = [line for line in markdown.splitlines() if line.strip()][-1]
    assert last == decision["decision"]
    assert decision["decision"] in cps_gates.ALLOWED_DECISIONS


def test_simpler_model_branch_recomputes_gates_against_the_simpler_model():
    """Freezing CURRENT_FRAME_ONLY must not let it inherit FULL_CAUSAL's numbers."""
    final, contract, partition, dataset = _fake_reports(
        mae=0.02, current_mae=0.021, zero_mae=0.04, cosine=0.9)
    resolved = cps_gates.resolve(final, contract, partition, dataset)
    assert resolved["frozen_primary"] == cps_gates.BASELINE_CURRENT
    assert resolved["branch"] == "temporal_history_added_no_measurable_value"
    assert resolved["decision"] == cps_gates.DECISION_READY
    observed = next(g for g in resolved["gates"]["generalization"]
                    if g["gate"] == "offline_mae_beats_zero_by_25pct")["observed"]
    assert all("CURRENT_FRAME_ONLY" in k for k in observed)


def test_simpler_branch_still_fails_when_the_simpler_model_cannot_clear_the_bar():
    final, contract, partition, dataset = _fake_reports(
        mae=0.02, current_mae=0.0205, zero_mae=0.021, cosine=0.9)
    resolved = cps_gates.resolve(final, contract, partition, dataset)
    assert resolved["decision"] != cps_gates.DECISION_READY


def test_full_causal_branch_taken_when_history_genuinely_helps():
    final, contract, partition, dataset = _fake_reports(mae=0.02, current_mae=0.04)
    resolved = cps_gates.resolve(final, contract, partition, dataset)
    assert resolved["frozen_primary"] == cps_gates.BASELINE_FULL
    assert resolved["decision"] == cps_gates.DECISION_READY
