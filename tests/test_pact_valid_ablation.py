from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "submodules/act"))

import analyze_pact_valid_ablation as analyzer
import build_pact_permuted_token_plan as token_builder
import eval_pact_valid_ablation_row as evaluator


def _hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_source_selection_is_unique_and_changes_episode_each_step():
    episodes = np.repeat(np.arange(100, dtype=np.int16), 300)
    timesteps = np.tile(np.arange(300, dtype=np.int16), 100)
    selected_episodes, selected_timesteps = token_builder.select_sources(
        episodes, timesteps
    )
    assert selected_episodes.shape == (40, 512)
    assert selected_timesteps.shape == (40, 512)
    assert all(
        np.all(row[1:] != row[:-1]) for row in selected_episodes
    )
    selected_pairs = set(
        zip(selected_episodes.ravel(), selected_timesteps.ravel())
    )
    assert len(selected_pairs) == 40 * 512


def test_evaluator_consumes_frozen_permuted_frame(monkeypatch):
    frames = np.arange(3 * 40 * 32, dtype=np.float32).reshape(
        3, 40, 32
    )
    monkeypatch.setattr(evaluator, "TOKEN_FRAMES", frames)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **_kwargs: self)
    policy = evaluator.PactPermutedInferencePolicy.__new__(
        evaluator.PactPermutedInferencePolicy
    )
    policy._step = 1
    observed = policy._surface_positions(np.zeros((40, 8, 8))).cpu().numpy()
    assert observed.shape == (1, 40, 32)
    assert np.array_equal(observed[0], frames[1])


def _result(episode: str, arm: str, success: bool) -> dict:
    return {
        "episode_id": episode,
        "arm": arm,
        "checkpoint_sha256": "checkpoint",
        "rollout_id": f"rollout-{episode}",
        "schedule_row_sha256": f"row-{episode}",
        "task_success": success,
        "collision_free_task_success": success,
        "failure_taxonomy": (
            "collision_free_task_success" if success else "no_gripper_close"
        ),
        "contact_audit": {
            "contact_class_totals": {
                "grasp_target": 0,
                "hazard_bar": 0,
                "other_environment": 0,
            }
        },
    }


def test_paired_rule_detects_distribution_matched_signal(tmp_path):
    output = tmp_path / "permuted"
    references = []
    rows = []
    for index in range(40):
        episode = f"episode-{index}"
        reference_dir = tmp_path / "reference" / str(index)
        reference_dir.mkdir(parents=True)
        pact = _result(episode, "PACT", index < 30)
        (reference_dir / "result.json").write_text(json.dumps(pact))
        (reference_dir / "driver_result.json").write_text(
            json.dumps({"status": "complete"})
        )
        references.append(
            {
                "instance_episode_id": episode,
                "checkpoint_sha256": "checkpoint",
                "result_path": str(reference_dir / "result.json"),
                "result_sha256": analyzer.file_hash(
                    reference_dir / "result.json"
                ),
                "driver_path": str(reference_dir / "driver_result.json"),
                "driver_sha256": analyzer.file_hash(
                    reference_dir / "driver_result.json"
                ),
            }
        )
        row_dir = output / f"rows/{index:03d}"
        row_dir.mkdir(parents=True)
        permuted = _result(episode, "PACT_PERMUTED", index < 10)
        (row_dir / "result.json").write_text(json.dumps(permuted))
        (row_dir / "driver_result.json").write_text(
            json.dumps({"status": "complete"})
        )
        rows.append(
            {
                "instance_episode_id": episode,
                "checkpoint_sha256": "checkpoint",
                "rollout_id": f"rollout-{episode}",
                "schedule_row_sha256": f"row-{episode}",
                "output_relpath": f"rows/{index:03d}",
            }
        )
    schedule = {
        "schedule_sha256": "schedule",
        "bootstrap_replicates": 20000,
        "bootstrap_seed": 7,
        "decision_rule": {
            "signal_present": "difference >= 0.10 and CI lower > 0",
            "weak_signal": "difference >= 0.05 otherwise",
            "no_signal": "difference < 0.05",
        },
        "paired_pact_reference": references,
        "rows": rows,
    }
    analysis, final = analyzer.analyze(schedule, output)
    assert analysis["reconciliation"]["reconciled"] is True
    contrast = analysis["paired_instance_bootstrap"][
        "PACT_minus_PACT_PERMUTED"
    ]
    assert contrast["difference"] == 0.5
    assert contrast["ci_95"][0] > 0
    assert final["decision"] == "VALID_ABLATION_SIGNAL_PRESENT"


def test_all_valid_ablation_tokens_are_nonconfirmatory():
    assert analyzer.TOKENS == {
        "VALID_ABLATION_SIGNAL_PRESENT",
        "VALID_ABLATION_WEAK_SIGNAL",
        "VALID_ABLATION_NO_SIGNAL",
        "VALID_ABLATION_INCONCLUSIVE",
    }
    assert not analyzer.TOKENS & {
        "PACT_BENEFIT_ESTABLISHED",
        "PACT_NO_CONFIRMED_BENEFIT",
        "PACT_WORSE_THAN_ACT",
    }
