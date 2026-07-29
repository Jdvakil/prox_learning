from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pact_r2_contract as contract  # noqa: E402

MANIFEST = ROOT / "configs/pact_confirmatory_r2_manifest_v1.json"
SCHEDULE = ROOT / "diagnostics_output/pact_vs_act_r2/schedule.json"
DISPATCH = ROOT / "diagnostics_output/pact_vs_act_r2/dispatch_contract.json"


def _canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_r2_manifest_is_fresh_and_balanced():
    manifest = contract.load_manifest(MANIFEST)
    assert manifest["total_candidates"] == 160
    assert manifest["r1_quarantine"]["r1_episode_id_count"] == 624
    assert manifest["r1_quarantine"]["overlap_episode_ids"] == []
    assert manifest["r1_quarantine"]["r1_endpoint_loaded"] is False
    assert Counter(row["intrusion_side"] for row in manifest["rows"]) == {
        "left": 80,
        "right": 80,
    }


def test_frozen_r2_schedule_is_complete_unique_and_balanced():
    schedule = json.loads(SCHEDULE.read_text())
    payload = dict(schedule)
    observed = payload.pop("schedule_sha256")
    assert observed == _canonical_hash(payload)
    assert schedule["instances"] == 160
    assert schedule["rollouts"] == 960
    assert schedule["workers"] == 8
    assert len({row["rollout_id"] for row in schedule["rows"]}) == 960
    assert len({row["output_relpath"] for row in schedule["rows"]}) == 960
    assert Counter(
        (row["arm"], row["checkpoint_seed"]) for row in schedule["rows"]
    ) == {
        (arm, seed): 160
        for arm in ("ACT", "PACT", "PACT_ZERO")
        for seed in (3101, 3102)
    }


def test_frozen_dispatch_binds_runtime_and_authorized_recovery():
    schedule = json.loads(SCHEDULE.read_text())
    dispatch = json.loads(DISPATCH.read_text())
    payload = dict(dispatch)
    observed = payload.pop("dispatch_contract_sha256")
    assert observed == _canonical_hash(payload)
    assert dispatch["scientific_schedule"]["schedule_sha256"] == schedule[
        "schedule_sha256"
    ]
    assert dispatch["scientific_schedule"]["file_sha256"] == _file_hash(
        SCHEDULE
    )
    assert dispatch["boundary_amendment"]["all_inflight_rows_rerun"] is True
    assert (
        dispatch["boundary_amendment"]["individual_post_observation_retry"]
        is False
    )
    assert (
        dispatch["detachment_proof"]["required_before_full_dispatch"] is True
    )
    assert dispatch["throughput"]["required_measurement_elapsed_minutes"] == 20
    for record in dispatch["frozen_inputs"]["runtime"].values():
        assert _file_hash(Path(record["path"])) == record["sha256"]
