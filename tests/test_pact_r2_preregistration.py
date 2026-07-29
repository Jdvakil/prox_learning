from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "configs" / "pact_r2_preregistration_v1.json"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_r2_preregistration_is_self_hashed_and_authorized():
    document = json.loads(PREREG.read_text())
    payload = dict(document)
    observed = payload.pop("preregistration_sha256")
    assert _canonical_hash(payload) == observed
    authorization = document["authorization"]
    assert _hash_file(Path(authorization["path"])) == authorization["sha256"]


def test_r2_carries_forward_the_frozen_scientific_inputs():
    document = json.loads(PREREG.read_text())
    analysis_path = ROOT / "scripts" / "analyze_pact_confirmatory.py"
    assert _hash_file(analysis_path) == document["analysis"][
        "frozen_analysis_script_sha256"
    ]
    training = json.loads(
        (
            ROOT
            / "diagnostics_output/pact_vs_act/policy_training_summary_v2.json"
        ).read_text()
    )
    observed = {
        arm: {
            str(record["seed"]): record["checkpoint_sha256"]
            for record in training["records"]
            if record["arm"] == arm
        }
        for arm in ("ACT", "PACT")
    }
    assert observed["ACT"] == document["carried_forward"][
        "act_checkpoint_sha256s"
    ]
    assert observed["PACT"] == document["carried_forward"][
        "pact_checkpoint_sha256s"
    ]
    assert training["surface_encoder_sha256"] == document["carried_forward"][
        "surface_encoder_sha256"
    ]


def test_r2_recovery_is_all_inflight_or_none():
    boundary = json.loads(PREREG.read_text())["boundary_amendment"]
    assert boundary["row_terminal_boundary"] == "valid scientific result.json"
    assert boundary["all_inflight_rows_rerun"] is True
    assert boundary["recovery_event_frozen_before_rerun"] is True
    assert boundary["individual_post_observation_retry"] is False
    assert boundary["cohort_exit_window_seconds"] == 5
