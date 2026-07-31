from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compact_pact_frontend_screen_storage as screen_compactor
import compact_pact_r2_storage as implementation


def test_screen_storage_exclusions_are_smoke_and_final():
    assert screen_compactor.EXCLUDED_SCHEDULE_INDICES == {0, 119}


def test_frozen_screen_amendment_binds_schedule_and_compactor():
    original = implementation.EXCLUDED_SCHEDULE_INDICES
    try:
        implementation.EXCLUDED_SCHEDULE_INDICES = set(
            screen_compactor.EXCLUDED_SCHEDULE_INDICES
        )
        config, schedule = implementation.validate_inputs(
            config_path=(
                ROOT
                / "configs/pact_frontend_screen_storage_amendment_v1.json"
            ),
            schedule_path=(
                ROOT
                / "diagnostics_output/pact_frontend_screen/schedule.json"
            ),
            output_root=Path(
                "/root/pact_frontend_screen_artifacts/evaluation_621764f8"
            ),
        )
    finally:
        implementation.EXCLUDED_SCHEDULE_INDICES = original
    wrapper = ROOT / "scripts/compact_pact_frontend_screen_storage.py"
    assert (
        implementation.sha256_file(wrapper)
        == config["screen_compactor_wrapper_sha256"]
    )
    assert schedule["schedule_sha256"] == config["schedule_sha256"]
    assert config["excluded_intact_schedule_indices"] == [0, 119]
    assert (
        config["provenance"][
            "endpoint_outcomes_inspected_before_amendment"
        ]
        is False
    )
    assert all(
        value is False
        for value in config["frozen_scientific_contract"].values()
    )


def test_dispatch_predeclared_same_raw_rows():
    dispatch = json.loads(
        (
            ROOT
            / "diagnostics_output/pact_frontend_screen/dispatch.json"
        ).read_text()
    )
    assert dispatch["storage"] == {
        "raw_smoke_schedule_index_preserved": 0,
        "raw_final_schedule_index_preserved": 119,
        "weights_and_rollout_payloads_not_committed": True,
    }
