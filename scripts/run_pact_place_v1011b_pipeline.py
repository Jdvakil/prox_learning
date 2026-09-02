#!/usr/bin/env python3
"""Run V10.11b's frozen validation sequence through owner review."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pact_place_v1011b_contract as contract  # noqa: E402

STATUS_ROOT = ROOT / "diagnostics_output/pact_place_v1011b_pipeline"
STATUS_PATH = STATUS_ROOT / "status.json"


def _write_status(stage: str, *, complete: bool = False, error: str | None = None) -> None:
    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(
        json.dumps(
            {
                "schema_version": "pact_place_v1011b_pipeline_status_v1",
                "stage": stage,
                "complete": complete,
                "error": error,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _run(stage: str, argv: list[str]) -> None:
    _write_status(stage)
    print(f"\n=== {stage} ===", flush=True)
    subprocess.run(argv, cwd=ROOT, check=True)


def main() -> int:
    if (ROOT / contract.REVIEW_ROOT / "review_manifest.json").exists():
        raise SystemExit("V10.11b review already exists; refusing a second pipeline run")
    try:
        _run(
            "targeted_tests",
            [
                sys.executable,
                "-m",
                "unittest",
                "tests.test_pact_place_v1011_mixed_clutter",
                "tests.test_pact_place_v106",
                "-q",
            ],
        )
        _write_status("contract")
        contract_path = ROOT / contract.CONTRACT_ROOT / "contract.json"
        contract.write_contract(contract_path)
        print(f"wrote {contract_path.relative_to(ROOT)}", flush=True)
        _run(
            "paired_raw_visibility",
            [sys.executable, "scripts/run_pact_place_v1011b_visibility.py"],
        )
        _run(
            "preflight_96",
            [sys.executable, "scripts/run_pact_place_v1011b_preflight.py", "--workers", "8"],
        )
        _run(
            "six_episode_review",
            [
                sys.executable,
                "scripts/run_pact_place_v1011b_review.py",
                "--workers",
                "12",
                "--render-workers",
                "3",
                "--batch-size",
                "12",
            ],
        )
        _write_status("complete", complete=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - durable background close-out
        _write_status("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
