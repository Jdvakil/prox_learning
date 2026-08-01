#!/usr/bin/env python3
"""Guard and commit only reconciled seed-replication result reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/act_retrain_venv/bin/python")
TOKEN_SET = {
    "SEED_REPLICATION_CONFIRMED",
    "SEED_REPLICATION_PARTIAL",
    "SEED_REPLICATION_FAILED",
    "SEED_REPLICATION_INCOMPLETE",
}
RESULT_PATHS = (
    "docs/PACT_SEED_REPLICATION_DECISION.md",
    "diagnostics_output/pact_seed_replication/analysis.json",
    "diagnostics_output/pact_seed_replication/final_decision.json",
    "diagnostics_output/pact_seed_replication/provenance.json",
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def validate_results() -> str:
    provenance_path = ROOT / RESULT_PATHS[-1]
    provenance = json.loads(provenance_path.read_text())
    payload = dict(provenance)
    observed = payload.pop("provenance_sha256")
    if observed != canonical_hash(payload):
        raise RuntimeError("provenance self-hash mismatch")
    decision = json.loads((ROOT / RESULT_PATHS[2]).read_text())
    token = decision["decision"]
    if token not in TOKEN_SET or provenance["decision"] != token:
        raise RuntimeError("seed-replication decision token mismatch")
    report_lines = [
        line.strip() for line in (ROOT / RESULT_PATHS[0]).read_text().splitlines() if line.strip()
    ]
    if not report_lines or report_lines[-1] != token:
        raise RuntimeError("decision report does not end in exact token")
    if (
        provenance["completion_records"] != 120
        or provenance["scientific_schedule_reconciled"] is not True
        or provenance["losslessly_compacted_rows"] != 118
    ):
        raise RuntimeError("result provenance is not reconciled")
    return token


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise SystemExit("poll interval must be positive")
    provenance = ROOT / RESULT_PATHS[-1]
    while not provenance.exists():
        time.sleep(args.poll_seconds)
    if git("rev-parse", "HEAD") != args.expected_head:
        raise SystemExit("HEAD changed while evaluation ran; refusing automatic commit")
    status_before = git("status", "--porcelain").splitlines()
    observed_paths = {line[3:] for line in status_before if len(line) >= 4}
    if observed_paths != set(RESULT_PATHS):
        raise SystemExit(
            "worktree differs beyond exact result files; refusing automatic commit: "
            f"{sorted(observed_paths)}"
        )
    token = validate_results()
    subprocess.run(
        [str(PYTHON), "-m", "pytest", "-q", "tests"],
        cwd=ROOT,
        check=True,
    )
    if git("rev-parse", "HEAD") != args.expected_head:
        raise SystemExit("HEAD changed during tests; refusing automatic commit")
    subprocess.run(["git", "add", "--", *RESULT_PATHS], cwd=ROOT, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"Report PACT seed replication: {token}"],
        cwd=ROOT,
        check=True,
    )
    print(git("rev-parse", "HEAD"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
