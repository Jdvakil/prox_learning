#!/usr/bin/env python3
"""Guard and commit only fully reconciled contact-endpoint result reports."""

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
TOKENS = {
    "CONTACT_REDUCTION_ESTABLISHED",
    "CONTACT_REDUCTION_WITH_TASK_BENEFIT",
    "CONTACT_REDUCTION_SUBSET_ONLY",
    "NO_CONTACT_REDUCTION",
    "CONTACT_INCREASE",
    "CONTACT_EXPERIMENT_INCOMPLETE",
}
RESULT_PATHS = (
    "docs/PACT_CONTACT_ENDPOINT_DECISION.md",
    "diagnostics_output/pact_contact_endpoint/analysis.json",
    "diagnostics_output/pact_contact_endpoint/final_decision.json",
    "diagnostics_output/pact_contact_endpoint/provenance.json",
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def validate_results() -> str:
    provenance = json.loads((ROOT / RESULT_PATHS[-1]).read_text())
    payload = dict(provenance)
    observed = payload.pop("provenance_sha256", None)
    if observed != canonical_hash(payload):
        raise RuntimeError("contact provenance self-hash mismatch")
    decision = json.loads((ROOT / RESULT_PATHS[2]).read_text())
    token = decision["decision"]
    if token not in TOKENS or provenance["decision"] != token:
        raise RuntimeError("contact decision token mismatch")
    report_lines = [
        line.strip() for line in (ROOT / RESULT_PATHS[0]).read_text().splitlines() if line.strip()
    ]
    if not report_lines or report_lines[-1] != token:
        raise RuntimeError("contact report does not end in exact token")
    if (
        provenance["completion_records"] != 1200
        or provenance["scientific_schedule_reconciled"] is not True
        or provenance["storage"]["endpoint_complete_compacted_rows"] != 1198
    ):
        raise RuntimeError("contact result provenance is not reconciled")
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
        raise SystemExit("HEAD changed while contact evaluation ran; refusing automatic commit")
    status = git("status", "--porcelain").splitlines()
    observed_paths = {line[3:] for line in status if len(line) >= 4}
    if observed_paths != set(RESULT_PATHS):
        raise SystemExit(
            "worktree differs beyond exact contact result files; refusing automatic commit: "
            f"{sorted(observed_paths)}"
        )
    token = validate_results()
    subprocess.run([str(PYTHON), "-m", "pytest", "-q", "tests"], cwd=ROOT, check=True)
    if git("rev-parse", "HEAD") != args.expected_head:
        raise SystemExit("HEAD changed during tests; refusing automatic commit")
    subprocess.run(["git", "add", "--", *RESULT_PATHS], cwd=ROOT, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"Report PACT contact endpoint: {token}"],
        cwd=ROOT,
        check=True,
    )
    print(git("rev-parse", "HEAD"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
