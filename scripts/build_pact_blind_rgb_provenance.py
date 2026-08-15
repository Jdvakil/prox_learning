#!/usr/bin/env python3
"""Record immutable provenance for the frozen blind-RGB experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from pact_blind_rgb_contract import sha256_file, sha256_payload


ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "submodules/act"


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--dispatch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    schedule = json.loads(args.schedule.read_text())
    dispatch = json.loads(args.dispatch.read_text())
    document = {
        "schema_version": "pact_blind_rgb_provenance_v1",
        "repository": {
            "path": str(ROOT),
            "branch": git(ROOT, "branch", "--show-current"),
            "head_at_freeze": git(ROOT, "rev-parse", "HEAD"),
            "act_submodule_head_at_freeze": git(ACT, "rev-parse", "HEAD"),
            "no_push_performed_by_this_experiment": True,
        },
        "frozen_documents": {
            "manifest": {"path": str(args.manifest.resolve()), "file_sha256": sha256_file(args.manifest), "self_sha256": manifest["manifest_sha256"]},
            "schedule": {"path": str(args.schedule.resolve()), "file_sha256": sha256_file(args.schedule), "self_sha256": schedule["schedule_sha256"]},
            "dispatch": {"path": str(args.dispatch.resolve()), "file_sha256": sha256_file(args.dispatch), "self_sha256": dispatch["dispatch_contract_sha256"]},
        },
        "intervention": manifest["intervention"],
        "predeclared_expected_outcome": schedule["predeclared_expected_outcome"],
        "scientific_design": schedule["analysis_contract"],
        "decision_rule": schedule["decision_rule"],
        "rollouts": schedule["rollouts"],
        "workers": schedule["workers"],
        "worker_sizing": dispatch["worker_sizing"],
        "no_retraining": True,
        "no_new_encoder": True,
        "no_new_demonstrations": True,
        "rgb_only_intervention": True,
        "checkpoint_records": manifest["frozen_artifacts"]["checkpoints"],
        "surface_encoder": manifest["frozen_artifacts"]["surface_encoder"],
        "token_plan": manifest["frozen_artifacts"]["token_plan"],
        "protected_scientific_artifacts": manifest["protected_scientific_artifacts"],
    }
    document["provenance_sha256"] = sha256_payload(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["provenance_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
