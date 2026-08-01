#!/usr/bin/env python3
"""Train only the frozen-recipe PACT seed-3102 replication policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import train_pact_frontend_screen_policy as implementation

POLICY_SEED = 3102
ACT_CHECKPOINT = Path(
    "/root/pact_remediation_artifacts_v2/full/policies_v2/act_seed3102/policy_best.ckpt"
)
ACT_CHECKPOINT_SHA256 = "e98d98bad87e2762cef37eb953d9ab55fcb65ed6355d2d8e9a881f38ef48c8d4"


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--summary-out", required=True, type=Path)
    known, _remaining = parser.parse_known_args()
    implementation.POLICY_SEED = POLICY_SEED
    implementation.ACT_CHECKPOINT = ACT_CHECKPOINT
    implementation.ACT_CHECKPOINT_SHA256 = ACT_CHECKPOINT_SHA256
    status = implementation.main()
    if status != 0:
        return status
    report = json.loads(known.summary_out.read_text())
    if (
        report.get("seed") != POLICY_SEED
        or report.get("reused_act", {}).get("checkpoint_sha256") != ACT_CHECKPOINT_SHA256
    ):
        raise SystemExit("seed-replication training identity changed")
    report["schema_version"] = "pact_seed_replication_policy_training_v1"
    report.pop("pact_zero", None)
    report["permuted_ablation"] = {
        "separately_trained": False,
        "checkpoint_alias": report["checkpoint"],
        "scheme": "frozen_distribution_matched_seed_2026073105",
    }
    report["only_recipe_difference_from_seed_3101"] = "seed=3102"
    known.summary_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
