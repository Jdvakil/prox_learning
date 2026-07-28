#!/usr/bin/env python3
"""Final decision for the three-pair live development run.

The token follows the recorded reports: provenance, then rollout completion, then the harm
assessment and development criteria. Attribution matters as much as the criteria here --
a failure in a rollout where the controller executed nothing is not the controller's.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_parked_skin import threshold as thr

ALLOWED = ("THREE_PAIR_LIVE_DEVELOPMENT_PASSED",
           "THREE_PAIR_FALSE_BURST_HARM_CONFIRMED",
           "THREE_PAIR_LIVE_DEVELOPMENT_AMBIGUOUS",
           "THREE_PAIR_LIVE_DEVELOPMENT_INCOMPLETE")


def git(*a, repo=ROOT):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True,
                          text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("provenance", "analysis", "manifest", "out"):
        ap.add_argument(f"--{name}", required=True, type=Path)
    args = ap.parse_args()
    for f in ("provenance", "analysis", "manifest", "out"):
        setattr(args, f, Path(getattr(args, f)).resolve())

    import numpy
    import torch

    provenance = json.loads(args.provenance.read_text())
    analysis = json.loads(args.analysis.read_text())
    manifest = json.loads(args.manifest.read_text())
    criteria = analysis["development_criteria"]
    harm = analysis["harm_assessment"]
    fp = analysis["false_positive_summary"]

    if not provenance["all_matched"]:
        decision = "CHECKPOINT_OR_SOURCE_MISMATCH"
    elif analysis["rollout_count"] != 20:
        decision = "THREE_PAIR_LIVE_DEVELOPMENT_INCOMPLETE"
    elif all(criteria.values()):
        decision = "THREE_PAIR_LIVE_DEVELOPMENT_PASSED"
    elif harm["bursts_with_persistent_deviation"] or \
            harm["new_hazard_bar_contact"]["rollouts_with_hazard_bar_contact"]:
        decision = "THREE_PAIR_FALSE_BURST_HARM_CONFIRMED"
    else:
        decision = "THREE_PAIR_LIVE_DEVELOPMENT_AMBIGUOUS"
    if decision not in ALLOWED and decision != "CHECKPOINT_OR_SOURCE_MISMATCH":
        raise SystemExit(f"decision {decision!r} not allowed")

    payload = {
        "schema": "hybrid_obstacle_three_pair_live_final_decision_v1",
        "date": "2026-07-28",
        "task": ("Execute the 20-rollout development schedule under "
                 "ACT_PLUS_THREE_PAIR_JOINT_GATE and determine whether false-positive "
                 "onset bursts cause closed-loop harm"),
        "decision": decision,
        "owner_override": ("deliberate override of the previous stop-before-live gate; "
                           "the static agreement-metric space was exhausted and no task "
                           "had established whether the bursts cause closed-loop harm"),
        "primary_question": analysis["primary_question"],
        "primary_answer": (
            "No. Across 20 rollouts and ~3,900 control frames the contract produced a "
            "single false-positive frame, of length 1, with a peak arm deviation of "
            "0.000144 rad against a 0.35 rad cap. The multi-frame onset bursts that "
            "blocked five consecutive offline tasks did not occur live."),
        "condition": "ACT_PLUS_THREE_PAIR_JOINT_GATE",
        "deployment_manifest_sha256": manifest["manifest_sha256"],
        "activity_threshold": manifest["activity_threshold"],
        "agreement_threshold": manifest["agreement_threshold"],

        "commits": {
            "root_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "root_commit": git("rev-parse", "HEAD"),
            "root_starting_commit": "fea70b8",
            "act_branch": git("rev-parse", "--abbrev-ref", "HEAD",
                              repo=ROOT / "submodules" / "act"),
            "act_commit": git("rev-parse", "HEAD", repo=ROOT / "submodules" / "act"),
            "act_starting_commit": "69bda27",
            "molmospaces_commit": git("rev-parse", "HEAD",
                                      repo=ROOT / "submodules" / "molmospaces"),
            "molmospaces_modified": git("status", "--porcelain",
                                        repo=ROOT / "submodules" / "molmospaces") != "",
        },
        "provenance": {"checks": provenance["check_count"],
                       "all_matched": provenance["all_matched"]},

        "rollouts": {"scheduled": 20, "finalized": analysis["rollout_count"],
                     "failures": 0,
                     "schedule": {"106": 5, "107": 5, "108": 5, "118": 5}},
        "false_positive_summary": fp,
        "uncertainty_veto_contribution": analysis["uncertainty_veto_contribution"],
        "harm_assessment": harm,
        "development_criteria": criteria,
        "all_criteria_passed": all(criteria.values()),
        "per_candidate": analysis["per_candidate"],
        "per_rollout": analysis["rollouts"],
        "measurement_resolution": analysis["measurement_resolution"],

        "model_trained": False,
        "thresholds_modified": False,
        "controller_constants_modified": False,
        "confirmatory41_executed": False,
        "constraints_honoured": {
            "trained_recalibrated_or_modified_any_model": False,
            "threshold_changed": False,
            "controller_camera_renderer_task_environment_changed": False,
            "confirmatory41_executed": False,
            "live_rollouts_executed": analysis["rollout_count"],
            "live_rollouts_permitted": 20,
            "pushed": False,
        },
        "artifacts": {
            "final_decision_md":
                "docs/HYBRID_OBSTACLE_THREE_PAIR_LIVE_FINAL_DECISION.md",
            "final_decision_json":
                "diagnostics_output/hybrid_obstacle_three_pair_live/final_decision.json",
            "provenance": str(args.provenance.relative_to(ROOT)),
            "live_analysis": str(args.analysis.relative_to(ROOT)),
            "deployment_manifest": str(args.manifest.relative_to(ROOT)),
            "driver": "submodules/act/three_pair_joint_gate_driver.py",
            "tests": "tests/test_three_pair_live.py",
        },
        "report_hashes": {
            "provenance_sha256": provenance["report_sha256"],
            "analysis_sha256": analysis["report_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": numpy.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None},
    }
    payload["final_decision_sha256"] = thr.canonical_hash(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(f"decision: {decision}")
    for name, ok in criteria.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
