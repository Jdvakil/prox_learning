#!/usr/bin/env python3
"""Run the proven shell-kill detachment proof with the contact launcher."""

from __future__ import annotations

import sys
from pathlib import Path

import prove_pact_frontend_screen_detachment as implementation


_base_validate_launch_smoke = implementation.validate_launch_smoke
CONTACT_SMOKE_VALIDATION = None


def validate_contact_launch_smoke(*, schedule, contract, output_root):
    global CONTACT_SMOKE_VALIDATION
    artifact = _base_validate_launch_smoke(
        schedule=schedule, contract=contract, output_root=output_root
    )
    smoke = contract["launch_smoke"]
    row_dir = output_root / smoke["output_relpath"]
    result = implementation.json.loads((row_dir / "result.json").read_text())
    info = result.get("policy_info", {})
    expected = {
        "arm": "PACT_PERMUTED",
        "token_plan_max_control_steps": 900,
        "token_plan_sha256": smoke["token_plan_sha256"],
    }
    for key, value in expected.items():
        if info.get(key) != value:
            raise RuntimeError(f"contact smoke {key} mismatch")
    consumed = int(info.get("token_plan_frames_consumed", -1))
    control_steps = int(info.get("control_steps", -2))
    if consumed <= 0 or consumed != control_steps or consumed > 900:
        raise RuntimeError("contact smoke did not consume its frozen token horizon correctly")
    audit = result.get("contact_audit", {})
    if (
        audit.get("contact_frame_payload_retained") is not False
        or int(audit.get("sample_count", 0)) < control_steps
        or set(audit.get("frames_with_contact", {}))
        != {"grasp_target", "hazard_bar", "other_environment"}
        or set(audit.get("maximum_penetration_depth_m", {}))
        != {"grasp_target", "hazard_bar", "other_environment"}
    ):
        raise RuntimeError("contact smoke endpoint instrumentation is incomplete")
    CONTACT_SMOKE_VALIDATION = {
        "passed": True,
        "arm": info["arm"],
        "token_plan_sha256": info["token_plan_sha256"],
        "token_plan_max_control_steps": info["token_plan_max_control_steps"],
        "token_plan_frames_consumed": consumed,
        "control_steps": control_steps,
        "summary_only_contact_audit": True,
        "contact_class_keys": sorted(audit["frames_with_contact"]),
        "maximum_penetration_keys": sorted(audit["maximum_penetration_depth_m"]),
        "endpoint_outcome_values_inspected": False,
    }
    return artifact


implementation.LAUNCHER = (
    implementation.ROOT / "scripts/launch_pact_contact_detached.py"
)
implementation.validate_launch_smoke = validate_contact_launch_smoke


if __name__ == "__main__":
    code = implementation.main()
    if code == 0:
        if CONTACT_SMOKE_VALIDATION is None:
            raise RuntimeError("contact smoke structural validation was not recorded")
        arguments = sys.argv
        output_root = Path(
            arguments[arguments.index("--output-root") + 1]
        ).resolve()
        proof_path = output_root / "detachment_proof.json"
        proof = implementation.json.loads(proof_path.read_text())
        proof.pop("detachment_proof_sha256", None)
        proof["contact_smoke_validation"] = CONTACT_SMOKE_VALIDATION
        proof["endpoint_outcome_values_inspected"] = False
        proof["detachment_proof_sha256"] = implementation.canonical_hash(proof)
        implementation.write_json_atomic(proof_path, proof)
        print(proof["detachment_proof_sha256"])
    raise SystemExit(code)
