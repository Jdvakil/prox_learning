#!/usr/bin/env python3
"""Run the proven shell-kill detachment proof with the contact launcher."""

from __future__ import annotations

import prove_pact_frontend_screen_detachment as implementation


_base_validate_launch_smoke = implementation.validate_launch_smoke


def validate_contact_launch_smoke(*, schedule, contract, output_root):
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
    return artifact


implementation.LAUNCHER = (
    implementation.ROOT / "scripts/launch_pact_contact_detached.py"
)
implementation.validate_launch_smoke = validate_contact_launch_smoke


if __name__ == "__main__":
    raise SystemExit(implementation.main())
