#!/usr/bin/env python3
"""Run the proven shell-kill detachment proof with the contact launcher."""

from __future__ import annotations

import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import prove_pact_frontend_screen_detachment as implementation


_base_validate_launch_smoke = implementation.validate_launch_smoke
CONTACT_SMOKE_VALIDATION = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class GpuMemorySampler:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.samples: list[tuple[str, int]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        completed = subprocess.run(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        total_mib = sum(int(line.strip()) for line in completed.stdout.splitlines() if line.strip())
        self.samples.append((utc_now(), total_mib))

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._sample()
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop_event.wait(1.0)

    def start(self) -> None:
        self.thread.start()

    def finish(self) -> dict:
        self.stop_event.set()
        self.thread.join(timeout=5.0)
        if not self.samples:
            raise RuntimeError("no GPU-memory sample was captured during contact smoke")
        peak_timestamp, peak_mib = max(self.samples, key=lambda item: item[1])
        return {
            "query": "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
            "sample_interval_seconds": 1.0,
            "sample_count": len(self.samples),
            "first_sample_utc": self.samples[0][0],
            "last_sample_utc": self.samples[-1][0],
            "peak_sample_utc": peak_timestamp,
            "peak_memory_used_mib": peak_mib,
            "endpoint_fields_read": False,
        }


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


implementation.LAUNCHER = implementation.ROOT / "scripts/launch_pact_contact_detached.py"
implementation.validate_launch_smoke = validate_contact_launch_smoke


if __name__ == "__main__":
    sampler = GpuMemorySampler()
    sampler.start()
    try:
        code = implementation.main()
    finally:
        gpu_memory = sampler.finish()
    if code == 0:
        if CONTACT_SMOKE_VALIDATION is None:
            raise RuntimeError("contact smoke structural validation was not recorded")
        arguments = sys.argv
        output_root = Path(arguments[arguments.index("--output-root") + 1]).resolve()
        proof_path = output_root / "detachment_proof.json"
        proof = implementation.json.loads(proof_path.read_text())
        proof.pop("detachment_proof_sha256", None)
        proof["contact_smoke_validation"] = CONTACT_SMOKE_VALIDATION
        proof["smoke_gpu_memory"] = gpu_memory
        proof["endpoint_outcome_values_inspected"] = False
        proof["detachment_proof_sha256"] = implementation.canonical_hash(proof)
        implementation.write_json_atomic(proof_path, proof)
        print(proof["detachment_proof_sha256"])
    raise SystemExit(code)
