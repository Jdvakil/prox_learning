#!/usr/bin/env python3
"""Freeze the outcome-blind worker-count arithmetic for RGB blur sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pact_blur_sweep_contract import sha256_payload


OBSERVED_V2_WORKERS = 8
OBSERVED_V2_PEAK_MIB = 12320.0
OBSERVED_V2_GPU_UTILIZATION_PERCENT = 43.0
OBSERVED_V2_CPU_LOAD = 8.5
CPU_LOGICAL_CORES = 128
CEILING_MIB = 19 * 1024
WORKER_CAP = 12


def build() -> dict:
    per_worker_peak_mib = OBSERVED_V2_PEAK_MIB / OBSERVED_V2_WORKERS
    arithmetic_limit = math.floor(CEILING_MIB / per_worker_peak_mib)
    selected_workers = min(WORKER_CAP, arithmetic_limit)
    document = {
        "schema_version": "pact_blur_sweep_worker_sizing_v1",
        "outcome_fields_read": False,
        "measurement": {
            "source": "outcome-blind v2 eight-worker runtime telemetry",
            "workers": OBSERVED_V2_WORKERS,
            "peak_gpu_memory_mib": OBSERVED_V2_PEAK_MIB,
            "gpu_utilization_percent": OBSERVED_V2_GPU_UTILIZATION_PERCENT,
            "cpu_load": OBSERVED_V2_CPU_LOAD,
            "logical_cpu_cores": CPU_LOGICAL_CORES,
        },
        "formula": "floor(ceiling_mib / (observed_peak_mib / observed_workers)), capped at 12",
        "per_worker_peak_mib": per_worker_peak_mib,
        "ceiling_mib": CEILING_MIB,
        "arithmetic_worker_limit": arithmetic_limit,
        "worker_cap": WORKER_CAP,
        "selected_workers": selected_workers,
        "projected_peak_mib": selected_workers * per_worker_peak_mib,
        "projected_peak_gib": selected_workers * per_worker_peak_mib / 1024.0,
        "headroom_mib": CEILING_MIB - selected_workers * per_worker_peak_mib,
        "fallback_to_10_required_by_arithmetic": selected_workers < 12,
    }
    document["worker_sizing_sha256"] = sha256_payload(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    document = build()
    if args.check:
        if not args.output.is_file() or json.loads(args.output.read_text()) != document:
            return 1
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["worker_sizing_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
