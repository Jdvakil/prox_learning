#!/usr/bin/env python3
"""V10.10 unattended pipeline driver: conversion through paired evaluation.

Runs each stage in order, logging to its own file, and halts on the first
failure rather than carrying a broken artifact forward. Stages already
completed are skipped, so the driver can be restarted safely.

Collection is not a stage here: it runs separately and must have written its
close-out with quotas met before the driver will start.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v1010_contract import (  # noqa: E402
    COLLECTION_ROOT, CONVERTED_DATASET_ROOT, ENCODER_SHA256, EVAL_ROOT,
    TARGET_SUCCESSES, WORK_ROOT,
)

ENCODER = "/root/pact_frontend_screen_artifacts/encoder_v1/embedding_encoder_frozen.pt"
PY = sys.executable


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


def stage(name: str, command: list[str], marker: Path | None, log_dir: Path,
          timeout: int | None = None) -> bool:
    if marker is not None and marker.exists():
        print(f"[{utc()}] {name}: already complete, skipping", flush=True)
        return True
    log = log_dir / f"{name}.log"
    print(f"[{utc()}] {name}: starting", flush=True)
    started = time.monotonic()
    with log.open("w") as stream:
        completed = subprocess.run(command, cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT, check=False,
                                   timeout=timeout)
    elapsed = (time.monotonic() - started) / 60
    ok = completed.returncode == 0
    print(f"[{utc()}] {name}: exit {completed.returncode} after {elapsed:.1f} min",
          flush=True)
    if not ok:
        print(f"    see {log}", flush=True)
        print("    " + "\n    ".join(log.read_text().splitlines()[-12:]), flush=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=4)
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    closeout = ROOT / COLLECTION_ROOT / "closeout.json"
    if not closeout.is_file():
        print("collection close-out is absent; refusing to start", flush=True)
        return 1
    summary = json.loads(closeout.read_text())
    if not summary.get("quotas_met") or summary.get("accepted_total") != TARGET_SUCCESSES:
        print(f"collection did not meet quota: accepted "
              f"{summary.get('accepted_total')} of {TARGET_SUCCESSES}, "
              f"short {summary.get('cells_short')}", flush=True)
        return 1

    work = ROOT / WORK_ROOT
    scratch = args.log_dir
    stages: list[tuple[str, list[str], Path | None]] = [
        ("01_source", [PY, "scripts/verify_pact_place_v1010_source.py",
                       "--workers", str(args.workers)],
         work / "source_manifest.json"),
        ("02_convert", [PY, "scripts/convert_pact_place_v109_to_act.py",
                        "--source-manifest", str(work / "source_manifest.json"),
                        "--dst", str(ROOT / CONVERTED_DATASET_ROOT),
                        "--manifest-out", str(work / "conversion_manifest.json"),
                        "--expected-episodes", str(TARGET_SUCCESSES),
                        "--workers", "6"],
         work / "conversion_manifest.json"),
        ("03_encode", [PY, "scripts/encode_pact_embedding_tokens.py",
                       "--dataset-dir", str(ROOT / CONVERTED_DATASET_ROOT),
                       "--checkpoint", ENCODER,
                       "--conversion-manifest", str(work / "conversion_manifest.json"),
                       "--updated-conversion-manifest-out",
                       str(scratch / "conversion_manifest_encoded_raw.json"),
                       "--report-out", str(scratch / "embedding_report_raw.json"),
                       "--batch-size", "640"],
         scratch / "conversion_manifest_encoded_raw.json"),
        ("04_embeddings", [PY, "scripts/verify_pact_place_v109_embeddings.py",
                           "--dataset-dir", str(ROOT / CONVERTED_DATASET_ROOT),
                           "--conversion-manifest", str(work / "conversion_manifest.json"),
                           "--raw-encoded-manifest",
                           str(scratch / "conversion_manifest_encoded_raw.json"),
                           "--raw-encoding-report", str(scratch / "embedding_report_raw.json"),
                           "--manifest-out", str(work / "conversion_manifest_encoded.json"),
                           "--report-out", str(work / "embedding_report.json"),
                           "--expected-episodes", str(TARGET_SUCCESSES),
                           "--workers", "6"],
         work / "conversion_manifest_encoded.json"),
        ("05_split", [PY, "scripts/build_pact_place_v1010_split.py"],
         work / "split_manifest.json"),
        ("06_train_preflight", [PY, "scripts/run_pact_place_v1010_train.py",
                                "--stage", "preflight", "--log-dir", str(scratch)],
         work / "training_preflight.json"),
        ("07_train", [PY, "scripts/run_pact_place_v1010_train.py",
                      "--stage", "train", "--log-dir", str(scratch)],
         None),
        ("08_train_verify", [PY, "scripts/verify_pact_place_v1010_training.py"],
         work / "training_verification.json"),
        ("09_eval_manifest", [PY, "scripts/build_pact_place_v1010_eval_manifest.py"],
         ROOT / EVAL_ROOT / "eval_manifest.json"),
        ("10_eval_smoke", [PY, "scripts/run_pact_place_v1010_eval.py",
                           "--stage", "smoke", "--workers", "4",
                           "--save-trajectory", "--h5-only"],
         ROOT / EVAL_ROOT / "smoke_run.json"),
        ("11_eval_full", [PY, "scripts/run_pact_place_v1010_eval.py",
                          "--stage", "full", "--workers", str(args.eval_workers),
                          "--save-trajectory", "--h5-only"],
         ROOT / EVAL_ROOT / "full_run.json"),
        ("12_analysis", [PY, "scripts/finalize_pact_place_v1010_eval.py"],
         ROOT / EVAL_ROOT / "analysis.json"),
    ]
    for name, command, marker in stages:
        if not stage(name, command, marker, args.log_dir):
            print(f"PIPELINE HALTED at {name}", flush=True)
            return 1
    print(f"[{utc()}] PIPELINE COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
