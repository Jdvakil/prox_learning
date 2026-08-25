#!/usr/bin/env python3
"""Build or verify the held-out chunk-1 place evaluation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pact_place_eval_chunk1_contract import build_manifest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs" / "pact_place_eval_chunk1_manifest.json"
RECOVERY = ROOT / "configs" / "pact_place_v5_recovery.json"


def build() -> dict:
    recovery = json.loads(RECOVERY.read_text())
    names = list(recovery["recovery"]["proximity_sensor_names"])
    training_seeds = {
        int(row["task_seed_u64"]) for row in recovery["recovery_rows"]
    }
    return build_manifest(names, training_seeds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    document = build()
    if args.check:
        if not OUTPUT.exists() or json.loads(OUTPUT.read_text()) != document:
            print("place chunk-1 manifest differs from deterministic regeneration")
            return 1
        print(f"manifest OK {document['manifest_sha256']}")
        return 0
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(document["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
