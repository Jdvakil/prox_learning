#!/usr/bin/env python3
"""Run the eight paired-side physics rows for V9.5 raw remediation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v9_contract import LAYOUT_FAMILIES, sha256_payload
from pact_place_v95_contract import build_v95_layout, load_v95_palette
from run_pact_place_v9_panel_smoke import _row, _run_job

DEFAULT_OUTPUT_ROOT = ROOT / "diagnostics_output/pact_place_v95_raw_smoke"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=955339)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    palette = load_v95_palette()
    implementation_sha256 = hashlib.sha256(
        (Path(__file__).read_bytes() + (ROOT / "scripts/pact_place_v95_contract.py").read_bytes())
    ).hexdigest()
    config_sha256 = sha256_payload(
        {
            "schema_version": "pact_place_v9_5_raw_smoke_config_v1",
            "implementation_sha256": implementation_sha256,
            "seed": args.seed,
        }
    )
    jobs = []
    for family_index, family_id in enumerate(LAYOUT_FAMILIES):
        for side_index, side in enumerate(("left", "right")):
            role_index = 600 + family_index * 2 + side_index
            row = _row(
                index=role_index,
                family_id=family_id,
                side=side,
                palette_document=palette,
                implementation_sha256=implementation_sha256,
                seed=args.seed,
            )
            row["pact_clutter_palette"] = list(palette["palette"])
            row["pact_clutter_layout"] = build_v95_layout(
                palette, family_id=family_id, intrusion_side=side
            )
            row["layout_id"] = row["pact_clutter_layout"]["layout_id"]
            row["episode_id"] = hashlib.sha256(
                f"pact-v9.5-raw:{implementation_sha256}:{family_id}:{side}:{args.seed}".encode()
            ).hexdigest()
            row.pop("row_sha256", None)
            row["row_sha256"] = sha256_payload(row)
            jobs.append(
                {
                    "role_index": role_index,
                    "family_id": family_id,
                    "side": side,
                    "row": row,
                    "config_sha256": config_sha256,
                    "output_root": str(output_root),
                }
            )
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(_run_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_map):
            item = future.result()
            print(json.dumps(item, sort_keys=True), flush=True)
            results.append(item)
    results.sort(key=lambda item: int(item["role_index"]))
    summary = {
        "schema_version": "pact_place_v9_5_raw_smoke_summary_v1",
        "role": "blocking_raw_prerequisite_physics",
        "authorizes_gate": False,
        "authorizes_collection": False,
        "seed": args.seed,
        "implementation_sha256": implementation_sha256,
        "config_sha256": config_sha256,
        "results": results,
        "manifest_rows": [job["row"] for job in jobs],
        "complete_rows": sum(item["status"] == "complete" for item in results),
        "clean_rows": sum(item["clean_success"] for item in results),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if summary["complete_rows"] == 8 else 2


if __name__ == "__main__":
    raise SystemExit(main())
