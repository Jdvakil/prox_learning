#!/usr/bin/env python3
"""Extract per-row ceiling-pendant bow diagnostics from a V9.8 screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_corridor_contract import sha256_payload  # noqa: E402


def _bow_block(result: dict[str, Any], prefix: str) -> dict[str, Any]:
    pendant = result.get("pendant_bow") or {}
    key = "inbound" if prefix.startswith("inbound") else "outbound"
    block = dict(pendant.get(key) or {})
    if not block:
        block = dict((result.get("bow_diagnostics") or {}).get(prefix) or {})
    return block


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    rows = []
    for path in sorted((source_root / "expert_screen_rows").glob("*/result.json")):
        result = json.loads(path.read_text())
        inbound = _bow_block(result, "inbound_ceiling_fixture")
        outbound = _bow_block(result, "outbound_ceiling_fixture")
        rows.append(
            {
                "role_index": result.get("role_index"),
                "episode_id": result.get("episode_id"),
                "intrusion_side": result.get("intrusion_side"),
                "clean_success": bool(result.get("clean_success")),
                "task_success": bool(result.get("task_success")),
                "status": result.get("status"),
                "terminal_policy_phase": result.get("terminal_policy_phase"),
                "planned_bow_m": {
                    "inbound_ceiling_fixture": inbound.get("planned_bow_m"),
                    "outbound_ceiling_fixture": outbound.get("planned_bow_m"),
                },
                "accepted_bow_m": {
                    "inbound_ceiling_fixture": inbound.get("accepted_bow_m"),
                    "outbound_ceiling_fixture": outbound.get("accepted_bow_m"),
                },
                "bow_fallback_taken": {
                    "inbound_ceiling_fixture": bool(inbound.get("bow_fallback_taken")),
                    "outbound_ceiling_fixture": bool(outbound.get("bow_fallback_taken")),
                },
            }
        )
    document = {
        "schema_version": "pact_place_v9_8_pendant_bow_attribution_v1",
        "source_root": str(source_root.relative_to(ROOT)),
        "n": len(rows),
        "rows": rows,
    }
    document["document_sha256"] = sha256_payload(document)
    output = args.output or (source_root / "bow_attribution.json")
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
