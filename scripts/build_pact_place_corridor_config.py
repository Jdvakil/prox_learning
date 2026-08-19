#!/usr/bin/env python3
"""Build the deterministic Phase-0 PACT place-corridor preregistration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pact_place_corridor_contract import ROOT, build_contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSON. Frozen v1/v2/v3/v4/v5/v6 contracts cannot be overwritten once written.",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=None,
        help="Master seed for a new contract. Defaults to PACT_PLACE_MASTER_SEED or 2026081901.",
    )
    args = parser.parse_args()
    frozen = {
        (ROOT / "configs" / name).resolve()
        for name in (
            "pact_place_corridor_v1.json",
            "pact_place_corridor_v2.json",
            "pact_place_corridor_v3.json",
            "pact_place_corridor_v4.json",
            "pact_place_corridor_v5.json",
            "pact_place_corridor_v6.json",
            "pact_place_corridor_v6b.json",
            "pact_place_corridor_v6c.json",
        )
    }
    if args.output.resolve() in frozen and args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen contract {args.output}")
    document = build_contract(master_seed=args.master_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(document["config_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
