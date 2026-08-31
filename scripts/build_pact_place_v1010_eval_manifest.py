#!/usr/bin/env python3
"""V10.10: build and freeze the paired 40-instance evaluation manifest."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pact_place_v109_contract import CANONICAL_SENSOR_NAMES, SENSOR_ORDER_SHA256  # noqa: E402
from pact_place_v1010_contract import (  # noqa: E402
    COLLECTION_ROOT, EVAL_ROOT, WORK_ROOT, canonical_payload_sha256, cell_seed,
    cells, empty_authorization, write_immutable_create_only,
)
from pact_place_v1010_eval_contract import build_manifest, load_manifest  # noqa: E402

MAX_ATTEMPT_INDEX = 64


def historical_seeds() -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    ledger = ROOT / COLLECTION_ROOT / "ledger.jsonl"
    if ledger.is_file():
        rows = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
        out["v1010_collection_attempts"] = [int(r["task_seed_u32"]) for r in rows]
    out["v1010_collection_stream_superset"] = [
        int(cell_seed(f, s, p, i)["seed_u32"])
        for f, s, p in cells() for i in range(MAX_ATTEMPT_INDEX)]
    for name, path, key in (
        ("v108_scientific", "diagnostics_output/pact_place_v108_collection/ledger.jsonl", None),
        ("v107_pool", "diagnostics_output/pact_place_v107_pool/pool_manifest.json", "expert_screen_rows"),
        ("v107_phase0", "diagnostics_output/pact_place_v107_phase0/gate_manifest.json", "expert_screen_rows"),
    ):
        p = ROOT / path
        if not p.is_file():
            continue
        if key is None:
            rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        else:
            rows = json.loads(p.read_text())[key]
        out[name] = [int(r["task_seed_u32"]) for r in rows]
    for name, path in (("v109_eval", "diagnostics_output/pact_place_v109_eval/eval_manifest.json"),):
        p = ROOT / path
        if p.is_file():
            d = json.loads(p.read_text())
            out[name] = [int(r["task_seed_u32"])
                         for r in d["rows"] + d["smoke"]["rows"]]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / EVAL_ROOT / "eval_manifest.json")
    args = parser.parse_args()
    excluded = historical_seeds()
    manifest = build_manifest(list(CANONICAL_SENSOR_NAMES), SENSOR_ORDER_SHA256, excluded)
    if not manifest["valid"]:
        print(json.dumps({"valid": False, "problems": manifest["problems"]}, indent=2))
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = write_immutable_create_only(args.out, manifest)
    reloaded = load_manifest(args.out)
    audit = {**empty_authorization(),
             "schema_version": "pact_place_v1010_eval_seed_audit_v1",
             "manifest_sha256": manifest["manifest_sha256"],
             "manifest_raw_file_sha256": written.get("raw_file_sha256"),
             "excluded_source_sizes": {k: len(v) for k, v in excluded.items()},
             "excluded_unique_seeds": len({s for v in excluded.values() for s in v}),
             "collisions": manifest["held_out_seed_audit"]["collisions"],
             "disjoint": manifest["held_out_seed_audit"]["disjoint"]}
    audit["payload_sha256"] = canonical_payload_sha256(audit)
    write_immutable_create_only(ROOT / EVAL_ROOT / "eval_seed_audit.json", audit)
    print(json.dumps({"valid": reloaded["valid"], "instances": reloaded["total_candidates"],
                      "balance": reloaded["balance"]["by_family"],
                      "seed_disjoint": audit["disjoint"],
                      "excluded_unique_seeds": audit["excluded_unique_seeds"],
                      "manifest_sha256": reloaded["manifest_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
