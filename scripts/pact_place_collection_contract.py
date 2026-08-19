#!/usr/bin/env python3
"""Frozen collection contract for the passed PACT place-corridor Phase 0."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from pact_place_corridor_contract import (
    MIN_CLEAN_SUCCESSES,
    ROOT,
    load_contract,
    sha256_file,
    sha256_payload,
    validate_contract as validate_screen_contract,
)

COLLECTION_MASTER_SEED = 2026082301
N_CANDIDATES = 310
TARGET_CLEAN = 255
SCREEN_CLEAN = 22
SCREEN_N = 24
YIELD_FLOOR_DROP = 0.10
YIELD_BURN_IN = 48
COLLECTION_CONFIG = ROOT / "configs" / "pact_place_corridor_v5_collection.json"
SCREEN_CONFIG = ROOT / "configs" / "pact_place_corridor_v5.json"


def _source_hashes(screen: dict[str, Any]) -> dict[str, str]:
    paths = [
        "scripts/pact_place_collection_contract.py",
        "scripts/run_pact_place_collection.py",
        "scripts/run_pact_place_expert_screen.py",
        "submodules/molmospaces/molmo_spaces/data_generation/custom_scenes/pact_place_corridor_v2.xml",
        "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py",
        "submodules/molmospaces/molmo_spaces/tasks/pact_place_contact_audit.py",
    ]
    hashes = {relative: sha256_file(ROOT / relative) for relative in paths}
    hashes["configs/pact_place_corridor_v5.json"] = screen["config_sha256"]
    return hashes


def row_seed(index: int, master_seed: int) -> tuple[int, int]:
    import hashlib

    digest = hashlib.sha256(
        f"pact-place-v5-collect:{master_seed}:{index}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big"), int.from_bytes(digest[:8], "big")


def episode_id_for(index: int, side: str, master_seed: int) -> str:
    import hashlib

    return hashlib.sha256(
        f"pact-place-v5:collect:{master_seed}:{index}:{side}".encode()
    ).hexdigest()


def build_collection_contract() -> dict[str, Any]:
    screen = load_contract(SCREEN_CONFIG)
    validate_screen_contract(screen)
    if screen["config_sha256"] != (
        "bd47f1c97d2815657211085590657f5211ca847b776f6039c9617f990da9c1f1"
    ):
        raise ValueError("collection must pin the frozen v5 screen contract")
    rng = random.Random(COLLECTION_MASTER_SEED)
    sides = ["left", "right"] * (N_CANDIDATES // 2)
    rng.shuffle(sides)
    rows = []
    for index, side in enumerate(sides):
        seed_u32, seed_u64 = row_seed(index, COLLECTION_MASTER_SEED)
        row = {
            "role_index": index,
            "role": "place_corridor_v5_collection",
            "episode_id": episode_id_for(index, side, COLLECTION_MASTER_SEED),
            "intrusion_side": side,
            "panel_x_jitter_m": round(rng.uniform(-0.015, 0.015), 9),
            "panel_face_jitter_m": round(rng.uniform(-0.005, 0.005), 9),
            "scene_template_house_index": 1,
            "task_seed_u32": seed_u32,
            "task_seed_u64": seed_u64,
            "max_sampling_retries": 4,
        }
        row["row_sha256"] = sha256_payload(row)
        rows.append(row)
    document: dict[str, Any] = {
        "schema_version": "pact_place_corridor_collection_v5",
        "status": "collection_preregistered",
        "created_utc": "2026-08-18T00:00:00Z",
        "master_seed": COLLECTION_MASTER_SEED,
        "screen_config_sha256": screen["config_sha256"],
        "scene": dict(screen["scene"]),
        "success_criterion": dict(screen["success_criterion"]),
        "expert": dict(screen["expert"]),
        "collection": {
            "n_candidates": N_CANDIDATES,
            "target_clean": TARGET_CLEAN,
            "do_not_lower_the_filter": True,
            "clean_success_definition": screen["phase0_gate"][
                "clean_success_definition"
            ],
            "screen_clean_successes": SCREEN_CLEAN,
            "screen_n": SCREEN_N,
            "screen_clean_rate": SCREEN_CLEAN / SCREEN_N,
            "yield_floor": (SCREEN_CLEAN / SCREEN_N) - YIELD_FLOOR_DROP,
            "yield_burn_in": YIELD_BURN_IN,
            "stop_if_running_rate_below_floor_after_burn_in": True,
            "encoder_training_eval_not_authorized": True,
            "endpoint_scalars_required": True,
            "log_discarded_failure_class_grasp_pose_panel_geometry": True,
            "minimum_clean_successes_unchanged": MIN_CLEAN_SUCCESSES,
        },
        "collection_rows": rows,
        "source_sha256": _source_hashes(screen),
        "protected_artifact_sha256_before": dict(
            screen["protected_artifact_sha256_before"]
        ),
    }
    document["config_sha256"] = sha256_payload(document)
    validate_collection_contract(document)
    return document


def validate_collection_contract(document: dict[str, Any]) -> None:
    payload = dict(document)
    observed = payload.pop("config_sha256")
    if observed != sha256_payload(payload):
        raise ValueError("collection config self-hash mismatch")
    rows = document["collection_rows"]
    if len(rows) != N_CANDIDATES:
        raise ValueError(f"expected {N_CANDIDATES} collection rows")
    if sum(row["intrusion_side"] == "left" for row in rows) != N_CANDIDATES // 2:
        raise ValueError("collection is not side-balanced")
    if len({row["episode_id"] for row in rows}) != N_CANDIDATES:
        raise ValueError("collection episode IDs are not unique")
    for row in rows:
        payload = dict(row)
        row_hash = payload.pop("row_sha256")
        if row_hash != sha256_payload(payload):
            raise ValueError(f"row self-hash mismatch at {row['role_index']}")
        if not -0.015 <= row["panel_x_jitter_m"] <= 0.015:
            raise ValueError("panel x jitter outside frozen corridor support")
        if not -0.005 <= row["panel_face_jitter_m"] <= 0.005:
            raise ValueError("panel face jitter outside frozen corridor support")
    scene = document["scene"]
    if not scene["place_tray_x_bounds_m"][1] < scene["aperture_plane_x_m"]:
        raise ValueError("place tray is not wholly outside the aperture")


def load_collection_contract(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    validate_collection_contract(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=COLLECTION_CONFIG,
    )
    args = parser.parse_args()
    if args.output.resolve() == COLLECTION_CONFIG.resolve() and args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen collection contract {args.output}")
    document = build_collection_contract()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(document["config_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
