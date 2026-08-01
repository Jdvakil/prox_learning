#!/usr/bin/env python3
"""Outcome-exploratory power calculation for the contact endpoint.

Only frozen prior runs are read.  Full archived JSON is scanned as a stream so
the large per-physics-step contact payload is never materialized in memory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import zstandard
from scipy.stats import norm


SCHEMA_VERSION = "pact_contact_endpoint_power_v1"
ARMS = ("ACT", "PACT", "PACT_PERMUTED")
SEEDS = (3101, 3102)
ALPHA = 0.05
POWER = 0.80
CHOSEN_INSTANCES = 100
FRAME_PATTERN = re.compile(rb'"frames_with_contact"\s*:\s*(\{[^}]*\})')


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_frames(stream: BinaryIO) -> dict[str, int]:
    buffer = b""
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        buffer += chunk
        match = FRAME_PATTERN.search(buffer)
        if match:
            value = json.loads(match.group(1))
            return {key: int(item) for key, item in value.items()}
        buffer = buffer[-4096:]
    raise ValueError("frames_with_contact not found in archived result")


def frames_from_result(result_path: Path) -> dict[str, int]:
    archive = result_path.with_name("result.full.json.zst")
    if archive.is_file():
        with archive.open("rb") as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
                return _scan_frames(stream)
    with result_path.open("rb") as stream:
        return _scan_frames(stream)


def source_rows(schedule: dict[str, Any], replication_root: Path) -> list[dict[str, Any]]:
    rows = []
    for reference in schedule["seed_3101_references"]:
        rows.append(
            {
                "checkpoint_seed": 3101,
                "episode_id": reference["instance_episode_id"],
                "arm": reference["arm"],
                "result_path": Path(reference["result_path"]),
            }
        )
    for row in schedule["rows"]:
        rows.append(
            {
                "checkpoint_seed": 3102,
                "episode_id": row["instance_episode_id"],
                "arm": row["arm"],
                "result_path": replication_root / row["output_relpath"] / "result.json",
            }
        )
    return rows


def _load_row(row: dict[str, Any]) -> dict[str, Any]:
    frames = frames_from_result(row["result_path"])
    return {
        "checkpoint_seed": row["checkpoint_seed"],
        "episode_id": row["episode_id"],
        "arm": row["arm"],
        "hazard_bar_contact_frames": frames["hazard_bar"],
        "other_environment_contact_frames": frames["other_environment"],
    }


def summarize(values: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sample_sd": float(np.std(values, ddof=1)),
        "nonzero": int(np.count_nonzero(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def mde(sample_sd: float, instances: int) -> float:
    multiplier = norm.ppf(1.0 - ALPHA / 2.0) + norm.ppf(POWER)
    return float(multiplier * sample_sd / math.sqrt(instances))


def instances_for_effect(sample_sd: float, effect: float) -> int | None:
    if effect <= 0.0:
        return None
    multiplier = norm.ppf(1.0 - ALPHA / 2.0) + norm.ppf(POWER)
    return int(math.ceil((multiplier * sample_sd / effect) ** 2))


def build_document(
    schedule_path: Path, replication_root: Path, *, workers: int
) -> dict[str, Any]:
    schedule = json.loads(schedule_path.read_text())
    rows = source_rows(schedule, replication_root)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        loaded = list(executor.map(_load_row, rows))
    counts = Counter((row["checkpoint_seed"], row["arm"]) for row in loaded)
    expected = Counter({(seed, arm): 40 for seed in SEEDS for arm in ARMS})
    if counts != expected:
        raise ValueError(f"prior-run cell mismatch: {counts}")

    arm_statistics: dict[str, Any] = {}
    paired_statistics: dict[str, Any] = {}
    by_identity = {
        (row["checkpoint_seed"], row["episode_id"], row["arm"]): row
        for row in loaded
    }
    episode_ids = sorted({row["episode_id"] for row in loaded})
    if len(episode_ids) != 40:
        raise ValueError(f"expected 40 prior instances, got {len(episode_ids)}")
    for seed in SEEDS:
        arm_statistics[str(seed)] = {}
        paired_statistics[str(seed)] = {}
        for arm in ARMS:
            values = np.asarray(
                [
                    by_identity[(seed, episode_id, arm)]["hazard_bar_contact_frames"]
                    for episode_id in episode_ids
                ],
                dtype=np.float64,
            )
            arm_statistics[str(seed)][arm] = summarize(values)
        for arm_a, arm_b in (("PACT", "PACT_PERMUTED"), ("PACT", "ACT")):
            differences = np.asarray(
                [
                    by_identity[(seed, episode_id, arm_a)][
                        "hazard_bar_contact_frames"
                    ]
                    - by_identity[(seed, episode_id, arm_b)][
                        "hazard_bar_contact_frames"
                    ]
                    for episode_id in episode_ids
                ],
                dtype=np.float64,
            )
            paired_statistics[str(seed)][f"{arm_a}_minus_{arm_b}"] = summarize(
                differences
            )

    cluster_differences = np.asarray(
        [
            np.mean(
                [
                    by_identity[(seed, episode_id, "PACT")][
                        "hazard_bar_contact_frames"
                    ]
                    - by_identity[(seed, episode_id, "PACT_PERMUTED")][
                        "hazard_bar_contact_frames"
                    ]
                    for seed in SEEDS
                ]
            )
            for episode_id in episode_ids
        ],
        dtype=np.float64,
    )
    binary_cluster_differences = np.asarray(
        [
            np.mean(
                [
                    float(
                        by_identity[(seed, episode_id, "PACT")][
                            "hazard_bar_contact_frames"
                        ]
                        > 0
                    )
                    - float(
                        by_identity[(seed, episode_id, "PACT_PERMUTED")][
                            "hazard_bar_contact_frames"
                        ]
                        > 0
                    )
                    for seed in SEEDS
                ]
            )
            for episode_id in episode_ids
        ],
        dtype=np.float64,
    )
    contact_summary = summarize(cluster_differences)
    binary_summary = summarize(binary_cluster_differences)
    observed_contact_effect = abs(contact_summary["mean"])
    observed_binary_effect = abs(binary_summary["mean"])
    contact_required = instances_for_effect(
        contact_summary["sample_sd"], observed_contact_effect
    )
    binary_required = instances_for_effect(
        binary_summary["sample_sd"], observed_binary_effect
    )
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "exploratory_prior_data_for_design_only",
        "source": {
            "schedule_path": str(schedule_path),
            "schedule_sha256": schedule["schedule_sha256"],
            "schedule_file_sha256": sha256_file(schedule_path),
            "replication_root": str(replication_root),
            "rows": len(loaded),
            "instances": len(episode_ids),
            "checkpoint_seeds": list(SEEDS),
            "arms": list(ARMS),
            "endpoint_fields_loaded": ["contact_audit.frames_with_contact"],
        },
        "arm_statistics": arm_statistics,
        "paired_statistics": paired_statistics,
        "two_seed_instance_cluster": {
            "PACT_minus_PACT_PERMUTED_contact_frames": contact_summary,
            "PACT_minus_PACT_PERMUTED_any_contact": binary_summary,
            "cluster_unit": "instance; both prior policy seeds move together",
        },
        "power": {
            "method": "two-sided normal approximation for a paired mean, using the sample SD across prior two-seed instance-cluster means",
            "alpha_two_sided": ALPHA,
            "target_power": POWER,
            "chosen_fresh_instances": CHOSEN_INSTANCES,
            "contact_frame_mde_at_chosen_n": mde(
                contact_summary["sample_sd"], CHOSEN_INSTANCES
            ),
            "historical_absolute_contact_frame_difference": observed_contact_effect,
            "instances_for_historical_contact_frame_difference": contact_required,
            "any_contact_probability_mde_at_chosen_n": mde(
                binary_summary["sample_sd"], CHOSEN_INSTANCES
            ),
            "historical_absolute_any_contact_probability_difference": observed_binary_effect,
            "instances_for_historical_any_contact_probability_difference": binary_required,
            "comparison": (
                f"The count endpoint requires approximately {contact_required} instances "
                f"for the historical effect versus {binary_required} for the binary any-contact "
                "endpoint under the same approximation."
            ),
            "choice_rationale": (
                "n=100 is the rounded-up design size that reaches approximately 80% power "
                "for the historical two-seed PACT-versus-permuted contact-frame reduction."
            ),
        },
    }
    document["power_sha256"] = sha256_payload(document)
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedule", default="diagnostics_output/pact_seed_replication/schedule.json"
    )
    parser.add_argument(
        "--replication-root",
        default="/root/pact_seed_replication_artifacts/evaluation_1490160c",
    )
    parser.add_argument(
        "--output", default="diagnostics_output/pact_contact_endpoint/power.json"
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = build_document(
        Path(args.schedule), Path(args.replication_root), workers=args.workers
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")
    print(json.dumps(document["power"], indent=2, sort_keys=True))
    print(f"sha256={document['power_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
