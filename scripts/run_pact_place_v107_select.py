#!/usr/bin/env python3
"""V10.7 Step 2: re-derive selection from the sealed V10.6 scores.

The winning bundle is not written down anywhere in this file. It is whatever
the registered risk-aligned ranking produces from the sealed row scores, and
the runner asserts that its own selection equals the independently recomputed
argmin of that ranking. Agreement with any externally expected bundle is
recorded as an observation, never as a gate.

The score NPZ is written before the manifest JSON so the JSON can bind its raw
SHA-256.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "submodules" / "molmospaces"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pact_place_v107_contract import (  # noqa: E402
    CONTRACT_VERSION_V107,
    ENVIRONMENT_VERSION,
    RISK_BAND_M,
    SELECTION_ROOT,
    SPEC_ROOT,
    V106_SITING_JSON,
    V106_SITING_NPZ,
    assert_no_drift,
    candidate_statistics,
    empty_authorization,
    is_qualified,
    recompute_payload_sha256,
    risk_aligned_rank_key,
    sha256_file,
    write_immutable_create_only,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / SELECTION_ROOT)
    parser.add_argument("--specification", type=Path,
                        default=ROOT / SPEC_ROOT / "specification.json")
    args = parser.parse_args()
    started = time.time()

    spec = json.loads(args.specification.resolve().read_text())
    drift = assert_no_drift(spec)

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)

    siting_path = ROOT / V106_SITING_JSON
    npz_path = ROOT / V106_SITING_NPZ
    siting = json.loads(siting_path.read_text())
    per_row = [
        json.loads(str(item))
        for item in np.load(npz_path, allow_pickle=True)["rows"]
    ]
    keys = sorted(siting["bundles"])
    stats = {key: candidate_statistics(per_row, key) for key in keys}
    qualification = {key: is_qualified(stats[key]) for key in keys}

    ranked = sorted(keys, key=lambda key: risk_aligned_rank_key(stats[key]))
    qualified = [key for key in ranked if qualification[key]["qualified"]]
    selected_key = qualified[0] if qualified else None

    # Independent recomputation of the argmin, deliberately not reusing the
    # sorted list above.
    best_key, best_value = None, None
    for key in keys:
        if not qualification[key]["qualified"]:
            continue
        value = risk_aligned_rank_key(stats[key])
        if best_value is None or value < best_value:
            best_key, best_value = key, value
    if selected_key != best_key:
        raise RuntimeError(
            f"ranking disagreement: sorted={selected_key!r} argmin={best_key!r}"
        )

    if selected_key is None:
        document = {
            "schema_version": "pact_place_v107_selection_v1",
            "contract_version": CONTRACT_VERSION_V107,
            "stop_reason": "no_bundle_met_the_registered_relevance_test",
            "qualification": qualification,
            **empty_authorization(),
            "selection_passed": False,
        }
        write_immutable_create_only(output_root / "selection.json", document)
        print(json.dumps({"selection_passed": False}, indent=2))
        return 1

    chosen = stats[selected_key]
    x_m, r_neg, r_pos = (float(v) for v in selected_key.split("|"))
    minima = [chosen["group_minimum_m"][k] for k in sorted(chosen["group_minimum_m"])]

    # --- score arrays first, so the JSON can bind their raw hash ------------
    group_names = sorted(chosen["group_minimum_m"])
    npz_out = output_root / "selection_scores.npz"
    np.savez_compressed(
        npz_out,
        bundle_key=np.array(keys, dtype=object),
        universal=np.array(
            [stats[k]["universal_clearance"] for k in keys], dtype=bool
        ),
        all_minima_in_band=np.array(
            [stats[k]["all_group_minima_in_band"] for k in keys], dtype=bool
        ),
        qualified=np.array(
            [qualification[k]["qualified"] for k in keys], dtype=bool
        ),
        absolute_min_clearance_m=np.array(
            [stats[k]["absolute_min_clearance_m"] or np.nan for k in keys],
            dtype=np.float64,
        ),
        band_evaluations_total=np.array(
            [stats[k]["band_evaluations_total"] for k in keys], dtype=np.int64
        ),
        mean_group_minimum_m=np.array(
            [stats[k]["mean_group_minimum_m"] or np.nan for k in keys],
            dtype=np.float64,
        ),
        n_below_floor=np.array(
            [stats[k]["n_below_floor"] for k in keys], dtype=np.int64
        ),
        n_contacts=np.array([stats[k]["n_contacts"] for k in keys], dtype=np.int64),
        n_evaluations=np.array(
            [stats[k]["n_evaluations"] for k in keys], dtype=np.int64
        ),
        rank_position=np.array([ranked.index(k) for k in keys], dtype=np.int64),
        selected_group_names=np.array(group_names, dtype=object),
        selected_group_minimum_m=np.array(
            [chosen["group_minimum_m"][g] for g in group_names], dtype=np.float64
        ),
        selected_group_band_evaluations=np.array(
            [chosen["group_band_evaluations"][g] for g in group_names],
            dtype=np.int64,
        ),
        allow_pickle=True,
    )
    npz_sha = sha256_file(npz_out)

    document = {
        "schema_version": "pact_place_v107_selection_v1",
        "contract_version": CONTRACT_VERSION_V107,
        "environment_version": ENVIRONMENT_VERSION,
        "specification_payload_sha256": recompute_payload_sha256(
            args.specification.resolve()
        ),
        "specification_raw_file_sha256": sha256_file(args.specification.resolve()),
        "drift_check": drift,
        "source": {
            "v106_siting_json": V106_SITING_JSON,
            "v106_siting_raw_file_sha256": sha256_file(siting_path),
            "v106_siting_payload_sha256": recompute_payload_sha256(siting_path),
            "v106_scores_npz": V106_SITING_NPZ,
            "v106_scores_raw_file_sha256": sha256_file(npz_path),
            "n_rows": len(per_row),
        },
        "selection_scores_npz": "selection_scores.npz",
        "selection_scores_raw_file_sha256": npz_sha,
        "ranking": {
            "name": "risk_aligned_v1",
            "keys_most_significant_first": [
                "universal >=15 mm clearance",
                "all six group minima inside 15-35 mm",
                "more risk-band evaluations",
                "smaller mean group minimum",
                "only then, more absolute clearance",
                "deterministic radii tie-break",
            ],
            "additional_clearance_is_demoted_below_risk": True,
            "truncated": False,
            "hardcoded_expected_bundle": False,
        },
        "candidates": {
            key: {
                **{k: v for k, v in stats[key].items() if k != "witnesses"},
                "qualification": qualification[key],
                "rank_key": list(risk_aligned_rank_key(stats[key])),
                "rank_position": ranked.index(key),
            }
            for key in keys
        },
        "ranked": ranked,
        "n_qualified": len(qualified),
        "selected_key": selected_key,
        "selected": {
            "x_m": x_m, "r_neg_m": r_neg, "r_pos_m": r_pos,
            "absolute_min_clearance_m": chosen["absolute_min_clearance_m"],
            "n_evaluations": chosen["n_evaluations"],
            "n_below_floor": chosen["n_below_floor"],
            "n_contacts": chosen["n_contacts"],
            "band_evaluations_total": chosen["band_evaluations_total"],
            "mean_group_minimum_m": chosen["mean_group_minimum_m"],
            "group_minimum_m": chosen["group_minimum_m"],
            "group_band_evaluations": chosen["group_band_evaluations"],
            "direction_band_witnesses": chosen["direction_band_witnesses"],
        },
        "selected_witnesses": chosen["witnesses"],
        "argmin_agrees_with_sorted_ranking": True,
        "risk_band_m": list(RISK_BAND_M),
        "all_six_group_minima_in_band": bool(
            len(minima) == 6
            and all(RISK_BAND_M[0] <= v <= RISK_BAND_M[1] for v in minima)
        ),
        "group_minima_mm": [round(v * 1000.0, 6) for v in minima],
        "creates_episode": False, "calls_env_step": False,
        "elapsed_s": time.time() - started,
        **empty_authorization(),
        "selection_passed": True,
    }
    hashes = write_immutable_create_only(output_root / "selection.json", document)
    print(json.dumps({
        "selection_passed": True,
        "selected_key": selected_key,
        "n_qualified": len(qualified),
        "absolute_min_clearance_mm": chosen["absolute_min_clearance_m"] * 1000.0,
        "band_evaluations_total": chosen["band_evaluations_total"],
        "all_six_group_minima_in_band": document["all_six_group_minima_in_band"],
        "group_minima_mm": document["group_minima_mm"],
        "selection_scores_npz_sha256": npz_sha,
        **hashes,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
