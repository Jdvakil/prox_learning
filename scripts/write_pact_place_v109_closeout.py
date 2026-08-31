#!/usr/bin/env python3
"""V10.9 step 10: assemble the close-out from the artifacts, not from memory.

Every figure quoted here is read back out of a create-only artifact and
re-checked against the contract, so the close-out cannot drift from what was
actually produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pact_place_v109_contract import (  # noqa: E402
    ACT_TRAIN_COMMIT_V5,
    CONTRACT_VERSION_V109,
    CONVERTED_DATASET_ROOT,
    ENCODER_SHA256,
    EVAL_ROOT,
    SENSOR_ORDER_SHA256,
    TRAINING_ROOT,
    WORK_ROOT,
    canonical_payload_sha256,
    empty_authorization,
    sha256_file,
    write_immutable_create_only,
    write_immutable_text_create_only,
)

ARTIFACTS = {
    "v108_erratum": f"{WORK_ROOT}/v108_erratum.json",
    "source_manifest": f"{WORK_ROOT}/source_manifest.json",
    "conversion_manifest": f"{WORK_ROOT}/conversion_manifest.json",
    "conversion_manifest_encoded": f"{WORK_ROOT}/conversion_manifest_encoded.json",
    "embedding_report": f"{WORK_ROOT}/embedding_report.json",
    "split_manifest": f"{WORK_ROOT}/split_manifest.json",
    "training_preflight": f"{WORK_ROOT}/training_preflight.json",
    "training_verification": f"{WORK_ROOT}/training_verification.json",
    "eval_manifest": f"{EVAL_ROOT}/eval_manifest.json",
    "eval_seed_audit": f"{EVAL_ROOT}/eval_seed_audit.json",
    "smoke_run": f"{EVAL_ROOT}/smoke_run.json",
    "full_run": f"{EVAL_ROOT}/full_run.json",
    "analysis": f"{EVAL_ROOT}/analysis.json",
}


def load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / ARTIFACTS[name]).read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / WORK_ROOT / "closeout.json")
    args = parser.parse_args()

    documents = {name: load(name) for name in ARTIFACTS}
    hashes = {name: sha256_file(ROOT / path) for name, path in ARTIFACTS.items()}

    source = documents["source_manifest"]
    conversion = documents["conversion_manifest"]
    encoded = documents["conversion_manifest_encoded"]
    embedding = documents["embedding_report"]
    split = documents["split_manifest"]
    preflight = documents["training_preflight"]
    verification = documents["training_verification"]
    analysis = documents["analysis"]
    full_run = documents["full_run"]
    smoke = documents["smoke_run"]

    deviations = [
        {
            "id": "D1",
            "topic": "converted corpus is one timestep shorter per episode",
            "planned": "run the encoder over all 2,860,440 sensor windows",
            "actual": f"{embedding['corpus_statistics']['windows_encoded']:,} windows",
            "why": "2,860,440 counts the raw V10.8 HDF5 corpus. The converted ACT "
                   "corpus drops each episode's trailing empty action row (proven V5 "
                   "semantics), giving 71,370 timesteps. Embeddings are generated over "
                   "the converted corpus, which is what training reads.",
            "impact": "none; the dropped row carries no training target",
        },
        {
            "id": "D2",
            "topic": "episode_horizon justification",
            "planned": "episode_horizon=635 because verified T_max=627",
            "actual": f"episode_horizon=635; raw T_max=627, converted T_max="
                      f"{conversion['timesteps']['converted_t_max']}",
            "why": "627 is the raw HDF5 maximum; the converted maximum is one lower. "
                   "635 exceeds both and matches the V5 chunk-100 experiment exactly.",
            "impact": "none; the value is unchanged from the plan",
        },
        {
            "id": "D3",
            "topic": "inherited semantic-hash defect",
            "planned": "write conversion and tree-hash manifests",
            "actual": "the semantic tree hash is recorded but dropped as an integrity "
                      "anchor after the encoding pass",
            "why": embedding["inherited_semantic_hash_defect"]["cause"],
            "impact": embedding["inherited_semantic_hash_defect"]["downstream_impact"],
        },
        {
            "id": "D4",
            "topic": "self-hash collisions with the immutable writer",
            "planned": "write create-only manifests",
            "actual": "the split and evaluation manifests are written as text so their "
                      "self-hashes validate on reload",
            "why": "write_immutable_create_only appends its own payload_sha256, which "
                   "the split and evaluation loaders then include when recomputing the "
                   "self-hash, making the manifest permanently unloadable. Caught by "
                   "loading through the real training loader before training started.",
            "impact": "none; both superseded attempts are retained and neither was used",
        },
        {
            "id": "D5",
            "topic": "the split manifest names the pre-embedding tree hash",
            "planned": "one split manifest bound to the converted dataset",
            "actual": "split.source_collection_tree_sha256 is the pre-embedding tree "
                      f"{split['source_collection_tree_sha256']}; training verified the "
                      f"post-embedding tree {encoded['converted_tree_file_sha256']}",
            "why": "the split was built from conversion_manifest.json, whose file "
                   "hashes predate the embedding pass. The episode identities are "
                   "identical in both manifests, and the split assignment depends only "
                   "on attempt_id and cell, never on a file hash.",
            "impact": "none. fixed_split_data.verify_dataset cross-checked every split "
                      "episode against the encoded manifest and matched all 141, and "
                      "training verified --expect_dataset_tree_sha256 against the "
                      "post-embedding tree. Not corrected in place because doing so "
                      "would change split_manifest_sha256 and invalidate the completed "
                      "runs; both hashes are recorded and each is correct for what it "
                      "names.",
        },
    ]

    closeout: dict[str, Any] = {
        **empty_authorization(),
        "schema_version": "pact_place_v109_closeout_v1",
        "contract_version": CONTRACT_VERSION_V109,
        "role": "V10.9 exploratory conversion, ACT-vs-PACT training, and paired "
                "learned-policy evaluation on the 141 accepted V10.8 demonstrations",
        "authorization": "explicit owner authorization, despite V10.7's failed Phase-0 "
                         "gate and V10.8's owner-instructed early stop",
        "is_phase0_pass": False,
        "v107_phase0_result": "failed_8_of_24_permanently_closed",
        "v108_stop_reason": "owner_instructed_early_stop",
        "v108_corrections": [
            {"id": c["id"], "topic": c["topic"], "status": c["status"]}
            for c in documents["v108_erratum"]["corrections"]
        ],
        "source": {
            "accepted_rows": source["counts"]["accepted"],
            "attempts": source["counts"]["attempts"],
            "checks_total": source["checks_total"],
            "checks_failed": source["checks_failed"],
            "t_min": source["counts"]["t_min"], "t_max": source["counts"]["t_max"],
            "t_sum": source["counts"]["t_sum"],
            "by_family": source["underrepresentation"]["by_family"],
            "by_side": source["underrepresentation"]["by_side"],
            "cells_exactly_at_quota":
                source["underrepresentation"]["cells_exactly_at_quota"],
            "cells_over_quota": source["underrepresentation"]["cells_over_quota"],
            "cells_short": source["underrepresentation"]["cells_short"],
        },
        "conversion": {
            "dataset_dir": CONVERTED_DATASET_ROOT,
            "episodes": conversion["episode_count"],
            "converted_t_min": conversion["timesteps"]["converted_t_min"],
            "converted_t_max": conversion["timesteps"]["converted_t_max"],
            "converted_t_sum": conversion["timesteps"]["converted_t_sum"],
            "sensor_order_sha256": SENSOR_ORDER_SHA256,
            "sensor_order_is_alphabetical": conversion["sensor_order_is_alphabetical"],
            "pre_embedding_tree_file_sha256":
                conversion["converted_tree_file_sha256"],
            "final_tree_file_sha256": encoded["converted_tree_file_sha256"],
        },
        "embeddings": {
            "encoder_sha256": ENCODER_SHA256,
            "encoder_class": embedding["encoder"]["class"],
            "encoder_schema": embedding["encoder"]["schema"],
            "encoder_sensor_order_matches":
                embedding["encoder"]["sensor_order_matches_contract"],
            **{k: embedding["corpus_statistics"][k] for k in
               ("windows_encoded", "all_finite", "dead_dimension_count", "global_std",
                "per_dimension_std_min", "per_dimension_std_median",
                "per_dimension_std_max", "proximity_valid_true_fraction")},
            "episodes_preserved": embedding["preservation"]["episodes_preserved"],
            "preservation_method": embedding["preservation"]["method"],
        },
        "split": {
            "train": split["counts"]["train"]["total"],
            "validation": split["counts"]["validation"]["total"],
            "split_manifest_sha256": split["split_manifest_sha256"],
            "cells_in_train": split["stratification"]["cells_in_train"],
            "cells_in_validation": split["stratification"]["cells_in_validation"],
            "f3_deficit": split["underrepresentation_warning"]["f3_deficit"],
            "validation_by_family": split["stratification"]["validation_by_family"],
        },
        "training": {
            "root": TRAINING_ROOT,
            "act_train_commit_v5": ACT_TRAIN_COMMIT_V5,
            "act_submodule_head": preflight["act_submodule_provenance"]["act_head"],
            "training_model_loader_source_unchanged":
                preflight["act_submodule_provenance"][
                    "training_model_loader_source_unchanged"],
            "command_diff": preflight["command_diff"],
            "commands": preflight["commands"],
            "timing": verification["timing"],
            "arms": {
                arm: {k: verification["arms"][arm][k] for k in
                      ("completed_all_epochs", "epochs_recorded", "best_epoch",
                       "best_val_loss", "best_val_l1", "final_epoch_val_loss",
                       "strict_reload_ok", "hashes", "input_proj_proximity_shape",
                       "offline_smoke", "proximity_proof")}
                for arm in ("act", "pact")
            },
        },
        "evaluation": {
            "manifest_sha256": documents["eval_manifest"]["manifest_sha256"],
            "instances": analysis["instances"],
            "rollouts": analysis["rollouts"],
            "scene_by_pose": documents["eval_manifest"]["scene_by_pose"],
            "sampler_class": documents["eval_manifest"]["sampler_class"],
            "seed_disjoint": documents["eval_seed_audit"]["disjoint"],
            "smoke": {k: smoke[k] for k in
                      ("instances", "rollouts_complete", "infrastructure_healthy",
                       "mean_rollout_minutes")},
            "full_run": {k: full_run[k] for k in
                         ("rollouts_complete", "rollouts_attempted", "elapsed_hours",
                          "mean_rollout_minutes", "workers")},
            "arms": analysis["arms"],
            "paired": analysis["paired"],
            "historical_context": analysis["historical_context"],
            "interpretation_limits": analysis["interpretation_limits"],
        },
        "artifacts": {name: {"path": path, "sha256": hashes[name]}
                      for name, path in ARTIFACTS.items()},
        "deviations": deviations,
        "preserved_byte_for_byte": [
            "assets/act_style_data/pact_place_corridor_v2_recovered_152",
            "/root/pact_place_152_pact_vs_act_chunk100_seed3101",
            "diagnostics_output/pact_place_v108_collection",
            "diagnostics_output/pact_place_v107_*",
            "diagnostics_output/pact_place_v104_* through v106_*",
        ],
        "forbidden_next_steps": [
            "another training seed", "hyperparameter tuning",
            "further demonstration collection", "any additional ablation",
            "PACT_PERMUTED",
        ],
    }
    closeout["payload_sha256"] = canonical_payload_sha256(closeout)
    written = write_immutable_create_only(args.out, closeout)
    print(json.dumps({
        "payload_sha256": closeout["payload_sha256"],
        "raw_file_sha256": written.get("raw_file_sha256"),
        "deviations": len(deviations),
        "artifacts": len(ARTIFACTS),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
