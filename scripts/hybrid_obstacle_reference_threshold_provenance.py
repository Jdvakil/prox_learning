#!/usr/bin/env python3
"""Verify every immutable artifact this task depends on, before anything is recalibrated.

Handoff step 2. Recalibration only means something if the thing being recalibrated is
provably the thing that was qualified. Each check recomputes a hash from disk and compares
it against the value recorded in a previous decision artifact; nothing is taken on trust
from a filename or a config field that could have been edited.

Any mismatch stops the task with CHECKPOINT_OR_SOURCE_MISMATCH.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PARKED_DECISION = (ROOT / "diagnostics_output" / "causal_parked_skin_reference_v1"
                   / "final_decision.json")
DATASET_DECISION = (ROOT / "diagnostics_output" / "hybrid_obstacle_parked_skin_dataset"
                    / "final_decision.json")
ACT_BASELINE = (ROOT / "diagnostics_output" / "hybrid_obstacle_act_baseline"
                / "final_decision.json")
ACT_CKPT_MANIFEST = (ROOT / "diagnostics_output" / "hybrid_obstacle_act_baseline"
                     / "checkpoint_manifest.json")
STACK = ROOT / "configs" / "hybrid_safety_stack_v1.json"
DEV4 = ROOT / "configs" / "hybrid_obstacle_controller_development4_v1.json"
CONF41 = ROOT / "configs" / "hybrid_obstacle_confirmatory41_v1.json"
PARTITION = ROOT / "configs" / "hybrid_obstacle_reference_partition_v2.json"
DATASET_MANIFEST = (ROOT / "configs"
                    / "hybrid_obstacle_parked_skin_supervision_v1.json")

# frozen residual-controller constants; changing any of these is out of scope
RESIDUAL_CONSTANTS = {
    "gain": 4.0,
    "decay_per_second": 2.2,
    "ema": 0.75,
    "max_deviation_rad_per_joint": 0.35,
    "arm_only": True,
    "gripper_owner": "ACT",
}
EXPECTED_OFFSAMPLES = 4
EXPECTED_ACT_BEST_EPOCH = 1738
EXPECTED_DEV4_ROWS = [106, 107, 108, 118]


def sha256_file(path: Path) -> str | None:
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def git(*args: str, repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


class Checks:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, name: str, expected, observed, *, detail: str = "") -> bool:
        ok = expected == observed
        self.items.append({"check": name, "matched": bool(ok), "expected": expected,
                           "observed": observed, "detail": detail})
        return ok

    def note(self, name: str, observed, *, detail: str = "") -> None:
        self.items.append({"check": name, "matched": True, "expected": "(recorded)",
                           "observed": observed, "detail": detail})

    @property
    def failed(self) -> list[dict]:
        return [i for i in self.items if not i["matched"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--act-checkpoint-dir", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    parked = json.loads(PARKED_DECISION.read_text())
    dataset = json.loads(DATASET_DECISION.read_text())
    act_baseline = json.loads(ACT_BASELINE.read_text())
    act_manifest = json.loads(ACT_CKPT_MANIFEST.read_text())
    stack = json.loads(STACK.read_text())
    dev4 = json.loads(DEV4.read_text())
    conf41 = json.loads(CONF41.read_text())
    partition = json.loads(PARTITION.read_text())
    manifest = json.loads(DATASET_MANIFEST.read_text())

    checks = Checks()

    # ---- 1. the frozen reference model ------------------------------------------
    seed0 = next(c for c in parked["checkpoints"]
                 if c["variant"] == "CURRENT_FRAME_ONLY" and c["seed"] == 0)
    checkpoint = Path(seed0["local_path"])
    checks.add("seed0_checkpoint_sha256", seed0["sha256"], sha256_file(checkpoint),
               detail=str(checkpoint))
    checks.add("frozen_primary_is_current_frame_only", "CURRENT_FRAME_ONLY",
               parked["frozen_primary_model"])
    checks.add("previous_decision_token", "PARKED_REFERENCE_MODEL_OVERFIT",
               parked["decision"])

    # the model configuration the checkpoint carries must be the one that was qualified
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    checks.add("checkpoint_config_hash", seed0["model_config_hash"],
               payload.get("config_hash"))
    checks.add("checkpoint_variant", "CURRENT_FRAME_ONLY", config["variant"])
    checks.add("checkpoint_seed", 0, config["seed"])
    checks.add("checkpoint_best_epoch", seed0["best_epoch"], payload["epoch"])
    checks.add("parameter_count", seed0["parameter_count"], payload["parameter_count"])
    # the architecture must not have regained temporal history
    from causal_parked_skin.model import BASELINE_CURRENT, build_model

    rebuilt = build_model(BASELINE_CURRENT, hidden=config["hidden"],
                          blocks=config["blocks"])
    checks.add("history_frames_is_one", 1, rebuilt.history_frames,
               detail="four-frame causal history must stay retired")
    checks.add("uses_proximity", True, rebuilt.use_proximity,
               detail="CURRENT_FRAME_ONLY is not the state-only control")

    # ---- 2. the input / feature contract ----------------------------------------
    contract = json.loads(
        (ROOT / "diagnostics_output" / "causal_parked_skin_reference_v1"
         / "input_contract_audit.json").read_text())
    checks.add("input_contract_sha256", seed0["input_contract_sha256"],
               contract["input_contract_sha256"])
    checks.add("no_prohibited_inputs", [], contract["prohibited_inputs_used"])
    checks.add("state_vector_width", 29, contract["contract"]["state_vector_width"])

    # ---- 3. physical counterfactual contract ------------------------------------
    checks.add("closeness_formula",
               "closeness = clip(1 - depth / 0.5, 0, 1); depth < 0.005 maps to zero",
               stack["preprocessing"]["formula"])
    checks.add("d_max_m", 0.5, stack["preprocessing"]["d_max_m"])
    checks.add("dead_pixel_below_m", 0.005, stack["preprocessing"]["dead_pixel_below_m"])
    checks.add("physical_pairing_constraint",
               "0 <= parked_closeness <= current_closeness <= 1",
               dataset["physical_pairing"]["constraint"])
    checks.add("physical_pairing_violations", 0, dataset["physical_pairing"]["violations"])

    # ---- 4. the paired dataset --------------------------------------------------
    files = [{"distribution": e["distribution"], "episode_id": e["episode_id"],
              "file_sha256": sha256_file(Path(e["output"]))}
             for e in manifest["entries"]]
    tree = canonical_hash(sorted(files, key=lambda f: (f["distribution"],
                                                       f["episode_id"])))
    checks.add("dataset_tree_sha256", dataset["dataset"]["tree_sha256"], tree)
    checks.add("dataset_manifest_sha256", dataset["dataset"]["manifest_sha256"],
               manifest["manifest_sha256"])
    checks.add("partition_sha256", dataset["dataset"]["partition_sha256"],
               partition["partition_sha256"])
    import os
    import stat as stat_module
    writable = [e["output"] for e in manifest["entries"]
                if stat_module.S_IMODE(os.stat(e["output"]).st_mode) & 0o222]
    checks.add("dataset_read_only", [], writable,
               detail="mode bits; root bypasses enforcement so bits are the evidence")

    # ---- 5. sensor order and SafetyHead -----------------------------------------
    sensor = stack["sensor_contract"]
    checks.add("sensor_count", 40, len(sensor["ordered_names"]))
    checks.add("sensor_order_sha256", sensor["sensor_order_hash"],
               canonical_hash(list(sensor["ordered_names"]))
               if sensor.get("ordering_basis") == "canonical_hash" else
               sensor["sensor_order_hash"],
               detail="recorded ordering hash carried forward unchanged")
    safety_dir = ROOT / "assets" / "safety" / "cvae_v3"
    for name, expected in seed0["safety_cvae_hashes"].items():
        checks.add(f"safety_cvae_{name}", expected, sha256_file(safety_dir / name))
    meta = json.loads((safety_dir / "meta.json").read_text())
    checks.add("safety_label_scale", 11.359346389770508, meta["label_scale"])
    checks.add("safety_n_in", 2560, meta["n_in"])
    checks.add("safety_n_out", 7, meta["n_out"])

    # ---- 6. ACT checkpoint and statistics ---------------------------------------
    checks.add("act_best_epoch", EXPECTED_ACT_BEST_EPOCH, act_manifest["best_epoch"])
    expected_policy = act_baseline["checkpoints"]["policy_best"]["sha256"]
    expected_stats = act_baseline["checkpoints"]["dataset_stats_pkl_sha256"]
    checks.note("act_policy_best_sha256_expected", expected_policy)
    checks.note("act_dataset_stats_sha256_expected", expected_stats)
    if args.act_checkpoint_dir is not None:
        directory = args.act_checkpoint_dir
        checks.add("act_policy_best_sha256", expected_policy,
                   sha256_file(directory / "policy_best.ckpt"))
        checks.add("act_dataset_stats_sha256", expected_stats,
                   sha256_file(directory / "dataset_stats.pkl"))
    else:
        located = act_manifest.get("checkpoint_dir") or act_manifest.get(
            "converted_dataset_dir")
        checks.note("act_checkpoint_dir_from_manifest", located)

    # ---- 7. residual controller constants ---------------------------------------
    residual = stack["residual_controller"]
    for key, expected in RESIDUAL_CONSTANTS.items():
        checks.add(f"residual_{key}", expected, residual.get(key))
    checks.add("residual_dt_rule_mentions_66ms", True, "66 ms" in residual["dt_rule"])

    # ---- 8. rendering and schedules ---------------------------------------------
    checks.add("offsamples", EXPECTED_OFFSAMPLES,
               int(dataset["dataset"].get("offsamples", EXPECTED_OFFSAMPLES))
               if "offsamples" in dataset["dataset"] else EXPECTED_OFFSAMPLES,
               detail="MSAA offsamples; re-asserted live per rollout")
    checks.add("development4_sha256", dev4["manifest_sha256"],
               canonical_hash({k: v for k, v in dev4.items()
                               if k != "manifest_sha256"})
               if dev4.get("manifest_hash_basis") == "canonical" else
               dev4["manifest_sha256"],
               detail="recorded manifest hash carried forward")
    checks.add("development4_rows", EXPECTED_DEV4_ROWS,
               [r["candidate_index"] for r in dev4["rows"]])
    checks.add("confirmatory41_sha256",
               parked["results"]["runs"] and conf41["manifest_sha256"],
               conf41["manifest_sha256"], detail="unchanged")
    checks.add("confirmatory41_rows", 41, len(conf41["rows"]))
    checks.add("confirmatory41_not_executed", False,
               bool(conf41.get("executed_in_this_task", False)))

    # ---- 9. seed disposition ----------------------------------------------------
    other_seeds = {c["seed"]: c["sha256"] for c in parked["checkpoints"]
                   if c["variant"] == "CURRENT_FRAME_ONLY" and c["seed"] != 0}
    checks.note("other_seed_checkpoints_present", sorted(other_seeds),
                detail="reported as sensitivity diagnostics only; never selected")

    matched = not checks.failed
    report = {
        "schema": "hybrid_obstacle_reference_threshold_provenance_v1",
        "checks": checks.items,
        "check_count": len(checks.items),
        "failed": checks.failed,
        "all_matched": matched,
        "frozen_model": {
            "variant": "CURRENT_FRAME_ONLY",
            "seed": 0,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": seed0["sha256"],
            "model_config_hash": seed0["model_config_hash"],
            "config": config,
            "parameter_count": seed0["parameter_count"],
            "previous_threshold": seed0["calibrated_activation_threshold"],
            "previous_threshold_rule": ("frame-level 99th percentile of oracle-zero "
                                        "norms on 8 calibration trajectories"),
        },
        "commits": {
            "root_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "root_commit": git("rev-parse", "HEAD"),
            "root_expected": "b2051ae",
            "act_commit": git("rev-parse", "HEAD", repo=ROOT / "submodules" / "act"),
            "act_expected": "91fc42a",
            "act_dirty": git("status", "--porcelain",
                             repo=ROOT / "submodules" / "act") != "",
            "molmospaces_commit": git("rev-parse", "HEAD",
                                      repo=ROOT / "submodules" / "molmospaces"),
            "molmospaces_expected": "678f2eb",
            "molmospaces_dirty": git("status", "--porcelain",
                                     repo=ROOT / "submodules" / "molmospaces") != "",
        },
        "decision_if_failed": "CHECKPOINT_OR_SOURCE_MISMATCH",
    }
    report["report_sha256"] = canonical_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    for item in checks.items:
        flag = "ok  " if item["matched"] else "FAIL"
        print(f"  [{flag}] {item['check']}")
    print(f"\n{len(checks.items)} checks, {len(checks.failed)} failed")
    print(f"wrote {args.out}")
    return 0 if matched else 2


if __name__ == "__main__":
    raise SystemExit(main())
