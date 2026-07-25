#!/usr/bin/env python3
"""Final offline validation for the hybrid-obstacle full collection.

Handoff step 15. Aggregates the checks that are not already produced by the
integrity audit, the canonical builder or the conversion wrapper, and rolls the
whole set into one machine-readable verdict:

* conversion A/B reproducibility, including an explicit classification of any
  nonsemantic difference (absolute output paths, HDF5 container bytes)
* canonical selection regeneration (byte-for-byte manifest hash)
* split leakage
* quota
* JSON / YAML parsing of every artifact written by this task
* Python byte compilation of the new audit tooling
* Ruff on the new root audit tooling
* ``git diff --check`` on the root repo
* clean-submodule verification
* a final process guard

No simulation is launched. Everything here is read-only apart from the report
and the regeneration scratch directory.
"""
from __future__ import annotations

import argparse
import compileall
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

ROOT = Path("/root/prox_learning_hybrid_safety")


def run(cmd: list[str], cwd: Path = ROOT) -> tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    return p.returncode, p.stdout, p.stderr


def compare_conversions(a_dir: Path, b_dir: Path, a_man: Path, b_man: Path) -> dict:
    ma = json.loads(a_man.read_text())
    mb = json.loads(b_man.read_text())

    a_files = sorted(p.name for p in a_dir.glob("episode_*.hdf5"))
    b_files = sorted(p.name for p in b_dir.glob("episode_*.hdf5"))

    a_by = {e["act_file"]: e for e in ma["episodes"]}
    b_by = {e["act_file"]: e for e in mb["episodes"]}

    file_hash_diffs = []
    semantic_diffs = []
    id_diffs = []
    for name in sorted(set(a_by) | set(b_by)):
        ea, eb = a_by.get(name), b_by.get(name)
        if ea is None or eb is None:
            id_diffs.append({"act_file": name, "in_A": ea is not None, "in_B": eb is not None})
            continue
        if ea["episode_id"] != eb["episode_id"]:
            id_diffs.append({"act_file": name, "A": ea["episode_id"], "B": eb["episode_id"]})
        if ea["act_file_sha256"] != eb["act_file_sha256"]:
            file_hash_diffs.append(
                {"act_file": name, "A": ea["act_file_sha256"], "B": eb["act_file_sha256"]}
            )
        if ea["act_semantic_sha256"] != eb["act_semantic_sha256"]:
            semantic_diffs.append(
                {"act_file": name, "A": ea["act_semantic_sha256"], "B": eb["act_semantic_sha256"]}
            )

    # Manifest comparison with the documented nonsemantic fields removed.
    NONSEMANTIC = {"wrapper_sha256"}
    def strip(m: dict) -> dict:
        d = {k: v for k, v in m.items() if k not in NONSEMANTIC and k != "episodes"}
        d["episodes"] = [
            {k: v for k, v in e.items() if k not in NONSEMANTIC} for e in m["episodes"]
        ]
        return d

    manifests_equal_semantic = strip(ma) == strip(mb)
    manifest_field_diffs = [
        k for k in set(ma) | set(mb) if k != "episodes" and ma.get(k) != mb.get(k)
    ]

    # If container bytes differ but content does not, prove it dataset-by-dataset
    # on a sample so the difference is identified rather than ignored.
    container_only_evidence = None
    if file_hash_diffs and not semantic_diffs:
        sample = file_hash_diffs[0]["act_file"]
        with h5py.File(a_dir / sample, "r") as fa, h5py.File(b_dir / sample, "r") as fb:
            names: list[str] = []
            fa.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
            per = {}
            for n in sorted(names):
                xa, xb = np.asarray(fa[n][()]), np.asarray(fb[n][()])
                per[n] = {
                    "shape_equal": xa.shape == xb.shape,
                    "dtype_equal": str(xa.dtype) == str(xb.dtype),
                    "bytes_equal": bool(xa.shape == xb.shape and np.array_equal(xa, xb)),
                }
            container_only_evidence = {
                "sample_file": sample,
                "all_datasets_bytes_equal": all(v["bytes_equal"] for v in per.values()),
                "per_dataset": per,
                "size_A": (a_dir / sample).stat().st_size,
                "size_B": (b_dir / sample).stat().st_size,
            }

    return {
        "file_lists_equal": a_files == b_files,
        "episode_count_A": len(a_by),
        "episode_count_B": len(b_by),
        "episode_id_differences": id_diffs,
        "file_hash_differences": file_hash_diffs,
        "semantic_hash_differences": semantic_diffs,
        "tree_file_sha256_A": ma["converted_tree_file_sha256"],
        "tree_file_sha256_B": mb["converted_tree_file_sha256"],
        "tree_file_sha256_equal": ma["converted_tree_file_sha256"] == mb["converted_tree_file_sha256"],
        "tree_semantic_sha256_A": ma["converted_tree_semantic_sha256"],
        "tree_semantic_sha256_B": mb["converted_tree_semantic_sha256"],
        "tree_semantic_sha256_equal": (
            ma["converted_tree_semantic_sha256"] == mb["converted_tree_semantic_sha256"]
        ),
        "manifest_field_differences": sorted(manifest_field_diffs),
        "manifests_equal_after_documented_nonsemantic_fields": manifests_equal_semantic,
        "container_only_evidence": container_only_evidence,
        "identical": bool(
            a_files == b_files
            and not id_diffs
            and not semantic_diffs
            and ma["converted_tree_semantic_sha256"] == mb["converted_tree_semantic_sha256"]
        ),
        "bit_identical_containers": bool(
            not file_hash_diffs
            and ma["converted_tree_file_sha256"] == mb["converted_tree_file_sha256"]
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--canonical", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--smoke-summary", required=True, type=Path)
    ap.add_argument("--conv-a", required=True, type=Path)
    ap.add_argument("--conv-b", required=True, type=Path)
    ap.add_argument("--conv-a-manifest", required=True, type=Path)
    ap.add_argument("--conv-b-manifest", required=True, type=Path)
    ap.add_argument("--source-manifest", required=True, type=Path)
    ap.add_argument("--regen-dir", required=True, type=Path)
    ap.add_argument("--builder", required=True, type=Path)
    ap.add_argument("--tooling", nargs="*", default=[], type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    checks: dict[str, Any] = {}

    audit = json.loads(args.audit.read_text())
    canonical = json.loads(args.canonical.read_text())
    split = json.loads(args.split.read_text())
    smoke = json.loads(args.smoke_summary.read_text())
    source_manifest = json.loads(args.source_manifest.read_text())

    # 1. source integrity + manifest identity + duplicates + sensors + hazard
    rec = audit["reconciliation"]
    checks["source_integrity_audit"] = {
        "ok": audit["ok"],
        "rows_accounted": rec["rows_with_outcome"],
        "manifest_rows": rec["manifest_rows"],
        "successes_clean": audit["successes_clean"],
        "successes_audited": audit["successes_audited"],
    }
    checks["manifest_identity_audit"] = {
        "ok": bool(
            rec["every_candidate_index_once"]
            and not rec["duplicate_candidate_indices"]
            and not rec["duplicate_episode_ids"]
            and not rec["duplicate_row_hashes"]
            and not rec["row_hash_mismatch"]
        ),
        "duplicate_candidate_indices": rec["duplicate_candidate_indices"],
        "duplicate_episode_ids": rec["duplicate_episode_ids"],
        "duplicate_row_hashes": rec["duplicate_row_hashes"],
        "row_hash_mismatch": rec["row_hash_mismatch"],
    }
    d = audit["duplicates"]
    checks["duplicate_content_audit"] = {
        "ok": not (d["all_leaf_sha256"] or d["core_trajectory_sha256"] or d["task_state_sha256"] or d["episode_spec_sha256"]),
        "core_trajectory_collisions": len(d["core_trajectory_sha256"]),
        "task_state_collisions": len(d["task_state_sha256"]),
        "all_leaf_collisions": len(d["all_leaf_sha256"]),
        "episode_spec_collisions": len(d["episode_spec_sha256"]),
        "replica_class_size_histogram": d["replica_class_size_histogram"],
    }
    bad_sensor = [
        r["episode_id"] for r in audit["per_row"]
        if r.get("proximity_count") != 40 or "bad_proximity_shapes" in r
    ]
    checks["sensor_schema_audit"] = {
        "ok": audit["sensor_order_ok"] and not bad_sensor,
        "sensor_order_ok": audit["sensor_order_ok"],
        "rows_with_bad_sensors": bad_sensor,
    }
    hz = [
        r["episode_id"] for r in audit["per_row"]
        if any("hazard" in p for p in r.get("problems", []))
    ]
    checks["hazard_label_geometry_audit"] = {
        "ok": not hz and not rec["hazard_label_mismatch"],
        "rows_with_hazard_problems": hz,
        "ledger_hazard_mismatch": rec["hazard_label_mismatch"],
    }
    wv = rec["worker_verdict"]
    checks["worker_completion_reconciliation"] = {
        "ok": bool(
            wv["ok"]
            and not rec["unresolved_claims"]
            and rec["collection_summary_present"]
        ),
        "unresolved_claims": rec["unresolved_claims"],
        "worker_histogram": rec["worker_histogram"],
        "summary_present": rec["collection_summary_present"],
        "summary_complete": wv["summary_complete"],
        "workers_complete": wv["workers_complete"],
        "row_reconciliation_ok": wv["row_reconciliation_ok"],
        "silently_lost_workers": wv["silently_lost_workers"],
        "missing_final_status": wv["missing_final_status"],
        "worker_exit_codes": wv["worker_exit_codes"],
        "every_worker_has_approved_final_record": wv["every_worker_has_approved_final_record"],
        "parent_worker_totals_reconcile": wv["parent_worker_totals_reconcile"],
        "per_worker_sums": wv["per_worker_sums"],
        "shared_counters": wv["shared_counters"],
        "stale_house_based_warning_present": wv["stale_house_based_warning_present"],
        "stale_house_based_warning_note": (
            "build_final_summary derives `complete` from a house-based comparison and a "
            "manifest run writes no houses, so it always inserts this warning; the "
            "manifest runner then overrides complete/status from row reconciliation but "
            "does not delete the key. The validated smoke reference carries it too. Not a "
            "worker-loss signal."
        ),
    }

    # 2. smoke reference
    checks["smoke_reference_comparison"] = {
        "ok": smoke["ok"],
        "episodes_compared": smoke["episodes_compared"],
        "all_invariant": smoke["all_invariant"],
        "all_bit_identical": smoke["all_bit_identical"],
        "reference_h5s_retained": smoke["reference_h5s_retained"],
    }

    # 3. quota
    ds = audit["distinct_successes"]
    checks["quota_check"] = {
        "ok": ds["hazard_present"] >= 75 and ds["hazard_absent"] >= 25,
        "distinct_successes": ds,
        "required": {"hazard_present": 75, "hazard_absent": 25},
    }

    # 4. canonical selection regeneration
    args.regen_dir.mkdir(parents=True, exist_ok=True)
    rc, _stdout, err = run([
        sys.executable, str(args.builder),
        "--run", str(args.run),
        "--manifest", str(ROOT / "configs" / "hybrid_obstacle_candidate_manifest_v2.json"),
        "--audit", str(args.audit),
        "--source-manifest", str(args.source_manifest),
        "--out-canonical", str(args.regen_dir / "canonical.json"),
        "--out-split", str(args.regen_dir / "split.json"),
    ])
    regen_canonical = json.loads((args.regen_dir / "canonical.json").read_text())
    regen_split = json.loads((args.regen_dir / "split.json").read_text())
    checks["canonical_selection_regeneration"] = {
        "ok": bool(
            rc == 0
            and regen_canonical["manifest_sha256"] == canonical["manifest_sha256"]
            and regen_split["split_manifest_sha256"] == split["split_manifest_sha256"]
        ),
        "exit_code": rc,
        "canonical_sha256_original": canonical["manifest_sha256"],
        "canonical_sha256_regenerated": regen_canonical["manifest_sha256"],
        "split_sha256_original": split["split_manifest_sha256"],
        "split_sha256_regenerated": regen_split["split_manifest_sha256"],
        "stderr_tail": err.strip().splitlines()[-3:] if err.strip() else [],
    }

    # 5. split leakage
    checks["split_leakage_audit"] = {
        "ok": bool(
            split["leakage_free"]
            and split["counts"]["train"]["total"] == 80
            and split["counts"]["validation"]["total"] == 20
            and split["counts"]["train"]["hazard_present"] == 60
            and split["counts"]["train"]["hazard_absent"] == 20
            and split["counts"]["validation"]["hazard_present"] == 15
            and split["counts"]["validation"]["hazard_absent"] == 5
        ),
        "counts": split["counts"],
        "leakage_audit": split["leakage_audit"],
    }

    # 6. conversion A/B
    conv = compare_conversions(args.conv_a, args.conv_b, args.conv_a_manifest, args.conv_b_manifest)
    ma = json.loads(args.conv_a_manifest.read_text())
    conv["composition_ok"] = bool(
        ma["episode_count"] == 100
        and ma["hazard_present"] == 75
        and ma["hazard_absent"] == 25
        and ma["train_count"] == 80
        and ma["validation_count"] == 20
    )
    conv["ok"] = bool(conv["identical"] and conv["composition_ok"])
    checks["double_conversion_reproducibility"] = conv

    # 7. JSON / YAML parsing
    json_files = [
        args.audit, args.canonical, args.split, args.smoke_summary,
        args.conv_a_manifest, args.conv_b_manifest, args.source_manifest,
        ROOT / "configs" / "hybrid_obstacle_candidate_manifest_v2.json",
        ROOT / "configs" / "hybrid_obstacle_manifest_v2_smoke8.json",
        ROOT / "configs" / "hybrid_safety_stack_v1.json",
    ]
    json_bad = []
    for p in json_files:
        try:
            json.loads(Path(p).read_text())
        except Exception as exc:  # noqa: BLE001
            json_bad.append({"path": str(p), "error": str(exc)})
    yaml_bad = []
    for p in [ROOT / "configs" / "hybrid_obstacle_independent_v2.yaml"]:
        try:
            yaml.safe_load(Path(p).read_text())
        except Exception as exc:  # noqa: BLE001
            yaml_bad.append({"path": str(p), "error": str(exc)})
    checks["json_yaml_parsing"] = {
        "ok": not json_bad and not yaml_bad,
        "json_checked": len(json_files),
        "json_errors": json_bad,
        "yaml_errors": yaml_bad,
    }

    # 8. byte compilation
    tooling = [Path(t) for t in args.tooling]
    compile_ok = True
    for t in tooling:
        if not compileall.compile_file(str(t), quiet=2, force=True):
            compile_ok = False
    checks["python_byte_compilation"] = {
        "ok": compile_ok,
        "files": [str(t) for t in tooling],
    }

    # 9. ruff
    if tooling:
        rc_r, out_r, err_r = run([sys.executable, "-m", "ruff", "check", *[str(t) for t in tooling]])
        checks["ruff"] = {
            "ok": rc_r == 0,
            "exit_code": rc_r,
            "stdout": out_r.strip()[-3000:],
            "stderr": err_r.strip()[-1500:],
        }
    else:
        checks["ruff"] = {"ok": True, "skipped": "no tooling passed"}

    # 10. git diff --check
    rc_g, out_g, _ = run(["git", "diff", "--check"])
    checks["git_diff_check"] = {"ok": rc_g == 0, "exit_code": rc_g, "stdout": out_g.strip()[-2000:]}

    # 11. clean submodules
    sub = {}
    for name in ("molmospaces", "act"):
        p = ROOT / "submodules" / name
        _, st, _ = run(["git", "status", "--porcelain"], cwd=p)
        _, head, _ = run(["git", "rev-parse", "HEAD"], cwd=p)
        sub[name] = {"status_porcelain": st.strip(), "head": head.strip(), "clean": not st.strip()}
    _, gitlinks, _ = run(["git", "submodule", "status"])
    checks["clean_submodule_verification"] = {
        "ok": all(v["clean"] for v in sub.values())
        and sub["molmospaces"]["head"] == "678f2eb4a0ac0d9e3d14e555aaac0e099089b9a5"
        and sub["act"]["head"] == "3d25c69edd8d972afa59fec5c3edb9d13a357f92",
        "submodules": sub,
        "gitlinks": gitlinks.strip(),
    }

    # 12. final process guard
    _, ps, _ = run(["bash", "-lc",
                    "ps -eo pid,etimes,cmd | grep -Ei 'run_hybrid_obstacle_manifest|manifest_runner|pytest|train_safety_cvae|imitate_episodes' | grep -v grep || true"])
    live = [line for line in ps.strip().splitlines() if line.strip()]
    checks["final_process_guard"] = {"ok": not live, "live_processes": live}

    # 13. source unchanged during the audit
    changed = []
    for entry in source_manifest["files"]:
        p = args.run / entry["path"]
        if not p.is_file():
            changed.append({"path": entry["path"], "reason": "missing"})
            continue
        if p.stat().st_size != entry["bytes"]:
            changed.append({"path": entry["path"], "reason": "size changed"})
    checks["source_unchanged_during_audit"] = {
        "ok": not changed,
        "files_checked": len(source_manifest["files"]),
        "changed": changed,
    }

    all_ok = all(v.get("ok") for v in checks.values())
    report = {
        "schema": "hybrid_obstacle_full_collection_final_offline_validation",
        "run_dir": str(args.run),
        "all_ok": all_ok,
        "failed_checks": [k for k, v in checks.items() if not v.get("ok")],
        "checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    width = max(len(k) for k in checks)
    for k, v in checks.items():
        print(f"  {'PASS' if v.get('ok') else 'FAIL'}  {k:<{width}}")
    print(f"\nALL CHECKS {'PASS' if all_ok else 'FAILED'}")
    print(f"wrote {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
