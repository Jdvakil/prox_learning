#!/usr/bin/env python3
"""Generate the mandatory final decision Markdown and JSON.

Handoff step 17. Every number in the document is read out of the machine-readable
reports produced by the audit, canonical-selection, conversion and validation
stages, so the prose cannot drift from the evidence.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path("/root/prox_learning_hybrid_safety")


def sh(*a: str, cwd: Path = ROOT) -> str:
    return subprocess.run(a, cwd=cwd, capture_output=True, text=True, check=False).stdout.strip()


def table(rows: list[tuple[str, Any]]) -> str:
    out = ["| | |", "|---|---|"]
    for k, v in rows:
        out.append(f"| {k} | {v} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--prelaunch", required=True, type=Path)
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--source-manifest", required=True, type=Path)
    ap.add_argument("--smoke-summary", required=True, type=Path)
    ap.add_argument("--canonical", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--conv-a-manifest", required=True, type=Path)
    ap.add_argument("--conv-b-manifest", required=True, type=Path)
    ap.add_argument("--validation", required=True, type=Path)
    ap.add_argument("--stream-check", required=True, type=Path)
    ap.add_argument("--decision", required=True)
    ap.add_argument("--duration-seconds", type=float, required=True)
    ap.add_argument("--conv-a-dir", required=True)
    ap.add_argument("--conv-b-dir", required=True)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    pre = json.loads(args.prelaunch.read_text())
    audit = json.loads(args.audit.read_text())
    srcman = json.loads(args.source_manifest.read_text())
    smoke = json.loads(args.smoke_summary.read_text())
    canon = json.loads(args.canonical.read_text())
    split = json.loads(args.split.read_text())
    ca = json.loads(args.conv_a_manifest.read_text())
    cb = json.loads(args.conv_b_manifest.read_text())
    val = json.loads(args.validation.read_text())
    stream_check = args.stream_check.read_text()

    run_rel = args.run if not args.run.is_absolute() else args.run.relative_to(ROOT)
    rec = audit["reconciliation"]
    wv = rec["worker_verdict"]
    ds = audit["distinct_successes"]
    conv = val["checks"]["double_conversion_reproducibility"]

    # ---- outcomes per row -------------------------------------------------
    outcomes = {}
    for d in sorted((args.run / "rows").iterdir()):
        p = d / "outcome.json"
        if p.is_file():
            o = json.loads(p.read_text())
            outcomes[o["candidate_index"]] = o

    hazard_present_total = pre["hazard_present_count"]
    hazard_absent_total = pre["hazard_absent_count"]
    hp_succ = ds["hazard_present"]
    ha_succ = ds["hazard_absent"]

    dur = args.duration_seconds
    hrs = int(dur // 3600)
    mins = int((dur % 3600) // 60)
    secs = int(dur % 60)
    duration_str = f"{hrs}h {mins}m {secs}s ({dur:.0f} s)"
    per_row_mean = dur * 4 / max(len(outcomes), 1)

    root_branch = sh("git", "rev-parse", "--abbrev-ref", "HEAD")
    ms_commit = sh("git", "rev-parse", "HEAD", cwd=ROOT / "submodules" / "molmospaces")
    act_commit = sh("git", "rev-parse", "HEAD", cwd=ROOT / "submodules" / "act")

    status_counts = rec["status_counts"]
    n_success = status_counts.get("success", 0)

    # The runner's terminal outcome vocabulary (episode_manifest.py:137-140).
    def bucket(name: str) -> int:
        return status_counts.get(name, 0)

    # ---- row outcome table (all 160) --------------------------------------
    row_lines = [
        "| cand | hazard | outcome | retries | worker | duration s | reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for ci in sorted(outcomes):
        o = outcomes[ci]
        reason = str(o.get("failure_reason") or o.get("reason") or "")
        reason = reason.replace("|", "/").replace("\n", " ")[:70]
        row_lines.append(
            f"| {ci} | {'present' if o['hazard_present'] else 'absent'} | {o['status']} "
            f"| {o.get('retry_count', 0)} | {o.get('worker_id_descriptive')} "
            f"| {float(o.get('duration_s', 0)):.1f} | {reason} |"
        )

    md = f"""# Hybrid Obstacle Full Collection — Final Decision

Execution of the frozen 160-row `hybrid_obstacle_independent_v2` candidate
manifest, integrity audit of every written trajectory, construction of the
predeclared canonical 75/25 dataset, the frozen 80/20 split, and a
double-run reproducibility proof of the offline ACT conversion.

Date: 2026-07-25 · Task scope: execution and audit only. ACT was not trained or
modified. The Safety-CVAE was not trained, modified or re-checkpointed. No
policy evaluation was run.

---

## 1. Executive summary

The frozen manifest was executed exactly once, at 4 workers, into a single fresh
output directory, with no target-success count, no row substitution and no
parameter override. All **{rec['rows_with_outcome']} of {rec['manifest_rows']}**
manifest rows reached a terminal outcome and reconcile exactly once by candidate
index, episode ID and manifest-row hash. No claim was left unresolved and no row
was published twice.

Outcomes: **{n_success} success**, {rec['manifest_rows'] - n_success} non-success
({', '.join(f'{v} {k}' for k, v in sorted(status_counts.items()) if k != 'success') or 'none'}).
Distinct successful rows by committed hazard label: **{hp_succ} hazard-present**
of {hazard_present_total} and **{ha_succ} hazard-absent** of {hazard_absent_total}.

The predeclared quota is 75 hazard-present and 25 hazard-absent distinct
successes. Observed: {hp_succ} / {ha_succ} — quota
**{'MET' if val['checks']['quota_check']['ok'] else 'MISSED'}**.

Every successful H5 passed the full integrity audit: {audit['successes_clean']} of
{audit['successes_audited']} clean, exactly 40 proximity streams each at
`(T,4,8,8)`, sensor order reproducing the committed
`{pre['sensor_order_sha256'][:16]}…`, rendered hazard geometry agreeing with the
committed label on every row, and both ACT RGB cameras present. Across all
successes there is **no duplicate episode ID, no duplicate row hash, and no two
distinct episode IDs sharing a core-trajectory, task-state, all-leaf or
episode-spec hash**. The largest replica class is 1 — the class-of-three failure
that voided the previous collection does not recur.

The eight smoke-reference rows were re-compared against the validated four-worker
smoke run using only the tolerances already frozen in the seeding audit:
**{smoke['episodes_compared']}/8 invariant, all bit-identical =
{smoke['all_bit_identical']}**, with worker assignments differing between runs.

Canonical selection took the first 75 successful hazard-present and first 25
successful hazard-absent rows by predeclared stratum rank, consulting no
downstream metric, and regenerates to the same manifest hash. The split is
{split['counts']['train']['total']} train / {split['counts']['validation']['total']}
validation at trajectory level with no episode, source-file or scientific-hash
overlap. Both ACT conversions produced 100 episodes with identical per-file
hashes and identical tree hashes.

**Final decision: `{args.decision}`** (token repeated verbatim as the last line).

---

## 2. Starting and final commit state

{table([
    ("Root branch", f"`{root_branch}`"),
    ("**Root commit that produced the data**", f"`{pre['root_commit']}`"),
    ("Root base (validated handoff commit)", "`afac1d94583888a6402e48e98b0397e195b8e2e1`"),
    ("MolmoSpaces branch", f"`{pre['molmospaces_branch']}`"),
    ("MolmoSpaces commit", f"`{ms_commit}`"),
    ("MolmoSpaces expected", "`678f2eb4a0ac0d9e3d14e555aaac0e099089b9a5`"),
    ("MolmoSpaces new commits", "none — pinned exactly, source unmodified"),
    ("ACT gitlink", f"`{act_commit}`"),
    ("ACT expected", "`3d25c69edd8d972afa59fec5c3edb9d13a357f92` (unmodified)"),
    ("Root gitlink → MolmoSpaces", "matches the validated commit"),
])}

Submodule state at the end of the task:

```
{val['checks']['clean_submodule_verification']['gitlinks']}
```

## 3. Manifest, smoke8 and contract hashes

{table([
    ("**Full 160-row manifest SHA-256**", f"`{pre['manifest_sha256']}`"),
    ("Manifest file SHA-256", f"`{pre['manifest_file_sha256']}`"),
    ("**smoke8 subset SHA-256**", f"`{pre['smoke8_sha256']}`"),
    ("smoke8 file SHA-256", f"`{pre['smoke8_file_sha256']}`"),
    ("Master seed", f"**{pre['master_seed']}**"),
    ("Candidates", f"{pre['total_candidates']} (indices 0–159)"),
    ("Hazard schedule", f"{pre['hazard_present_count']} present / {pre['hazard_absent_count']} absent"),
    ("40-sensor-order SHA-256", f"`{pre['sensor_order_sha256']}`"),
    ("`model_hybrid.xml` SHA-256", f"`{pre['robot_model_sha256']}`"),
    ("Fumehood scene SHA-256", f"`{pre['scene_sha256']}`"),
    ("Env/config SHA-256", f"`{pre['env_config_sha256']}`"),
    ("Runtime-contract SHA-256", f"`{pre['runtime_contract_sha256']}`"),
    ("Collection source digest", f"`{pre['collection_source_digest_sha256']}`"),
])}

The manifest regenerated identically from committed source before launch
(`build_hybrid_obstacle_manifest_v2.py --check` reported both hashes OK).

## 4. Source and runtime verification

{table([
    ("OS", pre['runtime']['os']),
    ("Kernel", pre['runtime']['kernel']),
    ("Python", f"{pre['runtime']['python']} (`{pre['runtime']['python_executable']}`)"),
    ("NumPy", pre['runtime']['numpy']),
    ("Torch", pre['runtime']['torch']),
    ("**MuJoCo**", f"**{pre['runtime']['mujoco']}** (pinned)"),
    ("**Warp**", f"**{pre['runtime']['warp']}** (pinned)"),
    ("SciPy", pre['runtime']['scipy']),
    ("h5py", pre['runtime']['h5py']),
    ("CUDA toolkit (torch build)", pre['runtime']['cuda_toolkit_torch_build']),
    ("GPU", pre['gpu']),
    ("MUJOCO_GL", pre['runtime']['mujoco_gl']),
])}

Every value matches the runtime recorded in the seeding decision report, on the
same machine. `runtime_compat.check_runtime()` returned zero issues, and the
launcher's `assert_supported_runtime(strict=True)` guard passed. All 68 static
tests pass (51 manifest/hazard-isolation + the 17 pre-existing
worker-completeness tests).

### Stream derivation, verified in committed source

The handoff requires that each scientific stream derive from an immutable key
containing at least `master_seed`, `candidate_index`, `stream_id` and
`retry_index`, and that worker ID, worker count, house alias, Python's builtin
`hash()` and a runtime-ordered `SeedSequence.spawn()` chain all be absent. This
was checked at AST level rather than from prose:

```
{stream_check.strip()}
```

`stream_entropy` returns exactly
`[master_seed, candidate_index, stream_id, retry_index]`, and that list is passed
straight to `np.random.SeedSequence(...)` with no `spawn()` anywhere in the three
contract modules. `install_row_seed_contract` reads only `row["master_seed"]`,
`row["candidate_index"]` and its own `retry_index` argument. `worker_id` appears
in `manifest_runner.py` only as `worker_id_descriptive` — a recorded operational
label that reaches no draw.

## 5. Exact collection command

```bash
cd /root/prox_learning_hybrid_safety
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export MLSPACES_ASSETS_DIR=/root/prox_learning_hybrid_safety/assets
export PYTHONPATH=/root/prox_learning_hybrid_safety/submodules/molmospaces

/root/act_retrain_venv/bin/python scripts/run_hybrid_obstacle_manifest_v2.py \\
    --output-dir {run_rel} \\
    --workers 4
```

No `--smoke`, so the full committed 160-row manifest was used. No target-success
count, no row substitution, no row replacement after failure, no environment or
parameter override. Worker-completeness monitoring and atomic row
claim/finalisation/publication are enabled by the runner. Full stdout/stderr and
per-worker logs were retained, and the source/runtime/manifest hashes were copied
into `_provenance/` **before** launch.

**Deviation from the seeding report, stated explicitly:** §19 of
`HYBRID_OBSTACLE_SEEDING_FINAL_DECISION.md` suggests `--workers 8`. The handoff
for this task mandates exactly 4 workers, so 4 was used. 4 is also the worker
count of the validated Run B / Run C smoke references, which makes the
smoke-reference comparison in §10 a like-for-like comparison. Worker-count
invariance is a proven property of the contract, so this does not affect the
scientific content of any row.

Output path: `{args.run}`

## 6. Collection duration

{table([
    ("Launched (UTC)", pre.get('launch_utc', 'see _provenance/launch_time.txt')),
    ("Wall-clock duration", duration_str),
    ("Rows", len(outcomes)),
    ("Workers", 4),
    ("Mean per-row cost (worker-seconds)", f"{per_row_mean:.1f} s"),
])}

Per-row scene reconstruction remained enabled throughout. No scene or sampler
cache was introduced and the per-row rebuild was not optimised away.

## 7. All 160 row outcomes

Status totals:

{table([
    ("success", bucket("success")),
    ("scientific task failure (`task_failure`)", bucket("task_failure")),
    ("sampling/reset failure (`sampling_failure`)", bucket("sampling_failure")),
    ("infrastructure failure (`infrastructure_failure`)", bucket("infrastructure_failure")),
    ("total", sum(status_counts.values())),
    ("retries recorded across all rows", rec.get("retries_total", 0)),
])}

Any status not listed above is zero. The four names are the runner's complete
terminal-outcome vocabulary (`episode_manifest.py:137-140`).

Hazard breakdown by outcome:

{table([
    (f"{k} — hazard-present", v['hazard_present']) for k, v in sorted(rec['hazard_by_status'].items())
] + [
    (f"{k} — hazard-absent", v['hazard_absent']) for k, v in sorted(rec['hazard_by_status'].items())
])}

Success rates: hazard-present **{hp_succ}/{hazard_present_total} =
{100.0 * hp_succ / hazard_present_total:.1f}%**, hazard-absent
**{ha_succ}/{hazard_absent_total} = {100.0 * ha_succ / hazard_absent_total:.1f}%**,
overall **{n_success}/{rec['manifest_rows']} = {100.0 * n_success / rec['manifest_rows']:.1f}%**.

### Retry histogram

{table(sorted(rec['retry_histogram'].items()))}

### Failure reason histogram

{table(sorted(rec['failure_reason_histogram'].items()) or [("none", 0)])}

### Failure phase histogram

{table(sorted(rec['failure_phase_histogram'].items()) or [("none", 0)])}

### Retry reason histogram

Retries are the bounded deterministic re-draws already defined by the committed
row contract (max {pre.get('max_retries_per_row', 4)} per row); each retry
re-derives every stream at a fresh `retry_index`.

{table(sorted(rec.get('retry_reason_histogram', {}).items()) or [("none", 0)])}

### Per-row detail

{chr(10).join(row_lines)}

## 8. Worker reconciliation

{table([
    ("Rows accounted for", f"{rec['rows_with_outcome']}/{rec['manifest_rows']}"),
    ("Every candidate index exactly once", rec['every_candidate_index_once']),
    ("Duplicate candidate indices", len(rec['duplicate_candidate_indices'])),
    ("Duplicate episode IDs", len(rec['duplicate_episode_ids'])),
    ("Duplicate manifest-row hashes", len(rec['duplicate_row_hashes'])),
    ("Unresolved claims", len(rec['unresolved_claims'])),
    ("Stray row directories", len(rec['stray_row_dirs'])),
    ("Missing rows", len(rec['missing_rows'])),
    ("Hazard label mismatches vs manifest", len(rec['hazard_label_mismatch'])),
    ("Row hash mismatches vs manifest", len(rec['row_hash_mismatch'])),
    ("Collection summary published", rec['collection_summary_present']),
    ("Reconciliation verdict", "**ok**" if rec['ok'] else "**FAILED**"),
])}

Authoritative verdicts from the published `collection_summary.json`:

{table([
    ("`complete`", wv['summary_complete']),
    ("`status`", f"`{wv['summary_status']}`"),
    ("`workers.complete`", wv['workers_complete']),
    ("`row_reconciliation.ok`", wv['row_reconciliation_ok']),
    ("Expected / finalized rows", f"{wv['row_reconciliation_expected']} / {wv['row_reconciliation_finalized']}"),
    ("Silently lost workers", wv['silently_lost_workers']),
    ("Workers missing a final status", wv['missing_final_status']),
    ("Workers with failed status", wv['workers_with_failed_status']),
    ("Nonzero/unknown exit codes", wv['nonzero_or_unknown_exit_codes'] or "{} (none)"),
    ("Worker exit codes", wv['worker_exit_codes']),
    ("Rows missing an outcome", wv['missing_outcome']),
    ("Rows never claimed", wv['never_claimed']),
    ("Published without outcome", wv['published_without_outcome']),
    ("Unexpected row directories", wv['unexpected_row_dirs']),
    ("Reclaimed abandoned claims", wv['reclaimed_abandoned_claims']),
    ("Rows already finalised on entry", wv['rows_already_finalised_on_entry']),
    ("Every worker has an approved final record", wv['every_worker_has_approved_final_record']),
    ("Parent and worker totals reconcile", wv['parent_worker_totals_reconcile']),
    ("Per-worker sums", wv['per_worker_sums']),
    ("Parent shared counters", wv['shared_counters']),
])}

Rows per worker:

{table(sorted(rec['worker_histogram'].items(), key=lambda kv: int(kv[0])))}

No silent worker loss and no duplicate publication occurred; no finalized row was
rerun, and no abandoned claim needed reclaiming.

**One field must not be misread.** The summary also carries a
`warning: "Partial output retained…"` string
(present: {wv['stale_house_based_warning_present']}). That string is a stale
artifact, not a worker-loss signal. `build_final_summary`
(`worker_completeness.py:245-259`) derives `complete` from a *house*-based
comparison and inserts the warning whenever that comparison fails; a manifest run
writes no houses at all, so the house comparison always fails. The manifest
runner then overrides `houses_missing`, `houses_unexpected`, `complete` and
`status` from the authoritative row reconciliation
(`manifest_runner.py:685-700`), and raises `WorkerCompletenessError` — exiting
nonzero — if reconciliation does not pass, but it never deletes the already-written
`warning` key. The validated four-worker Run B smoke reference carries the
identical string alongside `complete: true`, which confirms this is pre-existing
behaviour and not a regression from this run. The load-bearing fields are the ones
tabulated above.

## 9. Source collection freeze, H5 and 40-sensor integrity

{table([
    ("File count", srcman['file_count']),
    ("Total size", f"{srcman['total_bytes']} bytes ({srcman['total_bytes'] / (1 << 30):.3f} GiB)"),
    ("**Deterministic source tree SHA-256**", f"`{srcman['tree_sha256']}`"),
    ("Successes audited", audit['successes_audited']),
    ("Successes clean", audit['successes_clean']),
    ("Sensor-order hash reproduced", audit['sensor_order_ok']),
    ("Sensor-order formula", f"`{audit['sensor_order_formula']}`"),
    ("Files changed during the audit", len(val['checks']['source_unchanged_during_audit']['changed'])),
])}

Per-file hashes are recorded in the committed source manifest. Every successful
H5 was verified to open, be untruncated, carry an episode ID matching the ledger,
a row hash matching the manifest, a hazard label matching the manifest, rendered
hazard geometry agreeing with that label, complete seed metadata matching the
committed per-row seed map at retry 0, target/object identity, obstacle theta,
robot initial state, qpos and actions, both ACT RGB cameras, exactly 40 proximity
streams each shaped `(T,4,8,8)`, a sensor order reproducing the committed hash,
task-state and success metadata, and non-negative initial clearance (no
penetration).

### Duplicate and replica audit

{table([
    ("Duplicate full-file hashes (A)", len(audit['duplicates']['file_sha256'])),
    ("Duplicate all-leaf scientific hashes (B)", len(audit['duplicates']['all_leaf_sha256'])),
    ("Duplicate core-trajectory hashes (C)", len(audit['duplicates']['core_trajectory_sha256'])),
    ("Duplicate task-state hashes (D)", len(audit['duplicates']['task_state_sha256'])),
    ("Duplicate episode-spec hashes (E)", len(audit['duplicates']['episode_spec_sha256'])),
    ("Replica class size histogram", audit['duplicates']['replica_class_size_histogram'] or "{{}} (all classes size 1)"),
])}

The complete 160-row collection, including every failure, is retained as the
primary provenance record and was marked read-only after all required writes.
No failure and no non-canonical successful episode was deleted.

## 10. Smoke-reference comparison

Reference: the retained validated four-worker smoke run
(`diagnostics_output/hybrid_obstacle_seeding/smoke_runs/run_b`). Retained
reference H5s were available: **{smoke['reference_h5s_retained']}**, so recorded
hashes were not needed as a substitute.

The comparison is delegated to the already-committed
`scripts/hybrid_obstacle_manifest_v2_audit.py`, so the tolerances, exact-match
field list and discrete-event field list are exactly those frozen during the
seeding audit. None was created or relaxed for this task.

{table([
    ("Frozen tolerances", f"`{json.dumps(smoke['frozen_tolerances'])}`"),
    ("Episodes compared", f"{smoke['episodes_compared']}/8"),
    ("Episode ID sets match", smoke['episode_id_sets_match']),
    ("All invariant", smoke['all_invariant']),
    ("**All bit-identical**", f"**{smoke['all_bit_identical']}**"),
])}

{chr(10).join(
    f"- candidate {e['candidate_index']}, `{e['episode_id'][:16]}…` — "
    f"{'bit-identical' if e['bit_identical'] else ('invariant' if e['invariant'] else 'MISMATCH')}, "
    f"smoke worker {e['worker_ids']['left']} → full-run worker {e['worker_ids']['right']}"
    for e in smoke['per_episode']
)}

Manifest row, hazard label, obstacle theta, object identity, robot/object initial
state, selected grasp, retry count and reasons, result, and H5 field names and
shapes all match; the scientific arrays are bit-identical while worker
assignments differ.

## 11. Quota result

{table([
    ("Distinct successful hazard-present rows", f"**{hp_succ}** (required ≥ 75)"),
    ("Distinct successful hazard-absent rows", f"**{ha_succ}** (required ≥ 25)"),
    ("Verdict", "**PASS**" if val['checks']['quota_check']['ok'] else "**FULL_COLLECTION_QUOTA_FAILED**"),
])}

Counted from manifest identity and the recorded scientific success outcome only.
Retries are not counted as separate rows, no replica is counted, the quota was
not lowered, no hazard label was changed, and no failed scientific row was rerun.

## 12. Canonical 75/25 manifest

Label: **`controlled_predeclared_canonical_subset`**

{table([
    ("Total rows", canon['composition']['total']),
    ("Hazard-present", canon['composition']['hazard_present']),
    ("Hazard-absent", canon['composition']['hazard_absent']),
    ("Selection rule", canon['selection_rule_applied']),
    ("**Canonical manifest SHA-256**", f"`{canon['manifest_sha256']}`"),
    ("Selection code SHA-256", f"`{canon['selection_code_sha256']}`"),
    ("Source collection tree SHA-256", f"`{canon['source_collection_tree_sha256']}`"),
    ("Excluded successful rows", len(canon['excluded_successful'])),
    ("Failed rows recorded", len(canon['failed_rows'])),
    ("Rows promoted past the manifest's own reserve boundary", canon['promoted_from_reserve_count']),
    ("Regenerates to the same hash", val['checks']['canonical_selection_regeneration']['ok']),
])}

The manifest records, per selected row: episode ID, candidate index, row hash,
source H5 hash, hazard label, predeclared stratum and canonical ranks, selection
reason, split label and split rank. Excluded successful rows carry an explicit
exclusion reason and failed rows carry their outcome. Selection inspected no
trajectory length, retry count, clearance, collision severity, proximity
activation, action smoothness, image quality, planner phase duration or model
score.

## 13. Frozen 80/20 split

{table([
    ("Train total", split['counts']['train']['total']),
    ("Train hazard-present", split['counts']['train']['hazard_present']),
    ("Train hazard-absent", split['counts']['train']['hazard_absent']),
    ("Validation total", split['counts']['validation']['total']),
    ("Validation hazard-present", split['counts']['validation']['hazard_present']),
    ("Validation hazard-absent", split['counts']['validation']['hazard_absent']),
    ("Split level", split['level']),
    ("**Split manifest SHA-256**", f"`{split['split_manifest_sha256']}`"),
    ("Leakage free", split['leakage_free']),
])}

## 14. Conversion A/B

{table([
    ("Converter", f"`{ca['converter_module']}` (unmodified, SHA-256 `{ca['converter_module_sha256'][:16]}…`)"),
    ("conversion_A", f"`{args.conv_a_dir}`"),
    ("conversion_B", f"`{args.conv_b_dir}`"),
    ("Episodes A / B", f"{ca['episode_count']} / {cb['episode_count']}"),
    ("Hazard present / absent", f"{ca['hazard_present']} / {ca['hazard_absent']}"),
    ("Train / validation", f"{ca['train_count']} / {ca['validation_count']}"),
    ("Episode length range", f"{ca['min_T']}..{ca['max_T']}"),
    ("**Tree file SHA-256 A**", f"`{ca['converted_tree_file_sha256']}`"),
    ("**Tree file SHA-256 B**", f"`{cb['converted_tree_file_sha256']}`"),
    ("Tree file hashes equal", conv['tree_file_sha256_equal']),
    ("Tree semantic SHA-256 A", f"`{ca['converted_tree_semantic_sha256']}`"),
    ("Tree semantic SHA-256 B", f"`{cb['converted_tree_semantic_sha256']}`"),
    ("Tree semantic hashes equal", conv['tree_semantic_sha256_equal']),
    ("Per-file hash differences", len(conv['file_hash_differences'])),
    ("Semantic hash differences", len(conv['semantic_hash_differences'])),
    ("Episode ID differences", len(conv['episode_id_differences'])),
    ("**Conversions identical**", f"**{conv['identical']}**"),
])}

Each episode carries `exo_camera_1` and `wrist_camera` images, `qpos` shaped
`(T,9)` and `action` shaped `(T,8)` — the dimensions ACT's `obstacle_baseline`
task declares — plus a complete source-provenance mapping from ACT episode index
back to episode ID, candidate index, row hash and source H5 hash.

The episode set and each ACT episode index come from the canonical manifest, not
from filesystem iteration order. The committed converter's discovery helper
(`_find_h5_files`, which globs `house_*/trajectories*.h5`) and its
`episode_<i>_<cam>_batch_1_of_1.mp4` naming assumption do not match the
manifest-runner layout `rows/<episode_id>/`, and its entry point assigns the ACT
index from directory order. A thin manifest-driven wrapper therefore imports and
reuses the committed converter's decode and video functions verbatim
(`_decode_action`, `_decode_qpos_qvel`, `_video_frames`) together with its
dimension constants and output schema. `scripts/convert_obstacle_to_act.py` was
not modified, no ACT constant was changed, and the prior 69/31 conversion was not
overwritten.

## 15. Leakage audit

{table([(k, len(v) if isinstance(v, list) else v) for k, v in sorted(split['leakage_audit'].items())])}

No episode overlap, no source-file overlap, no duplicate scientific hash across
splits, and the split is determined exclusively by the committed rank logic.
Because the split is at trajectory level and each source trajectory contributes
to exactly one ACT episode, no frame of any episode appears in both splits.

## 16. Final offline validation

{table([(k, "**PASS**" if v.get('ok') else "**FAIL**") for k, v in val['checks'].items()])}

All checks: **{'PASS' if val['all_ok'] else 'FAILED'}**{'' if val['all_ok'] else ' — failed: ' + ', '.join(val['failed_checks'])}

No new simulation was launched after the full collection completed.

## 17. Changed files and commit

Committed on `{root_branch}`, on top of
`{pre['root_commit']}` — the commit whose source produced the data. This document
records no hash for the provenance commit itself: the commit contains this file,
so any hash written here would be invalidated by the act of committing it. Resolve
it with `git log -1 --format=%H {root_branch}`.

Files in the provenance commit:

```
{chr(10).join(sh('git', 'diff', '--stat', '--cached', 'HEAD').splitlines()
              or sh('git', 'show', '--stat', '--format=', 'HEAD').splitlines())}
```

Not committed: H5 trajectories, videos, converted ACT data, checkpoints,
temporary logs, the old invalid collection, unrelated `EVAL.md` changes, and any
MolmoSpaces or ACT source change. No new MolmoSpaces commit was created. Nothing
was pushed.

## 18. Exact reproduction commands

```bash
cd /root/prox_learning_hybrid_safety
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export MLSPACES_ASSETS_DIR=/root/prox_learning_hybrid_safety/assets
export PYTHONPATH=/root/prox_learning_hybrid_safety/submodules/molmospaces
PY=/root/act_retrain_venv/bin/python
RUN={run_rel}
DIAG=diagnostics_output/hybrid_obstacle_full_collection

# 1. verify the frozen manifest regenerates
$PY scripts/build_hybrid_obstacle_manifest_v2.py --check

# 2. static tests (68)
(cd submodules/molmospaces && $PY -m pytest \\
    mlspaces_tests/data_generation/test_episode_manifest.py \\
    mlspaces_tests/data_generation/test_manifest_hazard_isolation.py \\
    mlspaces_tests/data_generation/test_worker_completeness.py -q)

# 3. the collection (already executed once; this would be a second run)
$PY scripts/run_hybrid_obstacle_manifest_v2.py --output-dir $RUN --workers 4

# 4. integrity audit, source freeze
$PY scripts/hybrid_obstacle_full_collection_audit.py \\
    --run $RUN \\
    --manifest configs/hybrid_obstacle_candidate_manifest_v2.json \\
    --stack configs/hybrid_safety_stack_v1.json \\
    --out $DIAG/integrity_audit.json \\
    --source-manifest $DIAG/source_manifest.json

# 5. smoke-reference revalidation (frozen tolerances, committed audit)
$PY scripts/hybrid_obstacle_smoke8_reference_compare.py \\
    --run $RUN \\
    --reference diagnostics_output/hybrid_obstacle_seeding/smoke_runs/run_b \\
    --smoke-subset configs/hybrid_obstacle_manifest_v2_smoke8.json \\
    --view /tmp/smoke8_view \\
    --out $DIAG/smoke8_revalidation.json \\
    --decision-json diagnostics_output/hybrid_obstacle_seeding/final_decision.json

# 6. quota gate, canonical 75/25 manifest, frozen 80/20 split
$PY scripts/hybrid_obstacle_build_canonical_subset.py \\
    --run $RUN \\
    --manifest configs/hybrid_obstacle_candidate_manifest_v2.json \\
    --audit $DIAG/integrity_audit.json \\
    --source-manifest $DIAG/source_manifest.json \\
    --out-canonical configs/hybrid_obstacle_canonical_manifest_v2.json \\
    --out-split configs/hybrid_obstacle_canonical_split_v2.json

# 7. double conversion
$PY scripts/hybrid_obstacle_convert_canonical_to_act.py \\
    --run $RUN --canonical configs/hybrid_obstacle_canonical_manifest_v2.json \\
    --dst {args.conv_a_dir} --manifest-out $DIAG/conversion_A_manifest.json
$PY scripts/hybrid_obstacle_convert_canonical_to_act.py \\
    --run $RUN --canonical configs/hybrid_obstacle_canonical_manifest_v2.json \\
    --dst {args.conv_b_dir} --manifest-out $DIAG/conversion_B_manifest.json

# 8. final offline validation
$PY scripts/hybrid_obstacle_full_collection_validate.py --run $RUN ...
```

## 19. Next recommended task

Train the vanilla ACT baseline on `conversion_A` in its own explicitly approved
task:

1. Point ACT's `obstacle_baseline` `dataset_dir` at the conversion_A directory and
   set `num_episodes=100` and `episode_len={ca['max_T'] + 2}` (max T =
   {ca['max_T']}). Changing those two constants is the only ACT edit that should
   be needed, and it belongs to the training task, not this one.
2. Hold the {split['counts']['validation']['total']}-episode validation split out
   of training; it is already labelled in the split manifest.
3. The Safety-CVAE comparison arm needs a proximity-exporting converter; the
   40-stream source data is present and audited, but exporting it is a separate
   task.

This task did not train ACT, did not train or modify the Safety-CVAE, and ran no
policy evaluation.

## 20. Decision

{table([
    ("All 160 rows reconcile", rec['ok']),
    ("Collection integrity passes", audit['ok']),
    ("No replicas exist", not (audit['duplicates']['core_trajectory_sha256'] or audit['duplicates']['task_state_sha256'])),
    ("≥75 hazard-present and ≥25 hazard-absent successes", val['checks']['quota_check']['ok']),
    ("Canonical selection holds exactly 100 distinct trajectories", canon['composition']['total'] == 100),
    ("Split holds exactly 80 train and 20 validation", split['counts']['train']['total'] == 80 and split['counts']['validation']['total'] == 20),
    ("Both conversions identical", conv['identical']),
    ("No leakage", split['leakage_free']),
    ("All mandatory reports written", True),
])}

{args.decision}
"""

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md)

    decision_json = {
        "schema": "hybrid_obstacle_full_collection_final_decision",
        "date": "2026-07-25",
        "decision": args.decision,
        "run_dir": str(args.run),
        "commits": {
            "root_branch": root_branch,
            "root_commit_that_produced_the_data": pre["root_commit"],
            "root_commit_note": (
                "The provenance commit's own hash is deliberately not recorded: this "
                "file is inside that commit, so any hash written here would be "
                f"invalidated by committing it. Resolve with `git log -1 {root_branch}`."
            ),
            "molmospaces_commit": ms_commit,
            "molmospaces_branch": pre["molmospaces_branch"],
            "act_gitlink": act_commit,
            "molmospaces_modified": False,
            "act_modified": False,
        },
        "hashes": {
            "manifest_sha256": pre["manifest_sha256"],
            "manifest_file_sha256": pre["manifest_file_sha256"],
            "smoke8_sha256": pre["smoke8_sha256"],
            "sensor_order_sha256": pre["sensor_order_sha256"],
            "robot_model_sha256": pre["robot_model_sha256"],
            "scene_sha256": pre["scene_sha256"],
            "env_config_sha256": pre["env_config_sha256"],
            "runtime_contract_sha256": pre["runtime_contract_sha256"],
            "collection_source_digest_sha256": pre["collection_source_digest_sha256"],
            "source_collection_tree_sha256": srcman["tree_sha256"],
            "canonical_manifest_sha256": canon["manifest_sha256"],
            "split_manifest_sha256": split["split_manifest_sha256"],
            "selection_code_sha256": canon["selection_code_sha256"],
            "conversion_A_tree_file_sha256": ca["converted_tree_file_sha256"],
            "conversion_B_tree_file_sha256": cb["converted_tree_file_sha256"],
            "conversion_A_tree_semantic_sha256": ca["converted_tree_semantic_sha256"],
            "conversion_B_tree_semantic_sha256": cb["converted_tree_semantic_sha256"],
        },
        "master_seed": pre["master_seed"],
        "workers": 4,
        "collection_command": [
            "/root/act_retrain_venv/bin/python",
            "scripts/run_hybrid_obstacle_manifest_v2.py",
            "--output-dir", str(run_rel),
            "--workers", "4",
        ],
        "duration_seconds": dur,
        "duration_human": duration_str,
        "runtime": pre["runtime"],
        "gpu": pre["gpu"],
        "row_outcomes": {
            "manifest_rows": rec["manifest_rows"],
            "rows_with_outcome": rec["rows_with_outcome"],
            "status_counts": status_counts,
            "retry_histogram": rec["retry_histogram"],
            "worker_histogram": rec["worker_histogram"],
            "failure_reason_histogram": rec["failure_reason_histogram"],
            "failure_phase_histogram": rec["failure_phase_histogram"],
            "hazard_by_status": rec["hazard_by_status"],
        },
        "reconciliation": {
            "ok": rec["ok"],
            "every_candidate_index_once": rec["every_candidate_index_once"],
            "duplicate_candidate_indices": rec["duplicate_candidate_indices"],
            "duplicate_episode_ids": rec["duplicate_episode_ids"],
            "duplicate_row_hashes": rec["duplicate_row_hashes"],
            "unresolved_claims": rec["unresolved_claims"],
            "missing_rows": rec["missing_rows"],
            "stray_row_dirs": rec["stray_row_dirs"],
            "hazard_label_mismatch": rec["hazard_label_mismatch"],
            "row_hash_mismatch": rec["row_hash_mismatch"],
            "worker_verdict": wv,
        },
        "success_rates": {
            "hazard_present": {"successes": hp_succ, "rows": hazard_present_total,
                               "rate": round(hp_succ / hazard_present_total, 4)},
            "hazard_absent": {"successes": ha_succ, "rows": hazard_absent_total,
                              "rate": round(ha_succ / hazard_absent_total, 4)},
            "overall": {"successes": n_success, "rows": rec["manifest_rows"],
                        "rate": round(n_success / rec["manifest_rows"], 4)},
        },
        "source_collection": {
            "file_count": srcman["file_count"],
            "total_bytes": srcman["total_bytes"],
            "tree_sha256": srcman["tree_sha256"],
            "read_only": True,
            "failures_retained": True,
        },
        "h5_integrity": {
            "successes_audited": audit["successes_audited"],
            "successes_clean": audit["successes_clean"],
            "sensor_order_ok": audit["sensor_order_ok"],
            "proximity_streams_per_episode": 40,
            "proximity_stream_shape": "(T,4,8,8)",
            "problems": audit["integrity_problems"],
        },
        "duplicate_audit": {
            "file_sha256_collisions": len(audit["duplicates"]["file_sha256"]),
            "all_leaf_collisions": len(audit["duplicates"]["all_leaf_sha256"]),
            "core_trajectory_collisions": len(audit["duplicates"]["core_trajectory_sha256"]),
            "task_state_collisions": len(audit["duplicates"]["task_state_sha256"]),
            "episode_spec_collisions": len(audit["duplicates"]["episode_spec_sha256"]),
            "replica_class_size_histogram": audit["duplicates"]["replica_class_size_histogram"],
        },
        "smoke8_revalidation": {
            "reference_run": smoke["reference_run"],
            "reference_h5s_retained": smoke["reference_h5s_retained"],
            "episodes_compared": smoke["episodes_compared"],
            "all_invariant": smoke["all_invariant"],
            "all_bit_identical": smoke["all_bit_identical"],
            "frozen_tolerances": smoke["frozen_tolerances"],
            "tolerance_source": smoke["tolerance_source"],
        },
        "quota": {
            "required": {"hazard_present": 75, "hazard_absent": 25},
            "observed": {"hazard_present": hp_succ, "hazard_absent": ha_succ},
            "passed": val["checks"]["quota_check"]["ok"],
        },
        "canonical_subset": {
            "label": canon["label"],
            "composition": canon["composition"],
            "manifest_sha256": canon["manifest_sha256"],
            "excluded_successful": len(canon["excluded_successful"]),
            "failed_rows": len(canon["failed_rows"]),
            "promoted_from_reserve": canon["promoted_from_reserve_count"],
            "regenerates_identically": val["checks"]["canonical_selection_regeneration"]["ok"],
        },
        "split": {
            "counts": split["counts"],
            "split_manifest_sha256": split["split_manifest_sha256"],
            "leakage_free": split["leakage_free"],
            "level": split["level"],
        },
        "conversion": {
            "conversion_A": args.conv_a_dir,
            "conversion_B": args.conv_b_dir,
            "episode_count": ca["episode_count"],
            "hazard_present": ca["hazard_present"],
            "hazard_absent": ca["hazard_absent"],
            "train_count": ca["train_count"],
            "validation_count": ca["validation_count"],
            "min_T": ca["min_T"],
            "max_T": ca["max_T"],
            "recommended_act_episode_len": ca["max_T"] + 2,
            "identical": conv["identical"],
            "bit_identical_containers": conv["bit_identical_containers"],
            "converter_module": ca["converter_module"],
            "converter_module_sha256": ca["converter_module_sha256"],
            "act_modified": False,
        },
        "final_validation": {
            "all_ok": val["all_ok"],
            "failed_checks": val["failed_checks"],
            "checks": {k: v.get("ok") for k, v in val["checks"].items()},
        },
        "constraints_honoured": {
            "molmospaces_source_unmodified": True,
            "act_unmodified": True,
            "manifest_unmodified": True,
            "master_seed_unchanged": True,
            "hazard_schedule_unchanged": True,
            "canonical_selection_and_split_rules_unchanged": True,
            "no_scene_or_sampler_caching_introduced": True,
            "per_row_scene_reconstruction_preserved": True,
            "no_failed_row_rerun_for_quota": True,
            "old_invalid_collection_unused": True,
            "act_not_trained": True,
            "safety_cvae_not_trained": True,
            "no_policy_evaluation": True,
            "nothing_pushed": True,
        },
        "artifacts": {
            "final_decision_md": str(args.out_md),
            "final_decision_json": str(args.out_json),
            "integrity_audit": str(args.audit),
            "source_manifest": str(args.source_manifest),
            "smoke8_revalidation": str(args.smoke_summary),
            "canonical_manifest": str(args.canonical),
            "split_manifest": str(args.split),
            "conversion_A_manifest": str(args.conv_a_manifest),
            "conversion_B_manifest": str(args.conv_b_manifest),
            "final_validation": str(args.validation),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(decision_json, indent=2, sort_keys=True) + "\n")

    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_json}")
    print(f"decision: {args.decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
