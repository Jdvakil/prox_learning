#!/usr/bin/env python3
"""Offline integrity audit of the 160-row hybrid-obstacle full collection.

Covers handoff steps 7, 8 and 10:

* step 7  -- reconcile every manifest row, claim, worker report and total.
* step 8  -- per-success H5 audit plus the five hash families (A-E) and the
             duplicate / replica-class audit.
* step 10 -- source file count, byte size, deterministic tree hash, per-file
             hashes and a source manifest.

Read-only with respect to the collection: nothing under the run directory is
written except the report passed via ``--out``, which is written elsewhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SENSOR_INPUT_SHAPE = (4, 8, 8)
CAM_NAMES = ("exo_camera_1", "wrist_camera")

OUTCOME_SUCCESS = "success"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_payload(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def hash_array(h: hashlib._Hash, arr: np.ndarray) -> None:
    """Feed an array into a hash in a dtype/shape/byte-order-explicit way."""
    h.update(str(arr.dtype.str).encode())
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())


def as_bool(value: Any) -> bool | None:
    """Parse a truth value that may be a real bool, a numpy bool, or a string.

    The runner writes H5 root attributes as strings ("False") and the metadata
    group as lowercase bytes (b"false"), so a bare ``bool()`` would read every
    hazard-absent row as hazard-present.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes"):
            return True
        if v in ("false", "0", "no"):
            return False
    return None


def read_scalar_str(ds) -> str:
    v = ds[()]
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


def leaf_datasets(f: h5py.File, root: str = "") -> list[str]:
    names: list[str] = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            names.append(name)

    (f[root] if root else f).visititems(visit)
    return sorted(names)


def failure_reason_of(outcome: dict) -> str:
    """Human-readable terminal reason for a non-success row.

    The runner records the two failure paths differently. A ``ManifestRowFailure``
    (sampling/reset or infrastructure) carries an explicit ``reason``. A
    ``task_failure`` means the rollout ran to completion but the task was not
    achieved, so there is no exception text -- only whatever earlier retries
    recorded in ``retry_history``.
    """
    explicit = outcome.get("reason") or outcome.get("failure_reason")
    if explicit:
        return str(explicit)
    if outcome.get("status") == "task_failure":
        history = [str(e.get("reason", "")) for e in outcome.get("retry_history") or []]
        if history:
            return (
                "rollout completed without task success; earlier retries: "
                + "; ".join(history)
            )
        return "rollout completed without task success"
    return "unspecified"


def failure_phase_of(outcome: dict) -> str:
    """Coarse phase bucket for a non-success row."""
    status = outcome.get("status")
    if status == "task_failure":
        return "rollout_completed_task_not_achieved"
    reason = str(outcome.get("reason") or "")
    if "exhausted its" in reason:
        return "task_sampling_retries_exhausted"
    if "house_invalid" in reason:
        return "house_invalid_for_task"
    if "no saveable observations" in reason or "H5 was not written" in reason:
        return "output_publication"
    if "shutdown" in reason:
        return "shutdown_requested"
    return status or "unknown"


# --------------------------------------------------------------------------- #
# step 7 -- reconciliation
# --------------------------------------------------------------------------- #
def reconcile(run: Path, manifest: dict) -> dict:
    rows_dir = run / "rows"
    manifest_rows = {r["episode_id"]: r for r in manifest["rows"]}
    by_index = {r["candidate_index"]: r for r in manifest["rows"]}

    outcomes: dict[str, dict] = {}
    claims: dict[str, dict] = {}
    unresolved_claims: list[str] = []
    stray_dirs: list[str] = []

    for d in sorted(rows_dir.iterdir()) if rows_dir.is_dir() else []:
        if not d.is_dir():
            continue
        if d.name not in manifest_rows:
            stray_dirs.append(d.name)
        oc = d / "outcome.json"
        cl = d / "claim.json"
        if cl.is_file():
            claims[d.name] = json.loads(cl.read_text())
        if oc.is_file():
            outcomes[d.name] = json.loads(oc.read_text())
        elif cl.is_file():
            unresolved_claims.append(d.name)

    summary_path = run / "collection_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None

    idx_counter = Counter(o["candidate_index"] for o in outcomes.values())
    eid_counter = Counter(o["episode_id"] for o in outcomes.values())
    row_hash_counter = Counter(o["row_sha256"] for o in outcomes.values())

    status_counts = Counter(o["status"] for o in outcomes.values())
    retry_counts = Counter(int(o.get("retry_count", 0)) for o in outcomes.values())
    worker_counts = Counter(int(o.get("worker_id_descriptive", -1)) for o in outcomes.values())

    hazard_by_status: dict[str, Counter] = defaultdict(Counter)
    for o in outcomes.values():
        hazard_by_status[o["status"]][bool(o["hazard_present"])] += 1

    failure_reasons = Counter()
    failure_phases = Counter()
    retry_reasons = Counter()
    for o in outcomes.values():
        for entry in o.get("retry_history") or []:
            retry_reasons[str(entry.get("reason", "unspecified"))[:180]] += 1
        if o["status"] == OUTCOME_SUCCESS:
            continue
        failure_reasons[failure_reason_of(o)[:180]] += 1
        failure_phases[failure_phase_of(o)] += 1

    # hazard label agreement with the committed manifest
    hazard_mismatch = [
        eid
        for eid, o in outcomes.items()
        if eid in manifest_rows and bool(o["hazard_present"]) != bool(manifest_rows[eid]["hazard_present"])
    ]
    rowhash_mismatch = [
        eid
        for eid, o in outcomes.items()
        if eid in manifest_rows and o["row_sha256"] != manifest_rows[eid]["row_sha256"]
    ]

    missing = sorted(set(manifest_rows) - set(outcomes))

    # ---- authoritative worker/row verdicts from the published summary ------
    #
    # `build_final_summary` derives `complete` from a *house*-based comparison,
    # and a manifest run writes no houses at all, so it always computes
    # complete=False and inserts a "Partial output retained" warning string.
    # `ManifestRolloutRunner` then overrides `houses_missing`,
    # `houses_unexpected`, `complete` and `status` from the authoritative row
    # reconciliation (manifest_runner.py:685-700) but never removes that stale
    # warning key. The validated four-worker smoke reference carries the same
    # stale string, so it is expected and carries no information here. The
    # authoritative fields are the ones read below.
    s = summary or {}
    workers = s.get("workers") or {}
    row_rec = s.get("row_reconciliation") or {}
    counters = s.get("shared_counters") or {}
    per_worker = workers.get("per_worker") or {}
    worker_verdict = {
        "summary_complete": s.get("complete"),
        "summary_status": s.get("status"),
        "stale_house_based_warning_present": "warning" in s,
        "stale_house_based_warning": s.get("warning"),
        "workers_complete": workers.get("complete"),
        "expected_workers": workers.get("expected_workers"),
        "reporting_workers": workers.get("reporting_workers"),
        "missing_final_status": workers.get("missing_final_status"),
        "silently_lost_workers": workers.get("silently_lost_workers"),
        "workers_with_failed_status": workers.get("workers_with_failed_status"),
        "nonzero_or_unknown_exit_codes": workers.get("nonzero_or_unknown_exit_codes"),
        "worker_exit_codes": workers.get("worker_exit_codes"),
        "worker_totals": workers.get("totals"),
        "per_worker": per_worker,
        "row_reconciliation_ok": row_rec.get("ok"),
        "row_reconciliation_expected": row_rec.get("expected_rows"),
        "row_reconciliation_finalized": row_rec.get("finalized_rows"),
        "missing_outcome": row_rec.get("missing_outcome"),
        "never_claimed": row_rec.get("never_claimed"),
        "published_without_outcome": row_rec.get("published_without_outcome"),
        "unexpected_row_dirs": row_rec.get("unexpected_row_dirs"),
        "shared_counters": counters,
        "reclaimed_abandoned_claims": (s.get("manifest") or {}).get("reclaimed_abandoned_claims"),
        "rows_already_finalised_on_entry": (s.get("manifest") or {}).get(
            "rows_already_finalised_on_entry"
        ),
        "manifest_sha256_in_summary": (s.get("manifest") or {}).get("manifest_sha256"),
    }
    # parent totals vs the sum of the per-worker records
    attempted = sum(int(w.get("episodes_attempted", 0)) for w in per_worker.values())
    written = sum(int(w.get("episodes_written", 0)) for w in per_worker.values())
    successful = sum(int(w.get("episodes_successful", 0)) for w in per_worker.values())
    worker_verdict["per_worker_sums"] = {
        "episodes_attempted": attempted,
        "episodes_written": written,
        "episodes_successful": successful,
    }
    # Reconcile the parent's counters against the per-worker records and the
    # on-disk outcomes. `rows_expected` is compared against the row count this
    # run was actually given (the full manifest for a full run, the 8-row subset
    # for a smoke run); that the run covered the whole manifest is asserted
    # separately by `missing_rows` and `every_candidate_index_once`.
    n_success = status_counts.get(OUTCOME_SUCCESS, 0)
    worker_verdict["parent_worker_totals_reconcile"] = bool(
        attempted == len(outcomes)
        and successful == n_success
        and written == n_success
        and counters.get("rows_finalized") == len(outcomes)
        and counters.get("rows_succeeded") == n_success
        and counters.get("rows_failed") == len(outcomes) - n_success
        and counters.get("rows_expected") == row_rec.get("expected_rows")
        and row_rec.get("finalized_rows") == len(outcomes)
    )
    worker_verdict["run_covered_full_manifest"] = (
        row_rec.get("expected_rows") == len(manifest_rows)
    )
    worker_verdict["every_worker_has_approved_final_record"] = bool(
        workers.get("complete")
        and not workers.get("missing_final_status")
        and not workers.get("silently_lost_workers")
        and not workers.get("workers_with_failed_status")
        and not workers.get("nonzero_or_unknown_exit_codes")
        and all(w.get("status") == "completed" for w in per_worker.values())
    )
    worker_verdict["ok"] = bool(
        s.get("complete")
        and row_rec.get("ok")
        and worker_verdict["every_worker_has_approved_final_record"]
        and worker_verdict["parent_worker_totals_reconcile"]
        and not row_rec.get("missing_outcome")
        and not row_rec.get("never_claimed")
        and not row_rec.get("published_without_outcome")
        and not row_rec.get("unexpected_row_dirs")
    )

    return {
        "manifest_rows": len(manifest_rows),
        "rows_with_outcome": len(outcomes),
        "rows_with_claim": len(claims),
        "missing_rows": missing,
        "stray_row_dirs": stray_dirs,
        "unresolved_claims": unresolved_claims,
        "duplicate_candidate_indices": {k: v for k, v in idx_counter.items() if v > 1},
        "duplicate_episode_ids": {k: v for k, v in eid_counter.items() if v > 1},
        "duplicate_row_hashes": {k: v for k, v in row_hash_counter.items() if v > 1},
        "every_candidate_index_once": sorted(idx_counter) == sorted(by_index),
        "status_counts": dict(status_counts),
        "retry_histogram": {str(k): v for k, v in sorted(retry_counts.items())},
        "worker_histogram": {str(k): v for k, v in sorted(worker_counts.items())},
        "hazard_by_status": {
            k: {"hazard_present": v[True], "hazard_absent": v[False]} for k, v in hazard_by_status.items()
        },
        "failure_reason_histogram": dict(failure_reasons),
        "failure_phase_histogram": dict(failure_phases),
        "retry_reason_histogram": dict(retry_reasons),
        "retries_total": sum(retry_reasons.values()),
        "hazard_label_mismatch": hazard_mismatch,
        "row_hash_mismatch": rowhash_mismatch,
        "collection_summary_present": summary is not None,
        "worker_verdict": worker_verdict,
        "ok": (
            not missing
            and not stray_dirs
            and not unresolved_claims
            and not hazard_mismatch
            and not rowhash_mismatch
            and len(outcomes) == len(manifest_rows)
            and sorted(idx_counter) == sorted(by_index)
            and all(v == 1 for v in idx_counter.values())
            and all(v == 1 for v in row_hash_counter.values())
            and worker_verdict["ok"]
        ),
        "_outcomes": outcomes,
        "_claims": claims,
    }


# --------------------------------------------------------------------------- #
# step 8 -- per-H5 audit and hash families
# --------------------------------------------------------------------------- #
def audit_h5(row_dir: Path, outcome: dict, manifest_row: dict, sensor_names: list[str]) -> dict:
    """Audit one successful row's trajectory.h5. Returns a per-row record."""
    rec: dict[str, Any] = {
        "episode_id": outcome["episode_id"],
        "candidate_index": outcome["candidate_index"],
        "hazard_present_manifest": bool(manifest_row["hazard_present"]),
        "problems": [],
    }
    p = row_dir / "trajectory.h5"
    if not p.is_file():
        rec["problems"].append("trajectory.h5 missing")
        return rec
    rec["file_size_bytes"] = p.stat().st_size

    # A. full source-file SHA-256
    rec["hash_A_file_sha256"] = sha256_file(p)

    try:
        f = h5py.File(p, "r")
    except Exception as exc:  # noqa: BLE001
        rec["problems"].append(f"open failed: {exc}")
        return rec

    with f:
        rec["opened"] = True
        attrs = dict(f.attrs)
        # identity
        if str(attrs.get("episode_id")) != outcome["episode_id"]:
            rec["problems"].append("attrs episode_id != ledger episode_id")
        if str(attrs.get("manifest_row_sha256")) != manifest_row["row_sha256"]:
            rec["problems"].append("attrs manifest_row_sha256 != manifest row_sha256")
        if int(attrs.get("candidate_index", -1)) != int(manifest_row["candidate_index"]):
            rec["problems"].append("attrs candidate_index != manifest candidate_index")
        attr_hazard = as_bool(attrs.get("hazard_present"))
        if attr_hazard is None:
            rec["problems"].append("attrs hazard_present is unparseable")
        elif attr_hazard != bool(manifest_row["hazard_present"]):
            rec["problems"].append("attrs hazard_present != manifest hazard_present")
        rec["attrs"] = {k: str(v) for k, v in attrs.items()}

        traj_keys = [k for k in f if k.startswith("traj_")]
        if len(traj_keys) != 1:
            rec["problems"].append(f"expected exactly 1 traj group, found {len(traj_keys)}")
        if not traj_keys:
            return rec
        tk = traj_keys[0]
        g = f[tk]
        rec["traj_key"] = tk

        # --- metadata group -------------------------------------------------
        if "manifest" not in f:
            rec["problems"].append("manifest metadata group missing")
            meta = {}
        else:
            meta = {k: read_scalar_str(f["manifest"][k]) for k in f["manifest"]}
        rec["metadata_keys"] = sorted(meta)

        obs_meta: dict[str, Any] = {}
        if "observations" in meta:
            try:
                obs_meta = json.loads(meta["observations"])
            except json.JSONDecodeError:
                rec["problems"].append("metadata observations is not valid JSON")
        seed_meta: dict[str, Any] = {}
        if "seed_contract" in meta:
            try:
                seed_meta = json.loads(meta["seed_contract"])
            except json.JSONDecodeError:
                rec["problems"].append("metadata seed_contract is not valid JSON")

        # seed metadata present and matching the committed manifest seed map
        if not seed_meta.get("seed_map"):
            rec["problems"].append("seed metadata absent")
        else:
            got = {k: int(v["seed_u32"]) for k, v in seed_meta["seed_map"].items()}
            want = {k: int(v["seed_u32"]) for k, v in manifest_row["seed_map"].items()}
            if int(seed_meta.get("retry_index", 0)) == 0 and got != want:
                rec["problems"].append("retry-0 seed map != committed manifest seed map")
            rec["seed_streams"] = len(got)
            rec["retry_index"] = int(seed_meta.get("retry_index", 0))

        # target/object identity, obstacle theta, robot initial state
        theta = obs_meta.get("obstacle_theta") or {}
        if not theta:
            rec["problems"].append("obstacle theta absent")
        rec["obstacle_theta_keys"] = sorted(theta)
        if not obs_meta.get("target_uid"):
            rec["problems"].append("target identity absent")
        if not obs_meta.get("selected_object"):
            rec["problems"].append("selected object identity absent")
        if not obs_meta.get("robot_initial_qpos"):
            rec["problems"].append("robot initial state absent")
        rec["target_uid"] = obs_meta.get("target_uid")
        rec["selected_object"] = obs_meta.get("selected_object")
        rec["selected_grasp"] = obs_meta.get("selected_grasp")
        rec["planner_phase_path"] = obs_meta.get("planner_phase_path")
        rec["behavior_class"] = obs_meta.get("behavior_class")

        # rendered/compiled hazard presence vs the label
        observed = obs_meta.get("observed_hazard_present")
        rec["observed_hazard_present"] = observed
        if observed is None:
            rec["problems"].append("observed hazard presence not recorded")
        elif bool(observed) != bool(manifest_row["hazard_present"]):
            rec["problems"].append("observed hazard geometry != committed hazard label")
        aabbs = obs_meta.get("obstacle_aabbs")
        rec["obstacle_aabb_count"] = len(aabbs) if aabbs is not None else None
        prot = theta.get("protrusion_present")
        rec["protrusion_present"] = prot
        if prot is not None and bool(prot) != bool(manifest_row["hazard_present"]):
            rec["problems"].append("theta protrusion_present != committed hazard label")

        # --- qpos / actions -------------------------------------------------
        for req in ("obs/agent/qpos", "obs/agent/qvel", "actions/joint_pos"):
            if req not in g:
                rec["problems"].append(f"{req} missing")
        if "obs/agent/qpos" in g:
            rec["T_h5"] = int(g["obs/agent/qpos"].shape[0])
        if "actions/joint_pos" in g:
            rec["T_actions"] = int(g["actions/joint_pos"].shape[0])

        # --- ACT RGB cameras (mp4 sidecars) ---------------------------------
        ep_idx = int(str(meta.get("episode_index", "0")) or 0) if "episode_index" in meta else 0
        cams = {}
        for cam in CAM_NAMES:
            hits = sorted(row_dir.glob(f"episode_*_{cam}.mp4"))
            hits = [h for h in hits if "_depth" not in h.name]
            if not hits:
                rec["problems"].append(f"ACT camera video missing: {cam}")
                cams[cam] = None
            else:
                cams[cam] = {"name": hits[0].name, "bytes": hits[0].stat().st_size}
        rec["act_cameras"] = cams
        rec["_ep_idx"] = ep_idx

        # --- 40 proximity streams -------------------------------------------
        prox_path = f"{tk}/obs/proximity"
        if prox_path not in f:
            rec["problems"].append("proximity group missing")
            rec["proximity_count"] = 0
        else:
            pg = f[prox_path]
            found = [k for k in pg if isinstance(pg[k], h5py.Dataset)]
            rec["proximity_count"] = len(found)
            if len(found) != 40:
                rec["problems"].append(f"expected 40 proximity streams, found {len(found)}")
            # order must reproduce the committed hash
            order_hash = hashlib.sha256(
                json.dumps(found, separators=(",", ":")).encode()
            ).hexdigest()
            sorted_hash = hashlib.sha256(
                json.dumps(sorted(found), separators=(",", ":")).encode()
            ).hexdigest()
            rec["proximity_order_sha256"] = order_hash
            rec["proximity_sorted_order_sha256"] = sorted_hash
            expected = manifest_row["sensor_order_sha256"]
            if order_hash != expected and sorted_hash != expected:
                rec["problems"].append("proximity sensor order does not match the committed hash")
            if set(found) != set(sensor_names):
                rec["problems"].append("proximity sensor names differ from the committed contract")
            bad_shapes = {}
            T = rec.get("T_h5")
            for k in found:
                sh = tuple(pg[k].shape)
                if len(sh) != 4 or sh[1:] != SENSOR_INPUT_SHAPE or (T is not None and sh[0] != T):
                    bad_shapes[k] = list(sh)
            if bad_shapes:
                rec["problems"].append(f"proximity shape != (T,4,8,8) for {len(bad_shapes)} streams")
                rec["bad_proximity_shapes"] = bad_shapes

        # --- task state / success ------------------------------------------
        for req in ("success", "fail"):
            if req not in g:
                rec["problems"].append(f"{req} dataset missing")
        if "success" in g:
            succ = np.asarray(g["success"])
            rec["success_last"] = bool(succ[-1]) if succ.size else None
        if "fail" in g:
            fail = np.asarray(g["fail"])
            rec["fail_last"] = bool(fail[-1]) if fail.size else None
            if rec.get("fail_last"):
                rec["problems"].append("fail[-1] is set on a row recorded as a success")
        if as_bool(meta.get("success")) is not True:
            rec["problems"].append("metadata success flag is not true on a success row")
        meta_hazard = as_bool(meta.get("hazard_present"))
        if meta_hazard is not None and meta_hazard != bool(manifest_row["hazard_present"]):
            rec["problems"].append("metadata hazard_present != manifest hazard_present")

        # --- truncation / corruption ---------------------------------------
        leaves = leaf_datasets(f)
        rec["leaf_dataset_count"] = len(leaves)
        try:
            for name in leaves:
                ds = f[name]
                if ds.size:
                    _ = ds[(0,) * ds.ndim] if ds.ndim else ds[()]
            # touch the final frame of the biggest arrays to catch truncation
            if "obs/agent/qpos" in g:
                _ = g["obs/agent/qpos"][-1]
            if prox_path in f:
                pg = f[prox_path]
                for k in list(pg)[:3]:
                    _ = pg[k][-1]
        except Exception as exc:  # noqa: BLE001
            rec["problems"].append(f"read error (truncated/corrupt): {exc}")

        # --- initial penetration -------------------------------------------
        clearance = theta.get("clearance")
        rec["clearance"] = clearance
        if isinstance(clearance, (int, float)) and clearance < 0:
            rec["problems"].append(f"negative initial clearance {clearance} (penetration)")

        # --- hash families B..E --------------------------------------------
        hb = hashlib.sha256()
        for name in leaves:
            ds = f[name]
            hb.update(name.encode())
            hb.update(b"\x1f")
            try:
                hash_array(hb, np.asarray(ds[()]))
            except Exception:  # noqa: BLE001
                hb.update(b"<unreadable>")
        rec["hash_B_all_leaf_sha256"] = hb.hexdigest()

        hc = hashlib.sha256()
        for name in ("obs/agent/qpos", "actions/joint_pos", "actions/joint_pos_rel"):
            if name in g:
                hc.update(name.encode())
                hc.update(b"\x1f")
                hash_array(hc, np.asarray(g[name][()]))
        rec["hash_C_core_trajectory_sha256"] = hc.hexdigest()

        task_state = {
            "robot_initial_qpos": obs_meta.get("robot_initial_qpos"),
            "robot_initial_qvel": obs_meta.get("robot_initial_qvel"),
            "object_initial_pose": obs_meta.get("object_initial_pose"),
            "mocap_pos": obs_meta.get("mocap_pos"),
            "obstacle_theta": theta,
            "hazard_present": bool(manifest_row["hazard_present"]),
        }
        rec["hash_D_task_state_sha256"] = sha256_payload(task_state)
        rec["hash_E_episode_spec_sha256"] = obs_meta.get("episode_spec_sha256")
        if not rec["hash_E_episode_spec_sha256"]:
            rec["problems"].append("immutable episode-spec hash absent")

    rec["ok"] = not rec["problems"]
    return rec


# --------------------------------------------------------------------------- #
# step 10 -- source freeze
# --------------------------------------------------------------------------- #
def freeze_source(run: Path) -> dict:
    files: list[tuple[str, int, str]] = []
    total = 0
    for p in sorted(run.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(run))
        size = p.stat().st_size
        files.append((rel, size, sha256_file(p)))
        total += size
    tree = hashlib.sha256()
    for rel, size, digest in files:
        tree.update(f"{rel}\x1f{size}\x1f{digest}\n".encode())
    return {
        "file_count": len(files),
        "total_bytes": total,
        "total_gib": round(total / (1 << 30), 4),
        "tree_sha256": tree.hexdigest(),
        "files": [{"path": r, "bytes": s, "sha256": d} for r, s, d in files],
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--source-manifest", required=True, type=Path)
    ap.add_argument("--skip-freeze", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    stack = json.loads(args.stack.read_text())
    sensor_names = list(stack["sensor_contract"]["ordered_names"])
    sensor_hash = stack["sensor_contract"]["sensor_order_hash"]
    recomputed = hashlib.sha256(
        json.dumps(sensor_names, separators=(",", ":")).encode()
    ).hexdigest()

    rec = reconcile(args.run, manifest)
    outcomes = rec.pop("_outcomes")
    rec.pop("_claims")

    manifest_rows = {r["episode_id"]: r for r in manifest["rows"]}
    per_row = []
    for eid, o in sorted(outcomes.items(), key=lambda kv: kv[1]["candidate_index"]):
        if o["status"] != OUTCOME_SUCCESS:
            continue
        per_row.append(audit_h5(args.run / "rows" / eid, o, manifest_rows[eid], sensor_names))

    # duplicate / replica audit across distinct successful rows
    def dup(field: str) -> dict:
        m = defaultdict(list)
        for r in per_row:
            v = r.get(field)
            if v:
                m[v].append(r["episode_id"])
        return {k: v for k, v in m.items() if len(v) > 1}

    dup_file = dup("hash_A_file_sha256")
    dup_all_leaf = dup("hash_B_all_leaf_sha256")
    dup_core = dup("hash_C_core_trajectory_sha256")
    dup_task = dup("hash_D_task_state_sha256")
    dup_spec = dup("hash_E_episode_spec_sha256")

    class_sizes = Counter(len(v) for v in dup_core.values())

    integrity_problems = [
        {"episode_id": r["episode_id"], "candidate_index": r["candidate_index"], "problems": r["problems"]}
        for r in per_row
        if r["problems"]
    ]

    hazard_success = Counter()
    for o in outcomes.values():
        if o["status"] == OUTCOME_SUCCESS:
            hazard_success[bool(o["hazard_present"])] += 1

    freeze = None if args.skip_freeze else freeze_source(args.run)
    if freeze is not None:
        args.source_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.source_manifest.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
        freeze_summary = {k: v for k, v in freeze.items() if k != "files"}
    else:
        freeze_summary = None

    report = {
        "schema": "hybrid_obstacle_full_collection_integrity_audit",
        "run_dir": str(args.run),
        "manifest_sha256": manifest["manifest_sha256"],
        "sensor_order_sha256_committed": sensor_hash,
        "sensor_order_sha256_recomputed": recomputed,
        "sensor_order_formula": "sha256(json.dumps(ordered_names, separators=(',',':')))",
        "sensor_order_ok": recomputed == sensor_hash,
        "reconciliation": rec,
        "successes_audited": len(per_row),
        "successes_clean": sum(1 for r in per_row if r["ok"]),
        "integrity_problems": integrity_problems,
        "duplicates": {
            "file_sha256": dup_file,
            "all_leaf_sha256": dup_all_leaf,
            "core_trajectory_sha256": dup_core,
            "task_state_sha256": dup_task,
            "episode_spec_sha256": dup_spec,
            "replica_class_size_histogram": {str(k): v for k, v in sorted(class_sizes.items())},
        },
        "distinct_successes": {
            "hazard_present": hazard_success[True],
            "hazard_absent": hazard_success[False],
        },
        "source_freeze": freeze_summary,
        "per_row": per_row,
    }
    report["ok"] = bool(
        rec["ok"]
        and report["sensor_order_ok"]
        and not integrity_problems
        and not dup_all_leaf
        and not dup_core
        and not dup_task
        and not dup_spec
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"reconciliation ok       {rec['ok']}")
    print(f"rows with outcome       {rec['rows_with_outcome']}/{rec['manifest_rows']}")
    print(f"status counts           {rec['status_counts']}")
    print(f"successes audited       {len(per_row)} (clean {report['successes_clean']})")
    print(f"distinct successes      present={hazard_success[True]} absent={hazard_success[False]}")
    print(f"duplicate core traj     {len(dup_core)}")
    print(f"duplicate task state    {len(dup_task)}")
    print(f"duplicate all-leaf      {len(dup_all_leaf)}")
    if freeze_summary:
        print(f"source files            {freeze_summary['file_count']} ({freeze_summary['total_gib']} GiB)")
        print(f"source tree sha256      {freeze_summary['tree_sha256']}")
    print(f"AUDIT {'OK' if report['ok'] else 'FAILED'}")
    if integrity_problems:
        for p in integrity_problems[:20]:
            print(f"  [{p['candidate_index']}] {p['problems']}")
    print(f"wrote {args.out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
