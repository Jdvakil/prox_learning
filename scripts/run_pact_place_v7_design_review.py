#!/usr/bin/env python3
"""A0g: run the v7 lattice until 3 clean successes, render every episode, stop.

This is a human design review. It is not a gate, not a probe, and must never
be quoted as a clean-rate estimate. Probe and gate do not run from this
script.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MOLMO = ROOT / "submodules" / "molmospaces"
for search_path in (ROOT / "scripts", MOLMO):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pact_place_corridor_contract import (  # noqa: E402
    DESIGN_REVIEW_STOP_CLEAN,
    N_DESIGN_REVIEW_ROWS,
    V7_DESIGN_REVIEW_MASTER_SEED,
    build_design_review_contract,
    load_design_review_contract,
    sha256_payload,
)
from run_pact_place_expert_screen import (  # noqa: E402
    _protected_eval_processes,
    run_row,
    verify_protected_artifacts,
    write_json_atomic,
)
from run_pact_place_v7_replay_videos import (  # noqa: E402
    CONFIG_PATH,
    DEFAULT_OUTPUT,
    SCREEN_ROOT,
    build_jobs,
    render_row,
)
from run_pact_place_v7_scoring_check import run_scoring_check  # noqa: E402

V6C_ANALYSIS = ROOT / "diagnostics_output/pact_place_swept_volume_v7/analysis.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _v6c_baseline() -> dict[str, Any]:
    if not V6C_ANALYSIS.is_file():
        return {}
    analysis = json.loads(V6C_ANALYSIS.read_text())
    return dict(analysis.get("v6c_baseline") or {})


def write_review_config() -> dict[str, Any]:
    document = build_design_review_contract(master_seed=V7_DESIGN_REVIEW_MASTER_SEED)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    loaded = load_design_review_contract(CONFIG_PATH)
    if loaded["config_sha256"] != document["config_sha256"]:
        raise RuntimeError("design-review config changed on reload")
    return loaded


def _episode_record(row: dict[str, Any], result: dict[str, Any], clip: dict[str, Any] | None) -> dict[str, Any]:
    tracking = result.get("terminal_tracking") or {}
    audit = result.get("contact_audit") or {}
    metrics = (clip or {}).get("a0d_metrics") or {}
    return {
        "role_index": row["role_index"],
        "episode_id": row["episode_id"],
        "intrusion_side": row["intrusion_side"],
        "status": result.get("status"),
        "clean_success": bool(result.get("clean_success")),
        "task_success": bool(result.get("task_success")),
        "terminal_policy_phase": result.get("terminal_policy_phase"),
        "failure_branch": tracking.get("check_failure_branch"),
        "clutter_contact_pairs": int(
            (audit.get("contact_class_totals") or {}).get("clutter", 0)
        ),
        "clutter_contact_by_phase": (clip or {}).get("clutter_contact_by_phase") or {},
        "clutter_contact_by_link": (clip or {}).get("clutter_contact_by_link") or {},
        "skin_engagement": metrics.get("skin_engagement"),
        "wrist_visibility": metrics.get("wrist_visibility"),
        "min_clearance_by_link_m": metrics.get("min_clearance_by_link_m"),
        "clip": None if clip is None else clip.get("clip"),
    }


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ["PACT_CONTACT_AUDIT_SUMMARY_ONLY"] = "1"
    os.environ.pop("DISPLAY", None)

    protected = _protected_eval_processes()
    if protected:
        raise SystemExit(f"protected confirmatory evaluation is active: {protected}")

    SCREEN_ROOT.mkdir(parents=True, exist_ok=True)
    contract = write_review_config()
    verify_protected_artifacts(contract)
    scene_xml = str(ROOT / contract["scene"]["xml"])

    print("A0f scoring check: injecting pact_clutter_15 into the swept volume", flush=True)
    scoring = run_scoring_check()
    if not scoring.get("passed"):
        raise SystemExit("A0f scoring check failed; A0g will not run")
    print(
        "scoring check passed: "
        f"link={scoring['first_link_clutter_pair']['link']} "
        f"phase={scoring['first_link_clutter_pair']['policy_phase']}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    clean = 0
    stop_reason = "hard_cap"
    for row in contract["expert_screen_rows"]:
        if clean >= DESIGN_REVIEW_STOP_CLEAN:
            stop_reason = "reached_three_clean_successes"
            break
        if len(results) >= N_DESIGN_REVIEW_ROWS:
            break
        result = run_row(
            row,
            config_sha256=contract["config_sha256"],
            output_root=str(SCREEN_ROOT),
            scene_xml=scene_xml,
        )
        results.append(result)
        if result.get("clean_success"):
            clean += 1
        print(
            f"review={row['role_index']:02d} side={row['intrusion_side']} "
            f"status={result['status']} clean={result.get('clean_success')} "
            f"task={result.get('task_success')} "
            f"phase={result.get('terminal_policy_phase')} "
            f"accum_clean={clean}/{DESIGN_REVIEW_STOP_CLEAN}",
            flush=True,
        )
        if clean >= DESIGN_REVIEW_STOP_CLEAN:
            stop_reason = "reached_three_clean_successes"
            break
    if clean < DESIGN_REVIEW_STOP_CLEAN:
        stop_reason = "hard_cap_without_three_clean_successes"

    ran_rows = {int(item["role_index"]) for item in results}
    jobs = [
        job
        for job in build_jobs(contract, DEFAULT_OUTPUT, sorted(ran_rows))
        if int(job["role_index"]) in ran_rows
    ]
    clips: list[dict[str, Any]] = []
    for job in jobs:
        clip = render_row(job)
        clips.append(clip)
        print(
            f"clip={clip['clip']} skin={clip['a0d_metrics']['skin_engagement']:.4f} "
            f"wrist={clip['a0d_metrics']['wrist_visibility']:.4f}",
            flush=True,
        )
    clips_by_row = {int(item["role_index"]): item for item in clips}
    episodes = []
    for row, result in zip(
        [item for item in contract["expert_screen_rows"] if int(item["role_index"]) in ran_rows],
        results,
    ):
        episodes.append(_episode_record(row, result, clips_by_row.get(int(row["role_index"]))))

    failures = [item for item in episodes if not item["clean_success"]]
    skins = [
        float(item["skin_engagement"])
        for item in episodes
        if item["skin_engagement"] is not None
    ]
    wrists = [
        float(item["wrist_visibility"])
        for item in episodes
        if item["wrist_visibility"] is not None
    ]
    v6c = _v6c_baseline()
    episodes_to_third = None
    seen_clean = 0
    for item in episodes:
        if item["clean_success"]:
            seen_clean += 1
        if seen_clean == DESIGN_REVIEW_STOP_CLEAN:
            episodes_to_third = int(item["role_index"]) + 1
            break

    summary = {
        "schema_version": "pact_place_v7_design_review_v1",
        "role": "human_design_review_not_a_gate",
        "authorizes_collection": False,
        "authorizes_probe": False,
        "authorizes_gate": False,
        "not_a_clean_rate_estimate": True,
        "do_not_quote_clean_rate": True,
        "created_utc": utc_now(),
        "config_sha256": contract["config_sha256"],
        "master_seed": contract["master_seed"],
        "design_review_seeds": contract["design_review_seeds"],
        "scoring_check_passed": True,
        "scoring_check_path": str(
            (
                ROOT
                / "diagnostics_output/pact_place_corridor_v7_design_review/scoring_check/scoring_check.json"
            ).relative_to(ROOT)
        ),
        "n_episodes_run": len(results),
        "n_clean_successes": clean,
        "stop_reason": stop_reason,
        "episodes_to_third_clean_success": episodes_to_third,
        "hard_cap": N_DESIGN_REVIEW_ROWS,
        "stop_at_clean_successes": DESIGN_REVIEW_STOP_CLEAN,
        "failure_branches": [
            {
                "role_index": item["role_index"],
                "terminal_policy_phase": item["terminal_policy_phase"],
                "failure_branch": item["failure_branch"],
                "clutter_contact_pairs": item["clutter_contact_pairs"],
            }
            for item in failures
        ],
        "v6c_baseline": {
            "skin_engagement": v6c.get("skin_engagement"),
            "wrist_visibility": v6c.get("wrist_visibility"),
            "skin_engagement_passage_link4_6": v6c.get(
                "skin_engagement_passage_link4_6"
            ),
        },
        "v7_review_metrics": {
            "n_with_metrics": len(skins),
            "skin_engagement_mean": None if not skins else float(sum(skins) / len(skins)),
            "skin_engagement_min": None if not skins else float(min(skins)),
            "skin_engagement_max": None if not skins else float(max(skins)),
            "wrist_visibility_mean": None if not wrists else float(sum(wrists) / len(wrists)),
            "wrist_visibility_min": None if not wrists else float(min(wrists)),
            "wrist_visibility_max": None if not wrists else float(max(wrists)),
            "beats_v6c_skin_engagement": None
            if not skins or v6c.get("skin_engagement") is None
            else float(sum(skins) / len(skins)) > float(v6c["skin_engagement"]),
            "wrist_visibility_not_increased": None
            if not wrists or v6c.get("wrist_visibility") is None
            else float(sum(wrists) / len(wrists)) <= float(v6c["wrist_visibility"]),
        },
        "episodes": episodes,
        "clips_dir": str(DEFAULT_OUTPUT.relative_to(ROOT)),
        "next_action": "stop_for_human_review_of_clips",
    }
    summary["design_review_sha256"] = sha256_payload(summary)
    write_json_atomic(SCREEN_ROOT / "design_review.json", summary)
    verify_protected_artifacts(contract)
    print(json.dumps(
        {
            "stop_reason": stop_reason,
            "n_episodes_run": len(results),
            "n_clean_successes": clean,
            "episodes_to_third_clean_success": episodes_to_third,
            "clips_dir": str(DEFAULT_OUTPUT),
            "not_a_clean_rate_estimate": True,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
