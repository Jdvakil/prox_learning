#!/usr/bin/env python3
"""Audit the hazard draw of the completed hybrid-obstacle collection.

Read-only. Reads the sampler source to record the configured draw, then measures
the realized hazard rates from the collection itself.

The collection contains exact replicas (concurrent workers replay the same RNG
stream), so every rate is reported twice: ``as_written`` over all stored
trajectories, and ``distinct`` over one representative per identical-content
class. Only the ``distinct`` rates carry independent information; the
``as_written`` rates are pseudo-replicated and their intervals are not valid
sampling intervals. Exact Clopper-Pearson intervals are given for each
non-nested headline rate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import beta

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLER = REPO_ROOT / "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py"
SAMPLER_CFG = (
    REPO_ROOT
    / "submodules/molmospaces/molmo_spaces/data_generation/config"
    / "object_manipulation_datagen_configs.py"
)
TASK_SAMPLER = REPO_ROOT / "submodules/molmospaces/molmo_spaces/tasks/task_sampler.py"


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> dict[str, float | None]:
    if n == 0:
        return {"point": None, "lower": None, "upper": None}
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return {"point": k / n, "lower": lower, "upper": upper, "k": k, "n": n}


def read_sampler_contract() -> dict[str, Any]:
    src = SAMPLER.read_text()
    obstacle_p = re.search(r"OBSTACLE_P\s*=\s*([0-9.]+)", src)
    draw = re.search(r"if\s+(np\.random\.random\(\))\s*<\s*(self\.OBSTACLE_P)", src)
    check_p = re.search(r"class ObstacleFumehoodPickCheckSampler.*?OBSTACLE_P\s*=\s*([0-9.]+)", src, re.S)
    ts = TASK_SAMPLER.read_text()
    seed_fn = re.search(
        r"def seed_task_sampling\(self, seed\) -> None:\n(.*?)\n\n", ts, re.S
    )
    seed_init = re.search(
        r"seed = self\.config\.seed if self\.config\.seed is not None else ([^\n]+)", ts
    )
    cfg = SAMPLER_CFG.read_text()
    return {
        "sampler_file": str(SAMPLER),
        "configured_obstacle_p": float(obstacle_p.group(1)) if obstacle_p else None,
        "check_sampler_obstacle_p": float(check_p.group(1)) if check_p else None,
        "comparison_convention": (
            f"{draw.group(1)} < {draw.group(2)}" if draw else "not found"
        ),
        "rng_owner": (
            "global numpy legacy RandomState via np.random.random(), seeded by "
            "TaskSampler.seed_task_sampling"
        ),
        "seed_function_body": seed_fn.group(1).strip() if seed_fn else None,
        "seed_fallback_when_unset": seed_init.group(1).strip() if seed_init else None,
        "draw_site": "ObstacleFumehoodPickSampler._draw_theta",
        "hazard_fields_set_on_draw": [
            "cell='bar'",
            "protrusion_present=True",
            "protr_name",
            "protr_wall",
            "protr_pos_frac",
            "bar_face_y",
            "intrusion",
            "residual_margin",
            "obj_gap",
        ],
        "rejected_attempts_consume_draws": (
            "yes — a rejected rollout is re-sampled through _draw_theta, so each "
            "planner rejection consumes a fresh Bernoulli draw from the same stream"
        ),
        "hazard_affects_reset_or_rollout_success": (
            "yes — hazard episodes route through ObstacleAwarePickPlannerPolicy's "
            "deflection and can fail IK at pregrasp/lift, so success is not "
            "independent of hazard presence; conditional success rates are reported"
        ),
        "hazard_metadata_can_differ_from_rendered_bar": (
            "no — protrusion_present is set in the same _draw_theta call that emits "
            "protr_center/protr_half and the extra obstacle_aabbs entry, and "
            "_apply_theta colours that same geom; the integrity audit confirms "
            "label and geometry agree on every trajectory"
        ),
        "config_seed_note": (
            "FrankaSkinHybridObstacleConfig inherits a fixed seed; every worker "
            "constructs its own task sampler and seeds the global RNG with that same "
            "value, so worker RNG streams overlap completely"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrity_report", required=True)
    parser.add_argument("--worker_log")
    parser.add_argument("--config_seed", type=int)
    args = parser.parse_args()

    report = json.loads(Path(args.integrity_report).read_text())
    rows = report["trajectories_detail"]

    by_content: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_content[r["content_sha256"]].append(r)
    # deterministic representative: lowest trajectory_id in each identical-content class
    distinct = [sorted(v, key=lambda r: r["trajectory_id"])[0] for v in by_content.values()]
    distinct.sort(key=lambda r: r["trajectory_id"])

    def rates(sample: list[dict[str, Any]], label: str) -> dict[str, Any]:
        n = len(sample)
        hz = [r for r in sample if r["hazard_recorded"]]
        ab = [r for r in sample if not r["hazard_recorded"]]
        succ = [r for r in sample if r["successful"]]
        succ_hz = [r for r in succ if r["hazard_recorded"]]
        return {
            "label": label,
            "trajectories": n,
            "hazard_present": len(hz),
            "written_hazard_rate": clopper_pearson(len(hz), n),
            "successful": len(succ),
            "successful_hazard_present": len(succ_hz),
            "successful_hazard_absent": len(succ) - len(succ_hz),
            "successful_hazard_rate": clopper_pearson(len(succ_hz), len(succ)),
            "success_rate_given_hazard_present": clopper_pearson(
                sum(1 for r in hz if r["successful"]), len(hz)
            ),
            "success_rate_given_hazard_absent": clopper_pearson(
                sum(1 for r in ab if r["successful"]), len(ab)
            ),
            "overall_success_rate": clopper_pearson(len(succ), n),
        }

    # accepted theta draws recorded in the worker log
    theta: dict[str, Any] = {"available": False}
    if args.worker_log and Path(args.worker_log).is_file():
        text = re.sub(r"\x1b\[[0-9;]*m", "", Path(args.worker_log).read_text(errors="replace"))
        cells = re.findall(r"\[Enclosure\] cell=([a-z]+)", text)
        n = len(cells)
        bar = sum(1 for c in cells if c == "bar")
        multiplicity = len(rows) / len(distinct) if distinct else 1.0
        theta = {
            "available": True,
            "accepted_theta_draws_logged": n,
            "hazard_draws_logged": bar,
            "as_written_accepted_theta_hazard_rate": clopper_pearson(bar, n),
            "replica_multiplicity_of_stored_trajectories": multiplicity,
            "effective_independent_draws_estimate": round(n / multiplicity) if multiplicity else None,
            "note": (
                "Logged draws include the replicated worker streams. Because concurrent "
                "workers replay an identical RNG sequence, the logged count overstates the "
                "number of independent Bernoulli draws by roughly the replica multiplicity, "
                "so the as-written interval is narrower than the data justify."
            ),
        }

    out = {
        "schema_version": "hybrid_obstacle_hazard_audit_v1",
        "sampler_contract": read_sampler_contract(),
        "config_seed": args.config_seed,
        "worker_seed_overlap": {
            "each_worker_builds_its_own_task_sampler": True,
            "sampler_seeds_global_rng_at_construction": True,
            "seed_depends_on_worker_id": False,
            "seed_depends_on_house_index": False,
            "consequence": (
                "all workers start from the identical RNG state, and because every "
                "configured house index is congruent to 1 mod 24 and therefore selects "
                "the identical red-cup task, the k-th house processed by any worker "
                "replays the same episode sequence"
            ),
        },
        "as_written": rates(rows, "as_written (includes exact replicas)"),
        "distinct": rates(distinct, "distinct (one representative per content class)"),
        "replication": {
            "stored_trajectories": len(rows),
            "distinct_content_classes": len(distinct),
            "multiplicity_histogram": {
                str(k): sum(1 for v in by_content.values() if len(v) == k)
                for k in sorted({len(v) for v in by_content.values()})
            },
        },
        "accepted_theta_draws": theta,
        "interval_note": (
            "Clopper-Pearson exact 95% intervals. Intervals are reported for the "
            "non-nested headline rates: written hazard rate, successful hazard rate, "
            "and success conditional on hazard presence/absence."
        ),
    }
    json.dump(out, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
