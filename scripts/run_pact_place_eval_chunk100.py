#!/usr/bin/env python3
"""Run Amine's frozen 40-row place protocol on local ACT / PACT checkpoints.

Amine's ``eval_pact_place_chunk100_row.py`` cannot load these ckpts: it wants
chunk 100, frozen 32-d embeddings, ``run_manifest.json``, and hashed encoder
paths on his machine. This launcher keeps his 40-row scene contract and drives
``eval_act_place_corridor.py``, which reads ``prox_config.json``.

Arms:
  ACT           cameras-only chunk-50
  PACT          PACT-raw per-sensor
  PACT_READOUT  finetuned 128-d CLS
  PACT_PERMUTED not available (needs Amine's 32-d token plan + his PACT ckpt)

One GPU: default ``--workers 1``. ``--workers 10`` is clamped. Smoke = rows 0,1.
Worker stdout streams live. Default PACT skin is ``mj_multiRay`` (EGL was
2121 s / 2 eps gated). Pass ``--egl-prox`` only to restore the rasterizer.

    conda activate mlspaces
    cd /home/jaydv/code/prox_learning
    export OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
    python scripts/run_pact_place_eval_chunk100.py \
        --manifest configs/pact_place_eval_chunk100_manifest.json \
        --output-root /home/jaydv/code/prox_learning/eval_output/pact_place_chunk100_jay \
        --mode smoke --workers 1 --arms PACT_READOUT"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "submodules" / "act"
MOLMO = Path("/home/jaydv/code/molmospaces-pact-place")
SCRIPTS = ROOT / "scripts"
CKPT_ROOT = ACT / "ckpts" / "pact_place_corridor_v5"
DEFAULT_CKPTS = {
    "ACT": CKPT_ROOT / "20260825_161821_act_place_corridor_s0",
    "PACT": CKPT_ROOT / "20260825_215846_pact_place_corridor_raw_s0",
    "PACT_READOUT": CKPT_ROOT / "20260828_003136_pact_place_corridor_readout_s0",
}
KNOWN_ARMS = ("ACT", "PACT", "PACT_READOUT", "PACT_PERMUTED")
MAX_WORKERS = 2


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def shard_indices(n: int, workers: int, worker_id: int) -> list[int]:
    return [idx for idx in range(n) if idx % workers == worker_id]


def command_for(
    *,
    python: Path,
    arm: str,
    ckpt_dir: Path,
    manifest: Path,
    output_dir: Path,
    role_indices: list[int],
    horizon: int,
    chunk_size: int,
    molmo: Path,
    fast_prox_rays: bool,
    skip_existing: bool = True,
) -> list[str]:
    cmd = [
        str(python),
        str(ACT / "eval_act_place_corridor.py"),
        "--ckpt_dir",
        str(ckpt_dir),
        "--output_dir",
        str(output_dir),
        "--manifest",
        str(manifest),
        "--arm-name",
        arm,
        "--chunk_size",
        str(chunk_size),
        "--task_horizon",
        str(horizon),
        "--temp_agg_off",
        "--molmospaces_root",
        str(molmo),
        "--role-indices",
        ",".join(str(idx) for idx in role_indices),
    ]
    if not fast_prox_rays:
        cmd.append("--egl_prox")
    if not skip_existing:
        cmd.append("--no-skip-existing")
    return cmd


def run_shard(command: list[str], env: dict[str, str], log_dir: Path) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    start = time.monotonic()
    env = {**env, "PYTHONUNBUFFERED": "1"}
    stdout_path = log_dir / "stdout.log"
    with stdout_path.open("w") as log_f:
        proc = subprocess.Popen(
            command,
            cwd=str(ACT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        returncode = proc.wait()
    (log_dir / "stderr.log").write_text("")
    return {
        "command": command,
        "started_utc": started,
        "finished_utc": utc_now(),
        "wall_clock_seconds": time.monotonic() - start,
        "returncode": returncode,
        "log_dir": str(log_dir),
    }


def load_arm_results(arm_dir: Path) -> list[dict[str, Any]]:
    rows = []
    if not arm_dir.is_dir():
        return rows
    for result_path in sorted(arm_dir.glob("*/result.json")):
        payload = json.loads(result_path.read_text())
        payload["_path"] = str(result_path)
        rows.append(payload)
    return rows


def summarize_arm(arm: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    success = sum(1 for row in rows if row.get("task_success"))
    strict = sum(1 for row in rows if row.get("collision_free_task_success"))
    grip = sum(1 for row in rows if row.get("gripper_close_commanded"))
    bar = 0
    for row in rows:
        metric = row.get("episode_metric") or {}
        bar += int(metric.get("hit_bar") or 0)
    return {
        "arm": arm,
        "episodes": n,
        "task_success": success,
        "task_success_rate": (success / n) if n else None,
        "collision_free_task_success": strict,
        "collision_free_task_success_rate": (strict / n) if n else None,
        "bar_hits": bar,
        "bar_hit_rate": (bar / n) if n else None,
        "gripper_close_commanded": grip,
        "gripper_close_rate": (grip / n) if n else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--token-plan-manifest",
        type=Path,
        default=None,
        help="Ignored. PACT_PERMUTED is not wired for local ckpts.",
    )
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Cap how many manifest rows run. Overrides smoke=2 / full=40.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--arms", nargs="+", default=["ACT", "PACT"])
    parser.add_argument("--ckpt-act", type=Path, default=DEFAULT_CKPTS["ACT"])
    parser.add_argument("--ckpt-pact", type=Path, default=DEFAULT_CKPTS["PACT"])
    parser.add_argument(
        "--ckpt-pact-readout", type=Path, default=DEFAULT_CKPTS["PACT_READOUT"]
    )
    parser.add_argument("--task-horizon", type=int, default=900)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument(
        "--fast-prox-rays",
        action="store_true",
        default=True,
        help="Default. PACT skin via mj_multiRay. Kept so old commands still run.",
    )
    parser.add_argument(
        "--egl-prox",
        action="store_true",
        help=(
            "Slow 40-cam EGL. Smoke PACT was 2121s/2 eps with chunk-gate on. "
            "Do not use for the 18-day loop."
        ),
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Rerun rows even when result.json exists (killed EGL leftover).",
    )
    parser.add_argument("--molmospaces-root", type=Path, default=MOLMO)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Interpreter. Default is the current python.",
    )
    args = parser.parse_args()

    unknown = [arm for arm in args.arms if arm not in KNOWN_ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {KNOWN_ARMS}")

    skipped = []
    arms = []
    for arm in args.arms:
        if arm == "PACT_PERMUTED":
            skipped.append(
                "PACT_PERMUTED skipped: local ckpts are not 32-d frozen embeddings; "
                "Amine's token plan is not on this disk."
            )
            continue
        arms.append(arm)
    if not arms:
        raise SystemExit("no runnable arms left after dropping PACT_PERMUTED")

    workers = int(args.workers)
    if workers < 1:
        raise SystemExit("workers must be >= 1")
    if workers > MAX_WORKERS:
        print(
            f"[chunk100-jay] clamp workers {workers} -> {MAX_WORKERS} "
            "(one GPU; Amine's 10-wide pool OOMs here)",
            flush=True,
        )
        workers = MAX_WORKERS

    sys.path.insert(0, str(SCRIPTS))
    from pact_place_eval_chunk100_contract import load_manifest

    manifest = load_manifest(args.manifest)
    all_rows = list(manifest["rows"])
    if args.limit_rows is not None:
        if args.limit_rows < 1:
            raise SystemExit("limit-rows must be >= 1")
        rows = all_rows[: args.limit_rows]
    else:
        rows = all_rows[:2] if args.mode == "smoke" else all_rows
    n_rows = len(rows)
    role_indices = [int(row["role_index"]) for row in rows]

    ckpts = {
        "ACT": args.ckpt_act.expanduser().resolve(),
        "PACT": args.ckpt_pact.expanduser().resolve(),
        "PACT_READOUT": args.ckpt_pact_readout.expanduser().resolve(),
    }
    molmo = args.molmospaces_root.expanduser().resolve()
    if not (molmo / "molmo_spaces").is_dir():
        raise SystemExit(f"molmospaces worktree missing at {molmo}")
    python = args.python.expanduser().resolve()

    for arm in arms:
        ckpt = ckpts[arm] / "policy_best.ckpt"
        stats = ckpts[arm] / "dataset_stats.pkl"
        if not ckpt.is_file():
            raise SystemExit(f"{arm} missing {ckpt}")
        if not stats.is_file():
            raise SystemExit(f"{arm} missing {stats}")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2"),
        "PYTHONPATH": os.pathsep.join(
            [str(molmo), str(ACT), str(ROOT), str(SCRIPTS)]
            + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
        ),
        "MLSPACES_ASSETS_DIR": str(ROOT / "assets"),
    }
    env.pop("DISPLAY", None)

    for line in skipped:
        print(f"[chunk100-jay] {line}", flush=True)
    print(
        f"[chunk100-jay] mode={args.mode} arms={arms} rows={n_rows} "
        f"workers={workers} horizon={args.task_horizon} chunk={args.chunk_size} "
        f"fast_prox_rays={not bool(args.egl_prox)} egl_prox={bool(args.egl_prox)}",
        flush=True,
    )

    started = utc_now()
    driver_results = []
    errors = []
    for arm in arms:
        arm_dir = output_root / arm.lower()
        arm_dir.mkdir(parents=True, exist_ok=True)
        shards = []
        for worker_id in range(workers):
            indices = [
                role_indices[idx]
                for idx in shard_indices(n_rows, workers, worker_id)
            ]
            if not indices:
                continue
            shards.append((worker_id, indices))

        def launch(worker_id: int, indices: list[int], arm_name: str = arm) -> dict[str, Any]:
            log_dir = arm_dir / f"_worker{worker_id:02d}"
            cmd = command_for(
                python=python,
                arm=arm_name,
                ckpt_dir=ckpts[arm_name],
                manifest=args.manifest.resolve(),
                output_dir=arm_dir,
                role_indices=indices,
                horizon=args.task_horizon,
                chunk_size=args.chunk_size,
                molmo=molmo,
                fast_prox_rays=(not bool(args.egl_prox)),
                skip_existing=(not bool(args.no_skip_existing)),
            )
            print(
                f"[chunk100-jay] {arm_name} worker {worker_id} rows={indices[:8]}"
                f"{'...' if len(indices) > 8 else ''}",
                flush=True,
            )
            result = run_shard(cmd, env, log_dir)
            result.update({"arm": arm_name, "worker_id": worker_id, "role_indices": indices})
            return result

        if workers == 1:
            shard_results = [launch(*shards[0])]
        else:
            shard_results = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(launch, worker_id, indices)
                    for worker_id, indices in shards
                ]
                for future in as_completed(futures):
                    shard_results.append(future.result())
        for result in shard_results:
            driver_results.append(result)
            if result["returncode"] != 0:
                errors.append(
                    {
                        "arm": arm,
                        "worker_id": result["worker_id"],
                        "returncode": result["returncode"],
                        "log_dir": result["log_dir"],
                    }
                )
                print(
                    f"[chunk100-jay] {arm} worker {result['worker_id']} "
                    f"FAILED rc={result['returncode']} see {result['log_dir']}/stderr.log",
                    flush=True,
                )
            else:
                print(
                    f"[chunk100-jay] {arm} worker {result['worker_id']} "
                    f"ok {result['wall_clock_seconds']:.1f}s",
                    flush=True,
                )

    arm_summaries = {}
    for arm in arms:
        rows_out = load_arm_results(output_root / arm.lower())
        arm_summaries[arm] = summarize_arm(arm, rows_out)
        print(
            f"[chunk100-jay] {arm} "
            f"place={arm_summaries[arm]['task_success']}/{arm_summaries[arm]['episodes']} "
            f"strict={arm_summaries[arm]['collision_free_task_success']}/"
            f"{arm_summaries[arm]['episodes']} "
            f"bar={arm_summaries[arm]['bar_hits']}/{arm_summaries[arm]['episodes']} "
            f"grip={arm_summaries[arm]['gripper_close_commanded']}/"
            f"{arm_summaries[arm]['episodes']}",
            flush=True,
        )

    summary = {
        "schema_version": "pact_place_eval_chunk100_jay_launcher_v1",
        "mode": args.mode,
        "started_utc": started,
        "finished_utc": utc_now(),
        "workers": workers,
        "workers_requested": args.workers,
        "arms_requested": args.arms,
        "arms": arms,
        "skipped": skipped,
        "rows": n_rows,
        "limit_rows": args.limit_rows,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "task_horizon": args.task_horizon,
        "chunk_size": args.chunk_size,
        "temp_agg_off": True,
        "fast_prox_rays": (not bool(args.egl_prox)),
        "egl_prox": bool(args.egl_prox),
        "ckpts": {arm: str(ckpts[arm]) for arm in arms},
        "arm_summaries": arm_summaries,
        "errors": errors,
        "driver_results": driver_results,
        "token_plan_manifest": (
            str(args.token_plan_manifest) if args.token_plan_manifest else None
        ),
        "note": (
            "Scenes are Amine's frozen 40-row chunk100 contract. Policies are "
            "local chunk-50 ckpts with --temp_agg_off. Not byte-comparable to "
            "Amine's chunk-100 / 32-d eval."
            + (
                " egl_prox=1; 40-cam rasterizer, ~18 min/ep measured."
                if args.egl_prox
                else " mj_multiRay skin (default). Not bit-identical to EGL."
            )
        ),
    }
    out = output_root / f"{args.mode}_launcher_summary.json"
    write_json(out, summary)
    print(f"[chunk100-jay] wrote {out}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
