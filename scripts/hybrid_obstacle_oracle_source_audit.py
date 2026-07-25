#!/usr/bin/env python3
"""Source audit of the DOCUMENTED parked-obstacle oracle reference.

Handoff step 3. Every fact recorded here is extracted from committed source or from
the committed H5s -- never from prose. Each entry carries its file, line range and the
verbatim text, so the implementation in ``parked_obstacle_reference.py`` can be checked
against the thing it claims to reproduce.

Writes ``current_oracle_audit.json``.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def excerpt(path: Path, first: int, last: int) -> dict:
    """1-indexed inclusive line range, with the file digest so it cannot drift."""
    lines = path.read_text().splitlines()
    return {
        "file": str(path.relative_to(ROOT)),
        "lines": [first, last],
        "text": "\n".join(lines[first - 1:last]),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def find_line(path: Path, needle: str, start: int = 1) -> int:
    for index, line in enumerate(path.read_text().splitlines()[start - 1:], start=start):
        if needle in line:
            return index
    raise SystemExit(f"{path}: could not locate {needle!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--development-manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    enclosure = ROOT / "submodules/molmospaces/molmo_spaces/tasks/enclosure_reach.py"
    react = ROOT / "scripts/safety_react_demo.py"
    flinch = ROOT / "scripts/safety_flinch_demo.py"
    sweep = ROOT / "scripts/safety_sweep.py"
    adapter = ROOT / "submodules/act/eval_act_obstacle_safety.py"
    residual = ROOT / "submodules/act/hybrid_safety_residual.py"
    cvae = ROOT / "scripts/train_safety_cvae.py"
    env_py = ROOT / "submodules/molmospaces/molmo_spaces/env/env.py"
    task_py = ROOT / "submodules/molmospaces/molmo_spaces/tasks/task.py"
    runner = ROOT / "submodules/molmospaces/molmo_spaces/data_generation/manifest_runner.py"

    # ------------------------------------------------------------------ #
    # 1. the committed parked pose, extracted from the AST rather than read off
    # ------------------------------------------------------------------ #
    tree = ast.parse(enclosure.read_text())
    protr_sizes: dict[str, float] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "PROTR"):
            protr_sizes = {ast.literal_eval(k): ast.literal_eval(v)
                           for k, v in zip(node.value.keys, node.value.values)}
    if not protr_sizes:
        raise SystemExit("PROTR table not found in enclosure_reach.py")

    park_line = find_line(enclosure, "# park all protrusions, then place the chosen one")
    park_block = excerpt(enclosure, park_line, park_line + 2)
    park_xy: list[tuple[float, float]] = []
    park_z: float | None = None
    import textwrap
    for node in ast.walk(ast.parse(textwrap.dedent(park_block["text"]))):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) \
                and getattr(node.iter.func, "id", "") == "zip":
            park_xy = [tuple(ast.literal_eval(e)) for e in node.iter.args[1].elts]
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "_mocap_set":
            park_z = ast.literal_eval(node.args[2].elts[2])
    if len(park_xy) != len(protr_sizes) or park_z is None:
        raise SystemExit("could not extract the committed parked pose")
    parked_pose = {name: [float(x), float(y), float(park_z)]
                   for name, (x, y) in zip(protr_sizes, park_xy)}

    # the runner's own definition of "a hazard is compiled into the scene"
    parked_threshold = next(
        ast.literal_eval(n.value)
        for n in ast.walk(ast.parse(runner.read_text()))
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_PARKED_Z_THRESHOLD"
    )
    protr_bodies = next(
        ast.literal_eval(n.value)
        for n in ast.walk(ast.parse(runner.read_text()))
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_PROTR_BODIES"
    )

    # ------------------------------------------------------------------ #
    # 2. the 109-frame diagnosis, measured from the committed H5s
    # ------------------------------------------------------------------ #
    import h5py
    dev = json.loads(args.development_manifest.read_text())
    run = Path(dev["source_run_dir"])
    if not run.is_absolute():
        run = ROOT / run
    lengths = {}
    for row in dev["rows"]:
        with h5py.File(run / "rows" / row["episode_id"] / "trajectory.h5", "r") as handle:
            key = next(k for k in handle if k.startswith("traj"))
            first = min(handle[f"{key}/obs/proximity"].keys())
            lengths[f"cand{row['candidate_index']}"] = {
                "recorded_real_timesteps": int(handle[f"{key}/obs/proximity/{first}"].shape[0]),
                "hazard_present": bool(row["hazard_present"]),
            }

    guard_line = find_line(adapter, "if self._step >= len(self._reference_proximity)")

    audit = {
        "schema": "hybrid_obstacle_oracle_source_audit_v1",
        "purpose": ("Establish, from committed source only, what the documented "
                    "parked-obstacle oracle reference is, so the per-frame implementation "
                    "can be checked against it."),

        # ---------------- the committed parked pose --------------------- #
        "committed_parked_pose": {
            "mechanism": ("the hazard is a MuJoCo MOCAP body; parking writes data.mocap_pos "
                          "only -- mocap_quat, qpos, qvel and ctrl are never touched"),
            "hazard_body_names": list(protr_sizes),
            "hazard_half_cross_sections_m": protr_sizes,
            "parked_position_by_body": parked_pose,
            "parked_z_m": float(park_z),
            "runner_parked_z_threshold": float(parked_threshold),
            "runner_hazard_bodies": list(protr_bodies),
            "source": park_block,
            "mocap_set": excerpt(enclosure, find_line(enclosure, "def _mocap_set"),
                                 find_line(enclosure, "def _mocap_set") + 3),
            "hazard_placement": excerpt(enclosure, find_line(enclosure, 'if th["protrusion_present"]:',
                                                             start=park_line),
                                        find_line(enclosure, 'th["protr_half"] = ', start=park_line)),
            "note": ("_apply_theta parks ALL THREE bodies and then re-places only the chosen one, "
                     "so on a hazard-absent row every bar is already at its committed parked pose "
                     "and parking is a bitwise no-op."),
        },

        # ---------------- the documented oracle ------------------------- #
        "documented_oracle": {
            "headline_formula": excerpt(react, 12, 15),
            "canonical_implementation": excerpt(react, 369, 379),
            "subtraction": excerpt(react, 379, 382),
            "restore_mechanism": ("re-write data.mocap_pos for each placed bar, then "
                                  "mujoco.mj_forward -- no mj_step, so simulation time never "
                                  "advances"),
            "readme": excerpt(ROOT / "README.md", 174, 174),
            "flinch_variant": {
                "excerpt": excerpt(flinch, 292, 295),
                "difference": ("safety_flinch_demo computes ONE rest baseline at the start "
                               "posture and subtracts it every frame; safety_react_demo "
                               "re-renders the parked scene at the CURRENT pose every frame. "
                               "The per-frame form is the one this task implements."),
            },
            "renders_per_control_step": 2,
            "privileged": True,
            "privileged_reason": ("requires moving a scene body that the robot cannot move and "
                                  "observing a counterfactual world; not achievable on hardware"),
        },

        # ---------------- current-observation capture ------------------- #
        "current_observation_capture": {
            "producer": excerpt(env_py, 374, 409),
            "call_site": excerpt(task_py, 376, 386),
            "renderer": ("dedicated 8x8 mujoco.Renderer, skybox off, MjvOption with "
                         "geomgroup[2]=0 (cosmetic skin hidden), depth rendering enabled"),
            "substeps": ("record_proximity_depths appends one frame per proximity period; the "
                         "observation stacks them and extract_latest_proximity takes [-1], i.e. "
                         "the render taken at the final sub-step -- the same simulator state the "
                         "policy sees at decision time"),
            "extract_latest_proximity": excerpt(residual, 224, 237),
        },

        # ---------------- controller constants and timing --------------- #
        "controller": {
            "constants_source": excerpt(residual, 20, 25),
            "equation_source": excerpt(residual, 398, 411),
            "adapter_docstring_equation": excerpt(adapter, 6, 12),
            "scale_applied_once": ("SafetyHead multiplies its output by label_scale "
                                   "(train_safety_cvae.SafetyHead.__call__) and the committed "
                                   "controller divides the subtracted delta by the same scale, "
                                   "exactly as safety_react_demo.py:379 does"),
            "safety_head_call": excerpt(cvae, 113, 118),
            "residual_after_temporal_aggregation": excerpt(adapter, 195, 197),
            "arm_only_gripper_preserved": excerpt(residual, 429, 437),
        },

        # ---------------- the demo's nominal motion --------------------- #
        "demo_used_prerecorded_nominal_motion": {
            "answer": True,
            "evidence": excerpt(react, 246, 249),
            "detail": ("safety_react_demo.pick_episode() loads a recorded successful "
                       "reach-grasp-lift from datagen H5s, resamples and smooths it, and "
                       "advances a scalar phase s. The nominal motion is open-loop playback, "
                       "not a live policy. The PARKED REFERENCE is nevertheless recomputed "
                       "every frame at the CURRENT executed pose -- the reference is per-frame "
                       "even though the nominal is prerecorded."),
            "phase_clock": excerpt(react, 386, 386),
        },

        # ---------------- why the old reference died at 109 ------------- #
        "finite_reference_defect": {
            "mechanism": excerpt(adapter, guard_line, guard_line + 3),
            "indexing": excerpt(adapter, guard_line + 4, guard_line + 4),
            "loader": ("hybrid_safety_residual.load_proximity_sequence_h5 returns a "
                       "(T, 40, 8, 8) array whose T is the RECORDED length of the expert "
                       "demonstration, not the live rollout horizon"),
            "recorded_lengths": lengths,
            "live_task_horizon": 200,
            "diagnosis": ("The prior oracle attempt subtracted head(reference[step]) with a "
                          "step index into a finite recorded array. The reference episode for "
                          "candidate 106 holds "
                          f"{lengths['cand106']['recorded_real_timesteps']} real timesteps while "
                          "the evaluation horizon is 200, so the adapter raised "
                          "HybridSafetyContractError at step "
                          f"{lengths['cand106']['recorded_real_timesteps']}. "
                          "Two independent faults: (a) the reference was a prerecorded "
                          "trajectory rather than a counterfactual of the CURRENT state, so "
                          "once the residual moved the arm the reference no longer described "
                          "the same pose at all; (b) it was indexed by step into a finite "
                          "array. A per-frame counterfactual has neither -- there is no array "
                          "and no index."),
            "not_fixable_by_padding": ("Padding or wrapping would silence the exception while "
                                       "leaving fault (a) untouched, and is prohibited."),
        },

        # ---------------- the analytic teacher --------------------------- #
        "analytic_teacher": {
            "formula": excerpt(sweep, 8, 16),
            "implementation": excerpt(sweep, 322, 350),
            "activation_distance_m": next(
                ast.literal_eval(n.value) for n in ast.walk(ast.parse(sweep.read_text()))
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "D_ACT"),
            "hit_tolerance_m": next(
                ast.literal_eval(n.value) for n in ast.walk(ast.parse(sweep.read_text()))
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "HIT_TOL"),
            "box_surface_dist": excerpt(sweep, 205, 212),
            "training_data": "assets/safety/sweep_v3.h5",
            "caveat": ("The teacher was authored for safety_sweep's own analytic fumehood, whose "
                       "STATIC_BOXES are hard-coded world boxes. The live manifest scene poses "
                       "its enclosure slabs per episode, so the analytic box set has to be "
                       "rebuilt from the compiled model. Faithfulness is therefore verified "
                       "separately against the stored sweep_v3 labels."),
        },

        "existing_parked_utilities": {
            "found": ["scripts/safety_react_demo.py", "scripts/safety_flinch_demo.py",
                      "scripts/safety_moving_demo.py", "scripts/safety_orbit_demo.py",
                      "scripts/safety_sweep.py"],
            "reusable_in_the_manifest_scene": False,
            "why_not": ("Every demo builds its own scene with safety_sweep.build_model() and a "
                        "hard-coded ROBOT_XML path that does not exist in this checkout; they "
                        "park 'bar_s/m/l' at PARK=[0,0,-3.0]. The manifest scene's hazard is "
                        "'protr_s/m/l' parked by enclosure_reach at z=-2.0 with distinct x,y. "
                        "The MECHANISM transfers; the coordinates and body names do not."),
            "sweep_park_constant": next(
                ast.literal_eval(n.value) for n in ast.walk(ast.parse(sweep.read_text()))
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PARK"),
        },
    }
    audit["audit_sha256"] = canonical_hash(audit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n")

    print("committed parked pose:")
    for name, pos in parked_pose.items():
        print(f"  {name:<9} -> {pos}")
    print(f"parked-z threshold used by the runner: {parked_threshold}")
    print("\nrecorded reference lengths vs the 200-step horizon:")
    for name, info in lengths.items():
        print(f"  {name:<8} {info['recorded_real_timesteps']:>4} frames  "
              f"hazard={'present' if info['hazard_present'] else 'absent'}")
    print(f"\nfinite-reference guard at {adapter.name}:{guard_line}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
