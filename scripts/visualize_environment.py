#!/usr/bin/env python3
"""Render MolmoSpaces envs using XMLs in this repo's custom_scenes/.

Eval and this script share ``custom_scenes/pact_place_corridor_v10_7_*.xml``
(plus the v5→v3 include chain). Sampler Python still comes from molmospaces
origin/main (``MOLMOSPACES_PACT_V1010`` / default worktree).

No args: sample the v1011d / V10.10 four-object place env (house 1 = F0 left
center) and write PNGs under experiments_output/default/environment_viz/.

    python scripts/visualize_environment.py
    python scripts/visualize_environment.py --list --scope project
    python scripts/visualize_environment.py --show-hidden --format both
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUSTOM_SCENES = ROOT / "custom_scenes"
DEFAULT_WORKTREE = Path(
    os.environ.get("MOLMOSPACES_PACT_V1010", "/home/jaydv/code/molmospaces-pact-v1010")
)
DEFAULT_CONFIG = (
    "molmo_spaces.data_generation.config.pact_place_datagen_configs:"
    "FrankaSkinPactPlaceV1010FourObjectConfig"
)
ACT_DIR = ROOT / "submodules" / "act"

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))

if str(ACT_DIR) not in sys.path:
    sys.path.insert(0, str(ACT_DIR))
from eval_place_v1010_scene import (  # noqa: E402
    MOLMOSPACES_V1010_SHA,
    resolve_v1010_scenes_dir,
)


def _molmospaces_root() -> Path:
    root = DEFAULT_WORKTREE
    if not (root / "molmo_spaces").is_dir():
        raise SystemExit(
            f"[visualize-env] molmospaces worktree missing at {root}.\n"
            "  Sampler class lives there. XMLs are already in custom_scenes/.\n"
            "  git -C /home/jaydv/code/prox_learning/submodules/molmospaces "
            f"worktree add {root} {MOLMOSPACES_V1010_SHA}"
        )
    return root.resolve()


def _inject_default_config(argv: list[str]) -> list[str]:
    if any(a in argv for a in ("-h", "--help", "--list", "--all")):
        return argv
    if any(("Config" in a) or (":" in a and not a.startswith("-")) for a in argv):
        return argv
    extra = [DEFAULT_CONFIG]
    if "--house" not in argv:
        extra += ["--house", "1"]
    if "--show-hidden" not in argv:
        extra += ["--show-hidden"]
    return extra + argv


def _remap_scene_xml_paths(config) -> None:
    scenes_dir = resolve_v1010_scenes_dir(_molmospaces_root())
    sampler = getattr(config, "task_sampler_config", None)
    paths = getattr(sampler, "scene_xml_paths", None) if sampler is not None else None
    if not paths:
        return
    remapped = []
    for raw in paths:
        name = Path(raw).name
        local = scenes_dir / name
        remapped.append(str(local if local.is_file() else Path(raw)))
    sampler.scene_xml_paths = remapped


def _load_viz(molmo_root: Path):
    viz_path = molmo_root / "scripts" / "datagen" / "visualize_environment.py"
    if not viz_path.is_file():
        viz_path = (
            ROOT / "submodules" / "molmospaces" / "scripts" / "datagen" / "visualize_environment.py"
        )
    if str(molmo_root) not in sys.path:
        sys.path.insert(0, str(molmo_root))
    import importlib.util

    spec = importlib.util.spec_from_file_location("molmo_visualize_environment", viz_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[visualize-env] cannot load {viz_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    molmo_root = _molmospaces_root()
    sys.argv = [sys.argv[0], *_inject_default_config(sys.argv[1:])]
    viz = _load_viz(molmo_root)
    viz.PROJECT_CONFIG_RE = re.compile(
        r"^FrankaSkin("
        r"Cabinet|Shelf|Clutter|Pillar|RealTable|RealHouse|Enclosure|"
        r"Fumehood|Panel|Cubby|House|Hybrid|ProxNecessity|PactPlace"
        r")"
    )
    orig_prepare = viz._prepare_config

    def _prepare_config(config, args):
        _remap_scene_xml_paths(config)
        orig_prepare(config, args)
        print(
            f"[visualize-env] scenes_dir={resolve_v1010_scenes_dir(molmo_root)} "
            f"custom_scenes={CUSTOM_SCENES}",
            flush=True,
        )

    viz._prepare_config = _prepare_config

    orig_collect = viz._collect_environment_groups

    def _collect_environment_groups(scope, force_hybrid=True):
        groups, skipped = orig_collect(scope, force_hybrid=force_hybrid)
        scenes_dir = resolve_v1010_scenes_dir(molmo_root)
        for group in groups:
            src = str(group.get("source") or "")
            local = scenes_dir / Path(src).name
            if src.endswith(".xml") and local.is_file():
                group["source"] = str(local)
        return groups, skipped

    viz._collect_environment_groups = _collect_environment_groups
    viz.main()


if __name__ == "__main__":
    main()
