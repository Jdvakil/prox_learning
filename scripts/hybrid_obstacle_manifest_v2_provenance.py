#!/usr/bin/env python3
"""Record the exact runtime the manifest-v2 smoke runs executed on.

Determinism is claimed for THIS runtime only. No reproducibility claim is made
across different software versions or hardware from this smoke.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submodules" / "molmospaces"))

COLLECTION_SOURCES_MS = [
    "molmo_spaces/data_generation/episode_manifest.py",
    "molmo_spaces/data_generation/row_ledger.py",
    "molmo_spaces/data_generation/manifest_runner.py",
    "molmo_spaces/data_generation/pipeline.py",
    "molmo_spaces/data_generation/worker_completeness.py",
    "molmo_spaces/data_generation/runtime_compat.py",
    "molmo_spaces/tasks/enclosure_reach.py",
    "molmo_spaces/tasks/task_sampler.py",
    "molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py",
]
COLLECTION_SOURCES_ROOT = [
    "scripts/run_hybrid_obstacle_manifest_v2.py",
    "configs/hybrid_obstacle_candidate_manifest_v2.json",
    "configs/hybrid_obstacle_manifest_v2_smoke8.json",
]


def sh(command: list[str], cwd: Path = ROOT) -> str:
    try:
        return subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        return "unavailable"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except ImportError:
        return "not installed"
    value = getattr(module, "__version__", None)
    if value is None:
        value = getattr(getattr(module, "config", None), "version", None)
    return str(value) if value else "unknown"


def main() -> int:
    import torch

    molmospaces = ROOT / "submodules" / "molmospaces"
    act = ROOT / "submodules" / "act"

    collection_hashes = {
        f"molmospaces/{name}": sha256_file(molmospaces / name)
        for name in COLLECTION_SOURCES_MS
    }
    collection_hashes.update(
        {f"root/{name}": sha256_file(ROOT / name) for name in COLLECTION_SOURCES_ROOT}
    )
    collection_digest = hashlib.sha256(
        json.dumps(collection_hashes, sort_keys=True).encode()
    ).hexdigest()

    payload = {
        "schema": "hybrid_obstacle_manifest_v2_runtime_provenance",
        "scope": (
            "The tested contract is for this recorded runtime only. No claim is made "
            "about reproducibility across different software versions or hardware."
        ),
        "operating_system": platform.platform(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy": version("numpy"),
        "torch": version("torch"),
        "mujoco": version("mujoco"),
        "warp": version("warp"),
        "scipy": version("scipy"),
        "h5py": version("h5py"),
        "cuda_toolkit_torch_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_driver": sh(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ).splitlines()[:1],
        "gpu_model": sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines(),
        "mujoco_gl": os.environ.get("MUJOCO_GL", "unset"),
        "root_commit": sh(["git", "rev-parse", "HEAD"]),
        "root_branch": sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "molmospaces_commit": sh(["git", "rev-parse", "HEAD"], molmospaces),
        "molmospaces_branch": sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], molmospaces),
        "act_gitlink": sh(["git", "rev-parse", "HEAD"], act),
        "act_branch": sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], act),
        "act_modified_by_this_task": False,
        "model_hybrid_xml_sha256": sha256_file(
            ROOT / "assets" / "robots" / "franka_skin" / "model_hybrid.xml"
        ),
        "fumehood_scene_sha256": sha256_file(
            molmospaces / "molmo_spaces" / "data_generation" / "custom_scenes" / "fumehood.xml"
        ),
        "collection_source_hashes": collection_hashes,
        "collection_source_digest_sha256": collection_digest,
    }

    manifest = json.loads((ROOT / "configs" / "hybrid_obstacle_candidate_manifest_v2.json").read_text())
    smoke = json.loads((ROOT / "configs" / "hybrid_obstacle_manifest_v2_smoke8.json").read_text())
    payload["sensor_order_sha256"] = manifest["sensor_order_sha256"]
    payload["runtime_contract_sha256"] = manifest["runtime_contract_sha256"]
    payload["env_config_sha256"] = manifest["env_config_sha256"]
    payload["manifest_sha256"] = manifest["manifest_sha256"]
    payload["smoke8_sha256"] = smoke["subset_sha256"]

    out = ROOT / "diagnostics_output" / "hybrid_obstacle_seeding" / "runtime_provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
    print(f"  collection_source_digest {collection_digest}")
    for key in ("python", "numpy", "torch", "mujoco", "warp", "scipy", "h5py"):
        print(f"  {key:8s} {payload[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
