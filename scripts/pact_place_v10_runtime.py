#!/usr/bin/env python3
"""V10 runner helpers: local asset env and canonical hashing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pact_place_corridor_contract import sha256_payload

ROOT = Path(__file__).resolve().parents[1]


def establish_v10_runtime_env() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(ROOT / "assets"))
    os.environ.pop("DISPLAY", None)


def canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        items = [canonicalize(item) for item in value]
        if items and isinstance(items[0], dict):
            if "assembly_id" in items[0]:
                items.sort(key=lambda item: str(item.get("assembly_id")))
            elif "key" in items[0]:
                items.sort(key=lambda item: str(item.get("key")))
            elif "role_index" in items[0]:
                items.sort(key=lambda item: int(item.get("role_index", 0)))
        return items
    return value


def write_immutable(path: Path, document: dict[str, Any]) -> str:
    payload = canonicalize(dict(document))
    payload.pop("artifact_sha256", None)
    digest = sha256_payload(payload)
    payload["artifact_sha256"] = digest
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return digest
