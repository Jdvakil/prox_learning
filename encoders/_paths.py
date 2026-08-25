"""sys.path inserts so this folder can import ACT siblings without install.

Repo is not an installed package (README §3). Scripts add paths by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ACT_DIR = REPO_ROOT / "submodules" / "act"


def ensure_act_on_path() -> None:
    path = str(ACT_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def ensure_repo_on_path() -> None:
    path = str(REPO_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
