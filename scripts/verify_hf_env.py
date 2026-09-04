#!/usr/bin/env python3
"""Check this checkout can reproduce a published Hugging Face split.

    python scripts/verify_hf_env.py                 # every environment, offline
    python scripts/verify_hf_env.py --split v1011d  # one environment
    python scripts/verify_hf_env.py --online        # also read the hub manifest

Offline it confirms the submodule pin, that the sampler and policy classes
import, and that the scenes and gate artifacts are present. ``--online``
additionally reads the published manifest so a mismatch between the repo and
the data is caught before anyone spends GPU time on an evaluation.

Environments are discovered from ``environments/``; this script needs no edit
when one is added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environments import HF_DATASET, EnvSpec, all_specs  # noqa: E402

MANIFEST_URL = "https://huggingface.co/datasets/{ds}/resolve/main/{path}/manifest.json"


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {label}{f' — {detail}' if detail else ''}")
        if not ok:
            self.failures.append(label)

    def skip(self, label: str, detail: str) -> None:
        print(f"  [skip] {label} — {detail}")


def submodule_pin() -> str | None:
    out = subprocess.run(
        ["git", "ls-tree", "HEAD", "submodules/molmospaces"],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout.split()
    return out[2] if len(out) >= 3 else None


def check_pin(report: Report, specs: dict[str, EnvSpec]) -> None:
    """The repo has one pin; every environment must agree with it."""
    wanted = {spec.molmospaces_commit for spec in specs.values()}
    if len(wanted) > 1:
        report.check(
            False,
            "molmospaces pin",
            "environments disagree: "
            + ", ".join(f"{s.hub_split}={s.molmospaces_commit[:12]}" for s in specs.values()),
        )
        return

    expected = next(iter(wanted))
    pin = submodule_pin()
    if pin is None:
        report.skip("molmospaces pin", "not a git checkout")
        return
    report.check(
        pin == expected,
        "molmospaces pin",
        pin[:12] if pin == expected else f"{pin[:12]} (expected {expected[:12]})",
    )


def check_entrypoint(report: Report, spec: EnvSpec) -> None:
    report.check(
        (ROOT / spec.collect_entrypoint).is_file(), f"entrypoint {spec.collect_entrypoint}"
    )


def check_scenes(report: Report, spec: EnvSpec) -> None:
    for rel, path in zip(spec.scene_relative, spec.scene_paths):
        if not path.is_file():
            report.check(False, f"scene {rel}", "missing")
            continue
        report.check(True, f"scene {rel}", hashlib.sha256(path.read_bytes()).hexdigest()[:12])


def check_artifacts(report: Report, spec: EnvSpec) -> None:
    for rel, path in zip(spec.required_artifacts, spec.artifact_paths):
        report.check(path.is_file(), f"artifact {rel}")


def check_classes(report: Report, spec: EnvSpec) -> None:
    sys.path.insert(0, str(ROOT / "submodules" / "molmospaces"))
    try:
        from molmo_spaces.tasks import enclosure_reach
    except Exception as exc:  # pragma: no cover - environment dependent
        report.check(False, "import molmo_spaces.tasks.enclosure_reach", repr(exc))
        return
    for name in (spec.sampler_class, spec.policy_class):
        report.check(hasattr(enclosure_reach, name), f"class {name}")


def check_hub(report: Report, spec: EnvSpec) -> None:
    url = MANIFEST_URL.format(ds=HF_DATASET, path=spec.hub_path)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            manifest = json.load(response)
    except Exception as exc:  # pragma: no cover - network dependent
        report.check(False, f"fetch {spec.hub_path}/manifest.json", repr(exc))
        return
    report.check(
        manifest.get("schema_version") == spec.schema_version,
        "hub schema_version",
        f"{manifest.get('schema_version')} vs {spec.schema_version}",
    )
    rows = manifest.get("rows", [])
    report.check(len(rows) == spec.n_episodes, "hub episode count", f"{len(rows)} vs {spec.n_episodes}")
    hub_env = manifest.get("environment_version")
    if hub_env and hub_env != spec.environment_version:
        print(
            f"  [note] hub manifest says environment_version={hub_env!r}; the collect code "
            f"recorded {spec.environment_version!r}. The code is authoritative."
        )


def main() -> int:
    specs = all_specs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(specs), action="append")
    parser.add_argument("--online", action="store_true", help="also read the published manifest")
    args = parser.parse_args()

    report = Report()
    print(f"repo: {ROOT}")
    check_pin(report, specs)

    for name in args.split or sorted(specs):
        spec = specs[name]
        print(f"\n{spec.hub_path}  ->  {spec.environment_version}")
        check_entrypoint(report, spec)
        check_scenes(report, spec)
        check_artifacts(report, spec)
        check_classes(report, spec)
        if args.online:
            check_hub(report, spec)

    print()
    if report.failures:
        print(f"{len(report.failures)} check(s) failed: {', '.join(report.failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
