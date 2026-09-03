#!/usr/bin/env python3
"""Discover and run collect, convert, train and eval environment workflows."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from environment_workflows.registry import get_profile, list_profiles  # noqa: E402


def _status(profile, workflow: str) -> str:
    missing = profile.missing_paths(ROOT, workflow)
    return "ready" if not missing else f"missing {len(missing)} required path(s)"


def _print_profiles() -> None:
    for profile in list_profiles():
        capabilities = ", ".join(
            f"{name}:{_status(profile, name)}" for name in profile.capabilities
        )
        print(
            f"{profile.environment_id:24} {profile.environment_version}\n"
            f"  {capabilities}"
        )


def _describe(environment_id: str) -> int:
    profile = get_profile(environment_id)
    print(f"id: {profile.environment_id}")
    print(f"version: {profile.environment_version}")
    print(f"dataset_schema: {profile.dataset_schema}")
    print(f"observation_schema: {profile.observation_schema}")
    print(f"action_schema: {profile.action_schema}")
    print(f"description: {profile.description}")
    for workflow in profile.workflows:
        print(f"{workflow.name}: {_status(profile, workflow.name)}")
        print(f"  {workflow.description}")
        if workflow.required_arguments:
            print(f"  required arguments: {', '.join(workflow.required_arguments)}")
        for path in workflow.missing_paths(ROOT):
            print(f"  missing: {path}")
    return 0


def _has_argument(arguments: list[str], name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in arguments)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list registered environments")
    parser.add_argument("--describe", metavar="ENV", help="show one environment profile")
    parser.add_argument("--env", metavar="ENV", help="run a workflow for this environment")
    parser.add_argument(
        "--action",
        default="collect",
        help="registered workflow to run (default: collect)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the command without executing it",
    )
    args, workflow_args = parser.parse_known_args()

    if args.list:
        _print_profiles()
        return 0
    if args.describe:
        return _describe(args.describe)
    if not args.env:
        parser.error("choose --list, --describe ENV, or --env ENV")

    profile = get_profile(args.env)
    workflow = profile.workflow(args.action)
    missing = workflow.missing_paths(ROOT)
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(
            f"{profile.environment_id} {args.action} is unavailable from this "
            f"checkout; missing:\n{paths}"
        )

    if workflow_args[:1] == ["--"]:
        workflow_args = workflow_args[1:]
    all_arguments = [*workflow.default_args, *workflow_args]
    absent_arguments = [
        name for name in workflow.required_arguments if not _has_argument(all_arguments, name)
    ]
    if absent_arguments:
        parser.error(
            f"{profile.environment_id} {args.action} requires: "
            f"{', '.join(absent_arguments)}"
        )

    command = workflow.command(ROOT, sys.executable, workflow_args)
    working_directory = ROOT / workflow.working_directory
    if args.dry_run:
        print(f"cwd: {working_directory}")
        print(shlex.join(command))
        return 0

    environment = os.environ.copy()
    python_paths = (ROOT, ROOT / "scripts", ROOT / "submodules" / "molmospaces")
    environment["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in python_paths), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        command,
        cwd=working_directory,
        env=environment,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
