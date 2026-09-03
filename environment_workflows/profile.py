"""Declarative contracts for environment lifecycle workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkflowSpec:
    """One executable phase in an environment's lifecycle."""

    name: str
    script: str
    description: str
    required_paths: tuple[str, ...] = ()
    working_directory: str = "."
    default_args: tuple[str, ...] = ()
    required_arguments: tuple[str, ...] = ()

    def missing_paths(self, root: Path) -> tuple[str, ...]:
        required = (self.script, *self.required_paths)
        return tuple(path for path in required if not (root / path).exists())

    def command(self, root: Path, python: str, extra_args: list[str]) -> list[str]:
        return [python, str(root / self.script), *self.default_args, *extra_args]


@dataclass(frozen=True)
class EnvironmentProfile:
    """Versioned collect/convert/train/evaluate contract for one environment."""

    environment_id: str
    environment_version: str
    description: str
    dataset_schema: str
    observation_schema: str
    action_schema: str
    workflows: tuple[WorkflowSpec, ...]

    def __post_init__(self) -> None:
        names = self.capabilities
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate workflow in {self.environment_id!r}: {names}")

    def workflow(self, name: str) -> WorkflowSpec:
        for workflow in self.workflows:
            if workflow.name == name:
                return workflow
        supported = ", ".join(item.name for item in self.workflows) or "(none)"
        raise KeyError(
            f"{self.environment_id!r} does not support {name!r}; supported: {supported}"
        )

    @property
    def capabilities(self) -> tuple[str, ...]:
        return tuple(workflow.name for workflow in self.workflows)

    def missing_paths(self, root: Path, workflow: str) -> tuple[str, ...]:
        return self.workflow(workflow).missing_paths(root)
