from pathlib import Path

import pytest

from environment_workflows.profile import EnvironmentProfile, WorkflowSpec
from environment_workflows.registry import get_profile, list_profiles


def test_v12_is_discoverable_without_loading_simulation_dependencies():
    profile = get_profile("pact_place_v12")

    assert profile.environment_version == "pact_place_corridor_v12"
    assert profile.workflow("collect").script == "scripts/run_pact_place_v12_collect.py"
    assert profile in list_profiles()


def test_workflow_reports_missing_checkout_dependencies(tmp_path: Path):
    workflow = WorkflowSpec(
        name="collect",
        script="collect.py",
        description="test",
        required_paths=("present.txt", "missing.txt"),
    )
    (tmp_path / "collect.py").touch()
    (tmp_path / "present.txt").touch()

    assert workflow.missing_paths(tmp_path) == ("missing.txt",)


def test_workflow_builds_command_with_defaults_before_overrides(tmp_path: Path):
    workflow = WorkflowSpec(
        name="train",
        script="train.py",
        description="test",
        default_args=("--epochs", "2000"),
    )

    assert workflow.command(tmp_path, "python", ["--epochs", "2"]) == [
        "python",
        str(tmp_path / "train.py"),
        "--epochs",
        "2000",
        "--epochs",
        "2",
    ]


def test_v5_declares_convert_train_eval_schemas():
    profile = get_profile("pact_place_v5")

    assert profile.capabilities == ("convert", "train", "eval")
    assert profile.dataset_schema == "act_style_hdf5_v1"
    assert profile.workflow("eval").required_arguments == ("--ckpt_dir", "--output_dir")


def test_profile_rejects_duplicate_workflows():
    workflow = WorkflowSpec(name="collect", script="collect.py", description="test")
    with pytest.raises(ValueError, match="duplicate workflow"):
        EnvironmentProfile(
            environment_id="example",
            environment_version="example_v1",
            description="test",
            dataset_schema="dataset_v1",
            observation_schema="observations_v1",
            action_schema="actions_v1",
            workflows=(workflow, workflow),
        )


def test_unknown_environment_error_lists_available_profiles():
    with pytest.raises(KeyError, match="pact_place_v12"):
        get_profile("not_registered")
