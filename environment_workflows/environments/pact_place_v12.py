"""Lifecycle registration for the v12 place-corridor environment."""

from environment_workflows.profile import EnvironmentProfile, WorkflowSpec
from environment_workflows.registry import register

PROFILE = register(
    EnvironmentProfile(
        environment_id="pact_place_v12",
        environment_version="pact_place_corridor_v12",
        description="Four-object place corridor with standing kitchen clutter.",
        dataset_schema="pact_place_v12_v1",
        observation_schema="wrist+table RGB, wrist depth, 40 proximity depth sensors",
        action_schema="MolmoSpaces Franka joint action",
        workflows=(
            WorkflowSpec(
                name="collect",
                script="scripts/run_pact_place_v12_collect.py",
                description="Collect contract-screened v12 expert trajectories.",
                required_paths=(
                    "configs/pact_place_v12_clutter.json",
                    "custom_scenes/pact_place_corridor_v12.xml",
                    "custom_scenes/pact_place_corridor_v12_metadata.json",
                    "scripts/pact_place_v12_contract.py",
                    "submodules/molmospaces/molmo_spaces/tasks/pact_place.py",
                ),
            ),
        ),
    )
)
