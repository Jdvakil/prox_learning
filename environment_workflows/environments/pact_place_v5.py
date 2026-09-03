"""Lifecycle registration for the existing ACT-ready v5 corridor."""

from environment_workflows.profile import EnvironmentProfile, WorkflowSpec
from environment_workflows.registry import register

PROFILE = register(
    EnvironmentProfile(
        environment_id="pact_place_v5",
        environment_version="pact_place_corridor_v2",
        description="Existing 152-episode hallway dataset and ACT/PACT evaluation environment.",
        dataset_schema="act_style_hdf5_v1",
        observation_schema="wrist RGB with optional 40-sensor proximity history",
        action_schema="Franka qpos action sequence",
        workflows=(
            WorkflowSpec(
                name="convert",
                script="scripts/convert_pact_place_to_act.py",
                description="Convert accepted raw rows to ACT-style HDF5 episodes.",
                required_paths=("data/pact_place_corridor_v5",),
                default_args=(
                    "--src",
                    "data/pact_place_corridor_v5",
                    "--dst",
                    "act_style_data/pact_place_corridor_v5",
                    "--with-proximity",
                    "--prox-pool",
                    "min",
                    "--image-h",
                    "240",
                    "--image-w",
                    "320",
                ),
            ),
            WorkflowSpec(
                name="train",
                script="submodules/act/imitate_episodes.py",
                description="Train ACT or PACT against the converted v5 dataset.",
                required_paths=(
                    "act_style_data/pact_place_corridor_v5",
                    "submodules/act/constants.py",
                ),
                working_directory="submodules/act",
                default_args=(
                    "--task_name",
                    "pact_place_corridor_v5",
                    "--policy_class",
                    "ACT",
                    "--ckpt_dir",
                    "ckpts",
                    "--kl_weight",
                    "10",
                    "--chunk_size",
                    "50",
                    "--hidden_dim",
                    "512",
                    "--dim_feedforward",
                    "3200",
                    "--batch_size",
                    "8",
                    "--lr",
                    "1e-5",
                    "--seed",
                    "0",
                    "--num_epochs",
                    "2000",
                ),
                required_arguments=("--wandb_run_name",),
            ),
            WorkflowSpec(
                name="eval",
                script="submodules/act/eval_act_place_corridor.py",
                description="Evaluate a trained checkpoint in the matching v5 corridor.",
                required_paths=("submodules/molmospaces",),
                working_directory="submodules/act",
                default_args=(
                    "--num_rollouts",
                    "50",
                    "--chunk_size",
                    "50",
                    "--temp_agg_off",
                    "--task_horizon",
                    "800",
                ),
                required_arguments=("--ckpt_dir", "--output_dir"),
            ),
        ),
    )
)
