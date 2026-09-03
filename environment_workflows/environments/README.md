# Adding an environment workflow

Create one module in this directory and register one `EnvironmentProfile`.
The profile is the executable contract tying an environment version to its
dataset, observations, actions, and supported lifecycle commands.

For an end-to-end environment, register:

1. `collect`: deterministic seeds, frozen environment contract, resumable
   ledger, and acceptance telemetry.
2. `convert`: raw-row validation and conversion into the declared training
   dataset schema. Omit this only when collection writes that schema directly.
3. `train`: the matching task configuration and observation/action layout.
4. `eval`: the same sampler, scene version, observations, and action contract
   used by collection and training.

Each `WorkflowSpec` must list every file or directory it needs. The launcher
checks those paths before importing simulation or training code, so an
incomplete checkout fails immediately instead of producing an empty dataset or
evaluating in the wrong environment.

Profiles must not modify global behavior when imported. Keep expensive imports,
GPU initialization, dataset downloads, and execution inside the referenced
workflow scripts.
