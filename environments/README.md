# Adding an environment

Each directory here reproduces one published dataset split. Adding another one
is a directory, not an edit to a shared table: `environments/__init__.py`
discovers subpackages at call time and picks up any `SPEC` it finds, so
`scripts/verify_hf_env.py` and `docs/DATASETS.md` pick the new environment up
for free.

## Steps

1. Create `environments/hf_<split>/` named for the hub folder it reproduces.
2. Write `__init__.py` with a single `SPEC = EnvSpec(...)`. Copy the fields from
   a neighbour and read every value off the collect code that produced the dump,
   not off the hub metadata — the two have disagreed before.
3. Copy `collect.py` verbatim from a neighbour. It derives its own package name,
   so it needs no edits.
4. Vendor whatever a clean checkout is missing:
   - the collect entrypoint and any sibling modules it imports, into `scripts/`
   - scene XML into `custom_scenes/`, keeping the bytes identical so the sha256
     recorded in published rows still matches
   - gate artifacts into `diagnostics_output/`, listed in `required_artifacts`
5. Run `python scripts/verify_hf_env.py --split <split> --online`.
6. Smoke the collect until it writes one `result.json`, then add a row to
   `docs/DATASETS.md`.

## Field notes

`environment_version`, `sampler_class`, `policy_class` and `schema_version` are
history: they were recorded when the data was collected and must never be
edited to match a later refactor. If the hub metadata disagrees, keep the code's
value and say so in `notes`.

`required_artifacts` exists because a missing gate artifact otherwise surfaces
as a bare `FileNotFoundError` from deep inside preflight, long after the run
looked healthy. List every file the collect binds before it rolls out.

`molmospaces_commit` is per environment. The repo has only one submodule pin, so
if a new environment needs a different commit, `verify_hf_env.py` will report
the conflict rather than let it pass silently. Resolve it by confirming the
classes the two environments rely on are identical across the commits, the way
`hf_v12` and `hf_v1011d` share one pin today.
