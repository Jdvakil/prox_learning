# `docs/` — project documentation

Read these in order if you are new to the project. Reread them at the
start of each work day.

## Live tracking docs (updated as the project evolves)

| file                                                  | what's in it                                                          |
|-------------------------------------------------------|-----------------------------------------------------------------------|
| [STATUS.md](STATUS.md)                                | live milestones, day-by-day progress, risk register                   |
| [CHANGELOG.md](CHANGELOG.md)                          | user-facing summary of changes per day                                |
| [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md)        | append-only chronicle of *what* was built, *when*, and *why*          |
| [SANITY_CHECKS.md](SANITY_CHECKS.md)                  | every sanity-check command + verbatim output (Day 1 + future)         |

## Reference docs (the "why" — change rarely)

| file                                                  | what's in it                                                          |
|-------------------------------------------------------|-----------------------------------------------------------------------|
| [PROJECT.md](PROJECT.md)                              | full project brief, scientific claim, references                      |
| [TIMELINE.md](TIMELINE.md)                            | 26-day execution plan + risk register                                 |
| [ARCHITECTURE.md](ARCHITECTURE.md)                    | tensor shapes, module contracts, invariants                           |
| [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)            | every non-obvious choice with alternatives + rationale                |
| [STATISTICAL_PROTOCOL.md](STATISTICAL_PROTOCOL.md)    | exact eval methodology — bootstrap, pairing, multiple comparison       |
| [FILE_INVENTORY.md](FILE_INVENTORY.md)                | every file in the repo: purpose, LOC, where it's documented           |

## Legacy docs (historical, kept for context)

| file                                                  | what's in it                                                          |
|-------------------------------------------------------|-----------------------------------------------------------------------|
| [DATASET.md](DATASET.md)                              | `skin_pick_fixed_v1` statistics (legacy 10-traj sanity set)            |
| [SKIN_PIPELINE.md](SKIN_PIPELINE.md)                  | URDF→MJCF skin pipeline reference                                      |
| [CVAE.md](CVAE.md)                                    | Skin-proximity CVAE notes (encoder-pretrain proof)                     |

## Reading order

For a new collaborator:

1. **[PROJECT.md](PROJECT.md)** — what we're proving and why.
2. **[TIMELINE.md](TIMELINE.md)** — when things should happen.
3. **[STATUS.md](STATUS.md)** — what's done so far.
4. **[ARCHITECTURE.md](ARCHITECTURE.md)** — what the code looks like.
5. **`pla/README.md`** + the README in the subfolder you're touching.
6. **[DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)** — why we chose those shapes.
7. **[SANITY_CHECKS.md](SANITY_CHECKS.md)** — the contract you must not break.

For someone *running* the project (not editing it), STATUS.md +
[scripts/README.md](../scripts/README.md) is enough.

## Doc maintenance rules

- **STATUS.md, IMPLEMENTATION_LOG.md, SANITY_CHECKS.md** — update on every
  meaningful change. Append to IMPLEMENTATION_LOG / SANITY_CHECKS; revise
  STATUS in place.
- **ARCHITECTURE.md, DESIGN_DECISIONS.md** — update only when a
  contract or decision actually changes. Don't fight the file's history.
- **FILE_INVENTORY.md** — refresh whenever a new file lands in `pla/`,
  `configs/`, or `scripts/`.
- **Per-folder READMEs** — update when the API of that subpackage changes.
- **Legacy docs** — leave as-is; they document Day 0 state.
