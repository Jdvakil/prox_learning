# PACT versus ACT remediation-v2 preregistration

## Scope and isolation

This is a new experiment, not a reinterpretation of the 24-row v1 pilot. The
v1 decision remains valid under its frozen gate. No v1 rollout is loaded,
rescored, pooled, replaced, or used to select a v2 row.

The remediation runs in `/root/prox_learning_pact_remediation` on root branch
`experiment/pact-vs-act-remediation-v2`, with independent ACT and MolmoSpaces
git directories. The shared confirmatory checkout and its submodule checkout
state are read-only and out of scope. The corridor scene remains
`pact_collision_corridor_v1`; its XML, panel geometry, aperture, target
distribution, exact within-role side balance, and target/side independence are
unchanged.

The only expert changes are:

- nominal surface clearance raised from 0.08 m to 0.10 m;
- corridor-expert mid-rollout replans disabled.

Initial task/trajectory construction can still use the four deterministic
pre-action retry seeds. Once the initial reset is accepted, a planner failure
is an ordinary retained task failure rather than a replan or infrastructure
exception.

## Frozen candidate population

The machine-readable preregistration is
`configs/pact_collision_environment_v2.json`. The independently seeded
candidate population is
`configs/pact_collision_candidate_manifest_v2.json`, SHA-256
`cba7ff8879565e72eb667da3419267104dc9c3141a459de78d635fe760a41784`.
Its master seed is `2026072901`.

| Role | Fixed attempts |
|---|---:|
| Expert development | 32 |
| Pilot expert/train | 64 |
| Pilot ACT evaluation | 64 |
| Full train | 240 |
| Full validation | 64 |
| Confirmatory instances | 160 |

All rows are terminal ledger entries. No row is replaced or rerun based on
contact, task outcome, policy outcome, or process outcome.

## Demonstrator and infrastructure rules

A usable demonstration is defined before collection as ordinary task success
with zero `hazard_bar` and zero `other_environment` contact. Target contact is
exempt. All 64 pilot expert attempts remain in provenance; every usable row is
included in frozen role order. At least 48 usable demonstrations are required
to train pilot ACT. The attempted-row clean rate and Wilson 95% interval are
reported, but the rate itself is not an environment gate.

Development is ready at 24 usable clean demonstrations out of 32 fixed
attempts. The no-scientific-outcome rate—`sampling_failure` plus
`infrastructure_failure`—must be strictly below 5% before scale. Its Wilson
interval is reported. Missing scientific outcomes never count as policy
failures and can never award `PACT_ENVIRONMENT_INADEQUATE`; a missed
progression criterion yields `PACT_EXPERIMENT_INCOMPLETE`.

After an adequate pilot, the fixed full roles must yield at least 180 usable
clean demonstrations from 240 `full_train` attempts and at least 48 from 64
`full_validation` attempts. A shortfall yields
`PACT_EXPERIMENT_INCOMPLETE`; no attempt is replaced.

## Environment science gates

Only the following three properties can establish that the environment is
inadequate.

Surface observability keeps the v1 thresholds: active panel signal in at least
30% of pre-grasp frames inside 20 cm, at least 5% inside 12 cm, and an active
episode fraction of at least 5/6. Signal is panel-specific: the reconstructed
ray must enter the active panel AABB with 1 cm tolerance. Frame fractions use
20,000 deterministic whole-episode bootstrap replicates; active episodes use a
Wilson 95% interval. A pass must survive deletion of every one episode. A
failure requires a 95% upper bound below a threshold; a marginal result is
inconclusive.

Gate B uses collision-free task success from the 64 fixed wrist-RGB-plus-qpos
ACT evaluations. The target point band is `[1/3, 2/3]`, preserving the v1
`8–16/24` intent. A pass also requires the Wilson 95% interval to lie inside
`[0.20, 0.80]`. A robust low or high rate is an environment failure. Ordinary
task success is secondary.

Gate C requires at least 25% of scientific ACT rows to contact the intrusion
and a Wilson lower bound above 10%. `other_environment` contact is reported but
does not substitute for panel contact.

Gate B and C labels must be unchanged when one binary outcome is changed in
either direction. Otherwise the gate is inconclusive. At least 61 of the 64
ACT rows must have scientific outcomes.

The decision mapping is frozen:

- all three science gates robustly pass, at least 48 demos are usable, at least
  61 ACT rows are scientific, and infrastructure progression passes:
  `PACT_ENVIRONMENT_ADEQUATE`;
- a robust surface, Gate B, or Gate C failure:
  `PACT_ENVIRONMENT_INADEQUATE`;
- any marginal gate, demo shortfall, infrastructure shortfall, or
  unreconciled artifact: `PACT_EXPERIMENT_INCOMPLETE`.

## Execution hold

At preregistration time,
`pgrep -fc eval_act_obstacle_on_policy.py` returned 16. No simulator
collection, training, or evaluation may launch until it returns zero.

The static MolmoSpaces object/LMDB cache is read from the handoff path
`/root/prox_learning_hybrid_safety/assets`. The collection runner asserts that
its output directory resolves inside the isolated remediation worktree and
records both resolved paths per row; no PACT output is written to the shared
asset tree.
