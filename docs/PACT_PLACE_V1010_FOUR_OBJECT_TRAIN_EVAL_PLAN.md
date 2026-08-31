# V10.10 — Four-object ACT/PACT retraining and paired evaluation

**Status:** completed 2026-08-31. The registered run finished 80/80 paired
rollouts with zero infrastructure failures. PACT led directionally on the
registered primary endpoint (10/40 versus ACT 7/40 collision-free task
success), but the paired 95% interval crossed zero, so no superiority claim is
made.
**Derived from V10.7. V10.7 is not rewritten or reopened**; its Phase-0 gate
remains failed at 8/24 and permanently closed. Human review and Phase 0 are
skipped here at the owner's explicit request. No result below authorizes
downstream work.

## The four objects

Exactly four household objects are active. The target cup, the static pendant,
the intrusion panel, the tray and the enclosure are **not** among them.

| slot | uid | palette role |
|---|---|---|
| 01 | `Soap_Bottle_30` | outbound vessel |
| 03 | `Plate_10` | decor |
| 04 | `Plate_22` | decor |
| 06 | `Soap_Bottle_11` | inbound vessel |

Parked: 00 `Candle_2`, 02 `Mug_2`, 05 (can), 07 `Candle_1`.

Both route-bearing vessels stay active, so the corridor the task is about is
unchanged; what is removed is decor. All eight assets stay **compiled**, so
observations and checkpoints remain shape-compatible.

## Environment

`PactPlaceCorridorV1010FourObjectSampler` subclasses the V10.6 sampler and
overrides `_layout()` only. The inherited `_apply_theta` already parks any
compiled body whose slot is absent from the layout at its own `park_m`, so
parking introduces no new mechanism.

Reused byte-for-byte from V10.7: the three certified pendant XMLs and their
hashes, routes, speeds, target distribution, four layout families, two
intrusion sides, three pendant poses, cameras, the 40-sensor proximity suite,
and the contact taxonomy.

**A gate that had to be widened.** `_v106_enabled()` controls the V10.6 speed
amendment and the frame telemetry, and `_v9_enabled()` controls expert routing.
Both matched the V10.6 marker exactly, so a new marker would have silently
disabled the speed amendment and emitted no telemetry while appearing healthy.
V10.10 is admitted through `PACT_PLACE_V106_LANE_ENVIRONMENT_VERSIONS`, since it
is the same lane with four slots parked.

Every row binds `active_clutter_slots`, `inactive_clutter_slots`,
`active_clutter_count=4`, the four-object layout hash, the scene hash, the
sampler version and the seed.

## Storage (completed)

Retained artifacts were hashed and matched against the V10.9 close-out **before**
anything was removed; a create-only deletion manifest was written first.
Reclaimed 26.38 GiB → **32.93 GiB free**, zero hash drift across the 12 retained
files. Removed: `pact_place_v109_eval_traj/`, `act_style_data/pact_place_v108_141/`
(regenerable from the datagen rows), and `resume_bundle.ckpt`,
`policy_epoch_*.ckpt`, `policy_last.ckpt` from both training directories.
Retained: `policy_best.ckpt`, statistics, run manifests, epoch logs, timing and
compact evaluation results.

## Streams

Collection 2026101001, split 2026101002, evaluation 2026101003 — all asserted
disjoint from every V10.7–V10.9 stream.

## Preflight, collection, conversion, training

A non-episode preflight over all 24 family×side×pose cells checks: exactly four
active and four parked clutter bodies; correct identities and poses; no initial
robot, target, clutter, panel or pendant contact; stable and contained clutter;
the correct static-pendant scene and live contact parity.

Collect exactly **144** strict-clean expert successes, six per cell, with at most
one in-flight attempt per cell so parallel completion cannot exceed quota.
Capped at 900 attempts or 16 hours; if any cell is short, stop without training.
Accept only task-successful rows with zero disallowed contact and zero
clutter-stability event; record every rejection and its per-object cause.

Convert all 144 rows, preserving the canonical 40-sensor order, and generate
`(T,40,32)` embeddings with the frozen encoder
`6fd2dd037e3236b5b6bf7fce8cb2709ead0cf52adcbbe9cbad1061efc2fe3206`.

Freeze a deterministic **120/24** split: five training and one validation episode
from every cell.

Train ACT then PACT, seed 3101, V10.9/V5 chunk-100 settings, horizons
`max(635, T_max + 8)`. PACT differs only by its five proximity flags. After each
run: all epochs, strict best-checkpoint reload, finite/non-constant `(B,100,8)`
output, and for PACT a causal action change between real and zeroed embeddings.

## Paired evaluation

The V10.9 evaluator is adapted to the new sampler and four-object contract, with
fallback to a V2 or V10.7 sampler/scene identity explicitly refused. The
**original chunk-100 temporal ensemble and gripper decoder** are used; the V10.9R
event decoder is not.

Four-instance smoke, then 40 fresh paired instances (80 rollouts) with identical
initial-state hashes per arm. V10.9 balance preserved: all 24 cells present, 10
per family, 20 per side, poses 14/13/13. `num_queries=100`, horizon 900,
`end_on_success=false`, no action noise. No started scientific row is replaced or
reseeded.

Every trajectory and raw action chunk is retained. The **primary endpoint is
collision-free task success**, matching the V5 chunk-100 reporting. Also
reported: task success and strict-clean success; contacts and stability per
bottle and per plate; panel, pendant,
other-environment and initial-state contacts; the touch→hold→success funnel and
gripper timing; results by family, side, pose and cell; and paired PACT−ACT
differences with 95% intervals, discordant counts and exact McNemar.

**Superiority is claimed only if the primary interval excludes zero in PACT's
favour.** Otherwise the result is reported as descriptive, null or negative,
without tuning or rerunning.

## Stop rules

Failed preflight, insufficient storage, incomplete 144-row balance, corrupt
conversion, checkpoint reload failure, or evaluation infrastructure failure.
**Poor model performance is a result, not a reason to modify or repeat the
registered run.**

## Close-out

The four-object collection reached its frozen quota: 144/144 strict-clean
demonstrations, six in every family×side×pose cell, from 313 attempts in 6.50
hours. The deterministic split is 120 train / 24 validation, with exactly five
and one row per cell. ACT and PACT both completed 2,000 epochs and passed strict
checkpoint reload and offline smoke checks.

The repaired paired evaluation completed 40 shared instances (80 rollouts) in
5.746 hours. ACT/PACT task success was 11/40 versus 14/40; the registered
collision-free task-success endpoint was 7/40 versus 10/40, a PACT−ACT
difference of +7.5 percentage points with paired 95% CI [−7.02, +22.02] and
exact McNemar p=0.5078. This is directional evidence only. The registered rule
therefore records a null/inconclusive superiority result, without retuning or a
scientific retry.
