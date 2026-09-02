# Superseded V10.10 collection, attempt 01

Retained, not deleted. 32 attempts, **0 accepted**, stopped deliberately rather
than allowed to exhaust the 900-attempt budget. Its row directories were removed
because every attempt ran the wrong environment and none is scientifically
usable.

## Cause

`run_pact_place_expert_screen._make_config` resolves the sampler through an
explicit `if/elif` chain of known class names, and sets the registered task
horizon through a second one.
`PactPlaceCorridorV1010FourObjectSampler` was in neither, so it **fell through
to the default sampler** and to the generic 900-step horizon.

The signature was unmistakable once looked at: `pact_v106_frame_telemetry` was
`{}` in 32 of 32 attempts and `pact_v106_speed_amendment` was null, while a
known-good V10.8 row carries eight telemetry fields. Downstream,
`grasp_phase_success` was often true and `place_phase_success` was false in
every attempt — the expert was routing in an environment that was not the one
the row described.

This is the same failure mode recorded against the V9 expert version
allow-lists: a new sampler that is not added to them silently gets no routing at
all, and every check that does not test routing still passes. The preflight did
not catch it because the preflight constructs its own config directly and never
goes through `_make_config`.

## Fix

`_make_config` now resolves `PactPlaceCorridorV1010FourObjectSampler` and gives
it `TASK_HORIZON_V106`, since V10.10 is the V10.6 lane with four slots parked.
A single attempt is verified to emit non-empty V10.6 frame telemetry before the
collection is restarted.
