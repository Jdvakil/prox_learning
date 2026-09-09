# Superseded V10.10 preflight, attempt 01

Retained, not deleted. 0/24 cells passed, so nothing proceeded.

Two distinct causes, both addressed rather than waived:

**All 24 cells** failed one check: the row bound a hash of the *unjittered*
layout while the sampler hashed the *jittered* one. V9.3 applies per-episode
clutter jitter, so a positional hash is not knowable before sampling and cannot
be a row binding. The row now binds a four-object **identity** hash (slot, uid,
role), and the per-episode positional hash is recorded as telemetry alongside.

**Two cells** — F0|right|pos5 and F2|left|pos5 — additionally began in
robot-clutter contact at their first seed (2 and 11 pairs). That is not a defect
but exactly the condition the plan says to identify separately and reject before
a scientific rollout. The preflight now resamples on any initial-state contact
and records how many draws each cell needed.
