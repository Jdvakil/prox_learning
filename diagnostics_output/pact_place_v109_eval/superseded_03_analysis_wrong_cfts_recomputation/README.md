# Superseded V10.9 analysis, attempt 01

Retained, not deleted. `verified: false` with 10 problems, all of one kind:
"recorded collision-free flag disagrees with the recomputation".

The disagreement was in the **cross-check**, not the data, and the reported
counts were never affected — the headline `collision_free_task_success` reads
the recorded flag from the rollout, which is correct.

`PactPlaceContactAudit.summary()` defines

    non_target     = hazard_bar + other_environment + clutter + mounted_fixture
    collision_free = (non_target == 0)

with `grasp_target` and `place_receptacle` excluded. The cross-check instead
used `hazard_bar == 0 and other_environment == 0`, which is the V5 *converter's*
demonstration filter, not the V5 *evaluation* endpoint. Those two agree only
where there is no clutter and no mounted fixture — true of the V2 corridor V5
ran on, false of the real V9.5 clutter here. The 10 flagged rows are exactly the
successful episodes that touched clutter.

Fix: the recomputation now reproduces the audit definition, and
`non_target_contact_entries` is carried per row so the comparison is visible
rather than implicit.
