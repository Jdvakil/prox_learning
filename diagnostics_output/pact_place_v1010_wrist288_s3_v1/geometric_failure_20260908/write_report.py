"""Write a separate diagnosis; never change the experiment's EVAL.md reports."""
import json,sys
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).parent;N=int(sys.argv[1]) if len(sys.argv)>1 else 234
doc=json.loads((OUT/f'analysis_{N}.json').read_text());s=doc['summary'];rows=doc['rows']
def rr(seed,arm='PACT'):return [r for r in rows if r['seed']==seed and r['arm']==arm]
def fmt_count(seed,key):
    z=s[str(seed)]['PACT'];return f"{z['counts'][key]}/{z['n']}"
def number(seed,key):return s[str(seed)]['PACT']['counts'][key]
def stage(seed,key):return s[str(seed)]['PACT']['stages'].get(key,0)
def median(rs,key):
    import statistics
    vals=[r[key] for r in rs if r.get(key) is not None]
    return statistics.median(vals) if vals else None
def mm(x):return f'{x*1000:+.1f}' if x is not None else 'unavailable'
n3=s['3103']['PACT']['n'];p3=rr(3103)
blocked3=[r for r in p3 if r.get('rim_blocked_signature')]
paired3=s['3103']['paired'];actonly4=s['3104']['paired']['ACT_only']
reps={r['rollout_id']:r for r in rows};rep4=reps[doc['representatives'][0]];repact=reps[doc['representatives'][1]]
rep5=reps[doc['representatives'][2]];rep3=reps[doc['representatives'][3]]
successgap=number(3105,'success')-number(3104,'success')
liftgap=number(3105,'lift')-number(3104,'lift')
maxfk=max(r.get('fk_verification_max_error_around_close_m',0) for r in rows)
maxfkall=max(r['fk_verification_max_error_m'] for r in rows)
parts=[f'''# PACT 3104 versus 3105: geometric pickup diagnosis and seed 3103 check

Generated {datetime.now(timezone.utc).isoformat()}. Source snapshot: {N} validated rollouts,
comprising {n3} paired cases for seed 3103 and 50 paired cases each for 3104 and 3105.
{'All three seed evaluations are complete.' if n3==50 else 'Seed 3103 is provisional; its remaining evaluation continues unchanged.'}

## Finding

The dominant observable problem is **failure to acquire a stable cup-wall grasp**.
Seed 3104 more often arrives with an unsuitable hand position relative to the cup,
contacts the upper cup geometry, and closes while the actual hand has not reached
the requested grasp depth. The hand can be displaced upward during closure, leaving
only brief pad contact and little or no cup lift. Seed 3105 more often achieves a
deeper, offset grasp that pinches a cup wall and survives lift.

This is a diagnosis of the failure mechanism visible in the recorded trajectories.
It does **not** establish a unique training bug, nor prove that a particular obstacle
layout causes the entire seed gap. Training seed and final-instance seed both change
between 3104 and 3105; a same-instance cross-checkpoint experiment is needed to separate
their causal contributions.

## Where the 16-success gap arises

| PACT outcome | Seed 3104, n=50 | Seed 3105, n=50 | Seed 3103, n={n3} |
|---|---:|---:|---:|''']
for label,k in [('Final task success','success'),('Any target touch','touch'),('Any contact-only hold','held'),('Cup lifted at least 1 cm','lift'),('Any tray support','support'),('Collision-free task success','collision_free_success')]:
    parts.append(f'| {label} | {fmt_count(3104,k)} | {fmt_count(3105,k)} | {fmt_count(3103,k)} |')
parts.append('\n| Mutually exclusive final outcome | 3104 | 3105 | 3103 |\n|---|---:|---:|---:|')
for k in ('success','no_target_interaction','touched_without_hold','held_without_lift','lifted_without_placement','supported_without_final_success'):
    parts.append(f'| {k.replace("_"," ")} | {stage(3104,k)} | {stage(3105,k)} | {stage(3103,k)} |')
parts.append(f'''
PACT gains **{successgap} successes** in 3105. The count reaching a 1 cm lift rises by
**{liftgap}**, so {liftgap} of the {successgap}-case difference appears before that
milestone. Failures after a lift number {number(3104,'lift')-number(3104,'success')}
and {number(3105,'lift')-number(3105,'success')}, respectively. The largest individual
loss is touch without hold: **18 versus 8**. There are also four additional
no-touch failures in 3104. Those seven 3104 no-touch failures all move an empty hand
near the tray later in the rollout; avoidance without pickup is not task success.

The `held` sensor is explicitly a contact heuristic: it asks whether the object
touches the gripper without touching another object. It is not a stable-grasp
measurement. This analysis therefore also uses actual target translation and
simultaneous finger-pad contacts sampled every 2 ms.

## The geometry and the contact mechanism

The evaluated target uses **Cup_10's five primitive collision boxes**, verified
against the saved model positions, quaternions and sizes in every analyzed episode.
It is hollow: four side-wall boxes plus a bottom box. At the settled initial
orientation, the upper collision edge is **71.98 mm above the target body origin**.
The target's body origin is not the rim or a grasp point. All TCP poses were
transformed from the robot-base frame into world coordinates before comparison.

There are two relevant facts:

1. **A successful grasp is usually a wall pinch.** Simultaneous contacts on the
   same side-wall primitive last at least one continuous second in
   **{s['3104']['PACT']['successful_same_wall_grasp_over_1s']}/19** successful
   seed-3104 cases and **{s['3105']['PACT']['successful_same_wall_grasp_over_1s']}/35**
   successful seed-3105 cases. Both pads touching something is insufficient: in
   the touch-without-hold failures, neither seed has a continuous bilateral pad
   contact lasting one second.
2. **Seed 3104 more often approaches the cup centre and fails to get below the
   upper edge.** Its commanded hand centre is within 2 cm horizontally of the cup
   origin at first closure in **{s['3104']['PACT']['centered_command_under_2cm']}/50**
   cases, versus **{s['3105']['PACT']['centered_command_under_2cm']}/50** for seed 3105.
   Of the 18 versus 8 touch-without-hold failures,
   **{s['3104']['PACT']['shallow_touch_no_hold']} versus
   {s['3105']['PACT']['shallow_touch_no_hold']}** never put the TCP below the rim
   reference while near the cup in the defined early grasp window.

The 2 cm cutoff describes a tendency; it is not a universal failure boundary.
Cup yaw, hand orientation, approach side and the geometry of the fingers all matter.
Some centred grasps succeed, and some offset grasps fail.

| Median in PACT touch-without-hold failures | 3104 | 3105 |
|---|---:|---:|
| Minimum TCP height relative to rim in early grasp (mm) | {mm(s['3104']['PACT']['touch_no_hold_medians']['minimum_rim_relative_tcp_early_grasp_m'])} | {mm(s['3105']['PACT']['touch_no_hold_medians']['minimum_rim_relative_tcp_early_grasp_m'])} |
| Commanded TCP Z change over first two steps after closure (mm) | {mm(s['3104']['PACT']['touch_no_hold_medians']['command_z_change_2_steps_after_close_m'])} | {mm(s['3105']['PACT']['touch_no_hold_medians']['command_z_change_2_steps_after_close_m'])} |
| Actual TCP Z change over those two steps (mm) | {mm(s['3104']['PACT']['touch_no_hold_medians']['actual_z_change_2_steps_after_close_m'])} | {mm(s['3105']['PACT']['touch_no_hold_medians']['actual_z_change_2_steps_after_close_m'])} |

That last comparison rules out the simplistic interpretation that the immediate
upward motion is merely a commanded early lift: **the requested motion is still
downward, while the actual hand moves upward**. Recorded gripper–cup contacts and
the command/actual discrepancy support a blocked approach and closure-induced
displacement. The traces do not retain contact forces or complete passive-finger
states, so the exact force path and millimetre-level pad penetration cannot be
reconstructed uniquely.

The cup's primitive rim is also direction-dependent. Projecting its initial upper
wall corners onto the hand's closing direction gives a median span of
{median(rr(3104),'initial_rim_span_along_closing_xy_m')*1000:.1f} mm in 3104 and
{median(rr(3105),'initial_rim_span_along_closing_xy_m')*1000:.1f} mm in 3105, against
the gripper's nominal 87 mm maximum inner gap. This is a horizontal footprint
descriptor, not exact tilted-finger clearance. It explains why a wall pinch can
be more suitable than trying to surround the entire rim. **It does not explain
the result by itself**: {sum(r['success'] and r.get('initial_rim_span_exceeds_nominal_open_gap',False) for r in rr(3105))}
of 3105's 35 successes have a projected full-rim span above 87 mm. Cup yaw alone
is therefore not a reliable success/failure classifier.

The stricter diagnostic signature—touch, no 1 cm lift, command at least 5 mm below
the rim at closure, actual hand at least 5 mm above that command, and no achieved
TCP descent below the rim in the early window—occurs in
**{number(3104,'rim_blocked_signature')}/50** seed-3104 cases and
**{number(3105,'rim_blocked_signature')}/50** seed-3105 cases. This is a descriptive
flag, not a new task-success criterion or an outcome replacement rule.

## Identical-scene ACT control and representative trajectories

The strongest control available here is ACT versus PACT **within** a seed, where
the initial physics state is paired. Seed 3104 has **{actonly4['n']} ACT-only
successes; {actonly4['pact_stages'].get('touched_without_hold',0)}** are PACT
touch-without-hold failures. Across the ACT-only group, the median paired difference
in achieved minimum grasp height is
**{mm(actonly4['median_pact_minus_act_minimum_rim_height_m'])} mm**: PACT remains higher.
These instances are physically solvable under the unchanged task and controller.

For episode `{rep4['episode_id']}`:

- PACT closes at step {rep4['first_close_command_step']}, with a horizontal TCP
  offset of {rep4['first_close_xy_error_m']*1000:.1f} mm. Its actual TCP is
  {rep4['first_close_tcp_above_translated_rim_m']*1000:.1f} mm above the rim while
  its commanded TCP is {abs(rep4['command_at_close_tcp_above_rim_m'])*1000:.1f} mm
  below it. The hand moves up during closure; the cup never reaches a 1 cm lift.
- ACT on the identical instance reaches
  {abs(repact['minimum_rim_relative_tcp_early_grasp_m'])*1000:.1f} mm below the rim
  in the early grasp and completes the task. No external-obstacle contact explains
  PACT's failure in this example.
- The illustrated successful PACT-3105 case is the closest successful initial cup
  XY position in the same nominal cell, **17.78 mm** away. It is an illustration,
  not an identical-state causal control.

![Commanded and actual trajectories, cup-wall footprints, and lift](representative_trajectories_{N}.png)

## Is seed 3104's environment intrinsically harder?

The records do not support a single seed-specific obstacle defect as the main
explanation. All three blocks use the same sampler, scene variants, target asset,
four clutter identities, policy history and controller settings. The nominal core
contains two instances in each of the same 24 cells. After excluding the two
predeclared extras per seed, PACT still scores **19/48 versus 34/48**. The gap is
present across all four layout families, not just one family or intrusion side.

Actual target pose and robot-start jitter differ between blocks. The retained
geometry includes these values, exact initial cup boxes and the initial clutter
AABBs. A nearest-clutter AABB distance is only an origin-to-box descriptor; it is
not full-arm swept clearance. At first closure, cup visibility had occurred in
46/50 PACT-3104 cases and 48/50 PACT-3105 cases. Every touch-without-hold case in
both blocks had seen the target before closure.

Direct robot–clutter contact before the first target touch occurs in
**{number(3104,'robot_clutter_before_first_touch')}/50** PACT-3104 cases versus
**{number(3105,'robot_clutter_before_first_touch')}/50** PACT-3105 cases. Thus a
larger count of these collisions cannot account for the worse 3104 result.
Official hazard contact rates are also relatively low for PACT: 5/50 versus 3/50.
This does not mean obstacles never contribute; individual 3103 and 3104 failures
have clutter interference or target–bottle contact.

The apparent pendant-pose effect is not a consistent difficulty ordering:
the centre-pose cell group has PACT success 4/16 in 3104 and 14/17 in 3105.
ACT's overall success is nearly unchanged across the two blocks, 22/50 versus
23/50. These are evidence against a broad environment difficulty shift, while
still allowing policy-specific interactions with randomized target poses.

![All-case grasp geometry and initial target positions](geometry_overview_{N}.png)

## Does seed 3103 have this issue?

**Yes, the failure mode is present.** In the {n3} currently included pairs,
PACT has {stage(3103,'touched_without_hold')} touch-without-hold failures;
{s['3103']['PACT']['shallow_touch_no_hold']} satisfy the achieved-height description,
and {len(blocked3)} episodes satisfy the stricter blocked-descent signature.
Its task result in this snapshot is **{number(3103,'success')}/{n3}**, versus
**{s['3103']['ACT']['counts']['success']}/{n3}** for ACT.

The example `{rep3['episode_id']}` shows the same command/actual split: PACT
requests descent below the rim but remains above it and fails to lift, while
paired ACT succeeds. There is no concurrent robot–clutter contact in the
closure window of this example. Its earlier brief clutter contact is retained
in the record; this is not a claim of a completely collision-free rollout.

Seed 3103 is not an exact copy of 3104's distribution of errors. It also has
off-centre misses and cases involving the cup and nearby bottles. A whole
environment seed cannot be labelled as having or lacking a single geometric
defect independently of the policy trajectory.

## What is established, and the next discriminating test

Established: the large success gap is mainly pickup acquisition; a frequent
failure involves a shallow, misaligned, contact-blocked approach; successful
grasps usually maintain a cup-wall pinch; the failure also occurs in 3103.

Unresolved: why training seed 3104 learns this grasp-position tendency more often.
Plausible contributors include spatial generalization and averaging incompatible
grasp approaches. The saved final action trace does not isolate those mechanisms.
All checkpoints reached the same 60,000-update budget and use the same inference
recipe. More rollouts of a checkpoint do not change its behavior.

The most discriminating follow-up is to run the frozen 3104 and 3105 PACT
checkpoints on the **same predeclared physical instances**, including successful
and failed cases, and compare hand placement and achieved depth. A subsequent
grasp intervention could test an offset wall pinch or require achieved engagement
before lift. Merely changing the gripper threshold is not justified by these
traces: the commanded and physically achieved depths are already different.
No such interventions or additional diagnostic rollouts were launched here.

## Methods and limits

- Every included case comes from validated paired completions; actual zero-exit
  receipts and result hashes were checked. Commands were checked against all
  recorded arm/gripper actions, including the unchanged 127.5 decoder threshold.
- The command at index k follows observation k and produces observation k+1.
  Event plots align on the first close command, with 66 ms per control step.
- Cup collision geometry comes from the primitive asset actually named in
  telemetry, not its alternative mesh collision asset. All five boxes were
  matched against each saved initial model.
- The rim reference follows measured target translation while retaining initial
  target orientation. Dynamic target rotation is not retained. This is a TCP
  height reference, not an exact measure of finger insertion or penetration.
- Early grasp is through 60 control steps after first closure, restricted to
  hand–cup horizontal distance under 6 cm and cup lift below 1 cm. Missing near-cup
  samples are marked unavailable. Threshold sensitivity is retained in JSON.
- Forward kinematics uses the frozen robot asset and verifies the 35 cm mount
  offset against every recorded state. Maximum reconstruction discrepancy is
  **{maxfk*1000:.3f} mm around closure** and {maxfkall*1000:.3f} mm over all states;
  the centimetre-scale command/actual discrepancies exceed this error.
- Same-wall and opposite-wall contact modes use simultaneous raw physics contact
  identities. `clutter` telemetry can include target–clutter contacts, so direct
  robot–clutter contacts were separately identified for this diagnosis.
- No RGB videos were retained for these H5-only rollouts. Visibility statements
  use recorded target image-point counts; there was no video-based visual review.
- Raw trajectories, checkpoints, official scores, original reports, and current
  evaluation settings were not changed. This report is separate from the deleted
  environment audit and from the original n=50 experiment report.

Reproduction: run `extract.py`, then `augment.py features_{N}.json`,
`contact_modes.py geometry_{N}.json`, `command_geometry.py geometry_{N}.json`,
`summarize.py {N}`, and `write_report.py {N}` in this directory with
`/root/act_retrain_venv/bin/python`. Extractor snapshots can grow only as new pairs
finish; the numbered inputs pin this report's denominator. Analysis JSON, per-case
arrays, source hashes and PNG/PDF figures are retained beside this document.
''')
(OUT/f'GEOMETRIC_DIAGNOSIS_{N}.md').write_text('\n'.join(parts))
print(OUT/f'GEOMETRIC_DIAGNOSIS_{N}.md')
