# STATUS — prox_learning

**Date:** 2026-08-26 · **Branch:** `main` · **Machine:** one RTX 4090, 62 GB RAM

This file is the plain-English status of the project: what we are trying to prove, what we built, what
the numbers say, what went wrong, and what to do next. It is written to be readable by someone who has
never seen this repo.

**Update 2026-08-26 (hallway smoke, not a paper number).** We trained cameras-only (ACT) and
cameras-plus-skin (PACT-raw) on 152 coauthor hallway pick-and-place demos and tested each 20 times.
Place success **15% vs 35%**. Bar hits **30% vs 20%**. Fisher p = 0.27 and 0.72 — luck can still
explain the gap. The published headline is still the 2026-07-05 hidden-bar crash cut **66% vs 40%**
(n=50, p = 0.016). Details: `experiments.md`, README §13.1.

**How to read it.** Sections 1–5 are the whole story and need no background. Section 6 onward gets more
detailed. Section 11 is file paths and commands — skip it unless you are going to run something.
`README.md` is the manual (how to install and run); this file is the report (what happened and what it
means).

---

## 1. What this project is about

**The problem.** A robot arm has to reach into a tight space — think reaching into a drawer, or into a
lab fume hood — and there are things in the way. Cameras are bad at this. The arm's own body blocks the
view, the space is dark inside, and a thin obstacle can sit in a blind spot. A human in this situation
stops looking and starts *feeling*: you reach behind the sofa for your keys with your eyes on the
ceiling, and your hand tells you where the sofa is.

**What we built.** We gave a robot arm that sense of touch-at-a-distance. 40 tiny distance sensors are
spread over the whole arm — forearm, upper arm, wrist. Each one works like a car parking sensor: it
reports how far away the nearest thing is, in its own little cone. They are not cameras. They do not
see color, texture, or shape. They only answer "how close is something, right here?"

**The question we are trying to answer.** Does adding this skin make the robot *safer* than cameras
alone? Not smarter, not faster — safer. Fewer crashes into things.

**The answer so far: yes, but with an asterisk.** The skin helps a lot in the one situation where the
skin is the only sense that can perceive the obstacle. It helps less when the cameras can also see the
obstacle, and not at all when there is no obstacle. That pattern is exactly what you would expect if
the effect is real, which is the encouraging part.

**One surprise worth knowing up front.** We built a small neural network that turns skin readings into
a "flinch" motion — the reflex. It works beautifully on its own. But feeding that polished reflex
signal into the robot's decision-making did *nothing*. Feeding it the 40 raw distance numbers,
unprocessed, worked. The dumb version won. That is written up honestly in section 6.

---

## 2. Words you need (read once, then skip)

| Word | What it means here |
|---|---|
| **policy** / **brain** | The neural network that looks at the robot's senses and decides how to move. |
| **demonstration** | One recorded example of the task being done correctly, by a scripted expert that can "cheat" (it knows exactly where everything is). |
| **imitation learning** | How the brain is trained: copy the demonstrations. It never learns by trial and error, only by mimicry. This matters — a brain trained this way only learns to use a sense if the demonstrations *can't be explained without it*. |
| **ACT** | The specific off-the-shelf brain design we use. Nothing about it is ours. |
| **vanilla** | A brain with cameras only. Our comparison point. |
| **pact_raw** | Same brain, plus the 40 raw distance numbers. **This is the one that works.** |
| **pact_trunk** | Same brain, plus the processed reflex signal. Did nothing. Abandoned. |
| **success rate** | Out of N attempts, how many picked the object up. |
| **collision rate** | Out of N attempts, how many touched something they shouldn't at any point. Lower is better. |
| **strict success** | Picked it up **and** never touched anything. The honest score. |
| **the three test setups** | Every brain is tested in three conditions, described next. |
| **N** | How many attempts a number is based on. N = 50 is our standard; N = 25 turned out to be too few (see §4). |
| **"points"** | Percentage points. "A 26-point drop" means 66% fell to 40%. |
| **p-value** | Roughly, the chance of seeing a gap this big by pure luck. p = 0.016 means about a 1.6% chance. Below 0.05 is the usual bar for "probably not luck". |

**The three test setups.** We put a hazard — a bar lying across the workspace — into the scene, and we
control whether the *cameras* can see it. The physics never changes; the bar is always solidly there
when it is there. Only the cameras' view of it changes.

| Setup | Is the bar there? | Can the cameras see it? | Can the skin feel it? | What it tests |
|---|---|---|---|---|
| **visible** | yes | yes | yes | The normal situation. |
| **invisible** | yes | **no** | yes | The skin is the *only* sense that knows. |
| **free** | **no** | — | — | Control: nothing to hit. |

The trick that makes "invisible" possible is a rendering flag: the bar is put in a display layer that
the camera renderer ignores and the distance-sensor renderer includes. It is a clean lab version of an
ordinary real problem — fog, smoke, darkness, a dirty lens, or an obstacle the same color as the wall
behind it.

**Why "free" matters more than it looks.** With no bar in the scene, every collision the robot records
is the arm brushing the walls of the tight space it has to reach into. So the free setup measures the
*background* level of bumping. It turns out to be high — about 60% — and that becomes a recurring
problem in this document.

---

## 3. The main result

**Adding raw skin readings cuts collisions from 66% to 40% in the setup where only the skin can sense
the hazard — and the robot is no worse at the actual task.**

Measured 2026-07-05, 50 attempts per number.

Collisions (lower is better):

| Brain | no bar (free) | bar hidden (invisible) | bar visible |
|---|---|---|---|
| cameras only (`vanilla`) | 60% | 66% | 64% |
| **cameras + raw skin (`pact_raw`)** | 58% | **40%** | 50% |
| cameras + reflex signal (`pact_trunk`) | 64% | 72% | 58% |

Task success (higher is better):

| Brain | no bar | bar hidden | bar visible |
|---|---|---|---|
| cameras only | 22% | 36% | 28% |
| cameras + raw skin | 18% | 30% | 16% |
| cameras + reflex signal | 34% | 34% | 32% |

### Why we believe it

**1. It is not luck.** 66% → 40% has p = 0.016 — about a 1.6% chance of a gap that big by luck.

**2. It is not just timidity.** The obvious objection is that the skin made the arm nervous, so it
crawls around and bumps into less of everything. If that were true, collisions would drop in *all three*
setups. In the no-bar setup they don't move at all (60% → 58%). The improvement only shows up where
there is actually something to avoid.

**3. The improvement lines up with how much the skin is needed.** This is the strongest pattern in the
whole project:

| Setup | How much the skin helps | Why that makes sense |
|---|---|---|
| no bar | 2 points (nothing) | Nothing to avoid. |
| bar visible | 14 points | Cameras could help, skin adds some. |
| bar hidden | 26 points | Only the skin knows. |

Random noise does not sort itself into a neat order like that.

**4. The skin is not redundant with the cameras — it is doing a job nothing else does.** Look at the
cameras-only row: 64% collisions when the bar is fully visible, 66% when it's hidden. **The camera-only
brain does not avoid the bar even when it can see it perfectly.** From 105 demonstrations it never
learned obstacle avoidance from vision at all. So the skin is not a backup sense here; it is the only
sense carrying avoidance information in any brain we have trained.

**5. The skin makes the brain a better predictor.** Independently of the driving tests, the raw skin
readings improve the brain's ability to predict what the expert did next by about 21%. Three separate
measurements — this one, a statistical probe, and the driving tests — all agree.

### Why it might be wrong

**The collision counter is blunt.** It counts *any* contact between the robot and the environment
(excluding the floor and the object being held). It cannot tell "rammed the hazard bar" from "brushed
the wall of the narrow space it was told to reach into". And the background brushing rate is about 60%.

We measured how much the bar itself adds to the count: comparing no-bar to bar-visible for each brain,
the bar's own contribution is only about **4–6 points on top of a 60-point floor**. So we are trying to
observe a small effect through a very noisy instrument.

That does *not* invalidate the result — the effect is statistically solid and, more convincingly, it
sorts itself by setup. But it does change how we describe the mechanism. A 26-point drop is bigger than
the bar's own 4–6 point contribution, which means the skin is probably not just dodging the bar; it is
making the arm generally more careful, and it is most careful in the conditions where the skin fires
most. Both readings are interesting. Telling them apart needs a fix described in §7 item 1.

**It is one training run.** One random seed, one dataset. Section 4 is a cautionary tale about exactly
that.

---

## 4. The newest experiment: blurry cameras (finished this morning)

### What we were trying to do

If the story is "the skin matters because cameras fail", a natural test is to *make* the cameras fail
and watch what happens. The cheapest way to break a camera is to blur it. So we trained three extra
camera-only brains with every training image blurred — mildly (σ = 2), more (σ = 4), a lot (σ = 8) —
and then tested them on clear images. (σ, "sigma", is just the blur strength in pixels.)

The expectation: more blur during training → a worse, or at least differently-behaved, robot. An
orderly ladder.

The training numbers cooperated perfectly. The brains' prediction error rose in exact step with the
blur: 0.0755 → 0.0836 → 0.0948 → 0.1100. Textbook.

Then we ran the actual robot. 225 attempts, 13 hours 21 minutes, finished 08:32 today.

### What we got

| Blur | success, bar hidden | success, no bar | success, bar visible | collisions (hidden / none / visible) |
|---|---|---|---|---|
| none (our anchor) | 36% | 22% | 28% | 66% / 60% / 64% |
| σ = 2 | **0%** | 4% | 8% | 48% / 48% / 52% |
| σ = 4 | **40%** | 40% | 24% | 88% / 84% / 84% |
| σ = 8 | 24% | 24% | 24% | 68% / **28%** / 68% |

### What it means

**There is no ladder. The experiment did not work, and the reason is that random run-to-run variation
is bigger than the effect of blur.**

Four things say so, strongest first:

**1. One brain contradicts itself.** Look at the σ = 8 row: 68% collisions with the bar hidden, 68%
with the bar visible, and **28% with no bar at all**. Those three tests use the *same* brain and differ
only in whether a bar exists. A 40-point swing between them cannot be caused by blur — blur is
identical in all three. It is noise. And that tells us the size of the noise: **roughly ±40 points at
25 attempts.** Which is enormous, and is why 25 attempts is not enough for anything.

**2. Nothing is in order.** σ = 2 destroys the robot (0% success). σ = 4 *recovers*, to 40% — better
than the un-blurred anchor. σ = 8 lands at exactly 24% everywhere. Collisions go 48% → 85% → 28–68%.
A real effect would be ordered.

**3. Low collisions did not mean safe.** The σ = 2 brain has the fewest collisions of any brain here.
It also almost never completes the task. It learned to barely move — it spends fewer steps in contact
(25 out of 201, versus 31 for the normal brain) because it goes nowhere. This is the single most
important lesson in the section: **a low collision number can mean "cautious" or it can mean "broken",
and the collision number alone cannot tell you which.** Only strict success (did it *and* stayed clean)
separates them, and by that measure σ = 2 is dead last at 0–8%.

We checked the obvious alternative explanation: maybe the low-collision runs just ended early, before
reaching the dangerous part. No — every single run in all 12 conditions used the full 201 steps. The
robot had every opportunity to crash and simply wasn't going anywhere useful.

**4. Two results crossed the "probably not luck" line, in opposite directions.** Out of nine
comparisons, σ = 4 came out significantly *worse* (p = 0.040) and σ = 8 significantly *better*
(p = 0.014). When you run nine tests, a couple crossing 0.05 is expected, and here they disagree about
the direction. These are coin flips, not findings.

### What we keep from it

- **A retraction.** An earlier version of this document argued that a flat blur result would strengthen
  the main claim. It doesn't. This grid is too noisy to prove anything either way. It is now recorded
  as a **negative methodological result**: training a separate brain per blur level, one run each, does
  not produce a usable measurement in this environment.
- **A hard number for how noisy things are:** ±40 points at 25 attempts. Everything from now on uses 50
  attempts minimum. It is also why the §3 result — 50 attempts, with the no-bar control coming out
  flat — is trustworthy while this grid is not.
- **A warning that cost us 13 hours to learn:** the training error ladder was flawless and predicted
  none of the actual robot behavior. **Never claim anything about robot behavior from training numbers
  alone.**
- **One genuinely useful fact:** blurring to σ = 4 did not hurt task success. So this task needs only
  *coarse* vision — "the cup is roughly over there" — which survives blur. That reshapes the next
  experiment (§8).
- **The three brains keep one job:** they are the control for the next experiment (train blurry, test
  blurry at the same strength), which separates "information was lost" from "the test looked different
  from training".

### Two bugs fixed today

1. **The summary table printed nothing.** After 13 hours the script printed a table with one row in it.
   A file-matching pattern didn't account for the timestamp prefix on the output folder names, so it
   matched zero folders and the script cheerfully printed an empty table. **No data was lost** — all 9
   result files were on disk the whole time, and the tables above are computed from them. Fixed.
2. **No GPU check before starting.** On 2026-07-28 a graphics driver mismatch made every single attempt
   fail instantly, and the script ran all night without noticing. There is now a two-second check up
   front that refuses to start and prints the fix.

---

## 5. Timeline

| When | What happened |
|---|---|
| — | 40-sensor skin arm model built and verified |
| 2026-06-11 | Sensor proof: the skin reads **exactly the same numbers** through blur and near-darkness, while cameras fall apart. This one fact is the seed of the research idea in §8. |
| 2026-06-13 | Reflex network audited — usable, with caveats (§6) |
| June | Camera-only baseline sweep: two clean findings about training settings (§6) |
| 2026-06-18 | **First attempt at skin + cameras: no improvement.** p = 0.76, i.e. nothing. |
| 2026-07-02 | Full audit of the code, checked adversarially: **no bug.** The wiring was correct; the reason was in the data. |
| 2026-07-03 | Diagnosis confirmed with statistical probes. Version 2 of the dataset built, using the hidden-bar design. |
| 2026-07-04 | A setting called `--temp_agg_off` found to be broken. **Every result that used it before this date is invalid.** |
| 2026-07-05 | **The result in §3.** |
| 2026-07-06 | The reflex-signal version abandoned as useless. |
| 2026-07-24 | Three blurry-camera brains trained. |
| 2026-07-28 | Blur test destroyed by a graphics driver mismatch. Machine rebooted. |
| 2026-08-09 → 08-10 | **Blur test finished. It didn't work (§4).** Two script bugs fixed; new test-time blur switch added. |

---

## 6. What has been built

Eight pieces. Each is finished and verified unless said otherwise.

### The arm with skin

A model of a real Franka FR3 robot arm with 40 distance sensors grafted onto it, spread over 7 body
segments. Each sensor is an 8×8 depth image with a 45° cone.

- Aiming them was the hard part. The obvious approaches (using the mesh surface directions, or the
  robot's own joint axis definitions) both gave wrong answers, so each sensor is aimed outward from its
  segment's axis with ray-casting to repair the awkward ones. All 40 pass a test that they don't just
  see the arm itself.
- **The bug that would have silently ruined everything:** the simulator's depth camera has a minimum
  range that scales with the size of the scene. At room scale, the default threw away every reading
  closer than about 10–20 cm — which is the *entire* range this project cares about. Fixed by setting
  it very small. Any new sensor work must check this.
- Accuracy: a flat plate at 15 cm reads 14.5 cm ± 1.2 cm. Reconstructing the inside of a pipe comes out
  to 5.6 mm error. Range response is essentially perfectly linear.
- Coverage: the skin can sense in **83% of all directions** around the arm. The wrist camera manages
  10%.
- **The fact everything in §8 rests on:** skin readings are *bit-for-bit identical* under heavy blur and
  near-total darkness. Cameras collapse under both.

### The practice environments

Simulated scenes to collect demonstrations in — a bench with a hazard bar, a drawer to reach into, a
cluttered shelf, a fume hood. The demonstrations come from a scripted expert that knows exactly where
everything is and swings wide around the bar when the straight path would pass within 18 cm of it.

### The reflex network

A small network that takes the 40 skin readings and outputs a "back off this way" motion for the arm's
7 joints. It is not a crash detector — it outputs the escape move directly. Trained by copying a
hand-written physics rule ("push away from anything close"), so that afterwards it needs no knowledge
of the scene, just the skin.

Audited 2026-06-13. Verdict: usable, but with real caveats that must be stated in any writeup:

- **The easy way of splitting train/test data cheats.** The data has only 7,813 genuinely distinct arm
  poses spread over 15,000 samples, so 74% of the test set shares a pose with the training set.
  Headline numbers have to use the pose-grouped split.
- **Only one of its scores is honest.** Direction accuracy holds up under the honest split (0.926 →
  0.924). Its error magnitude gets 69% worse. Its "stay quiet when nothing is near" score is nearly
  meaningless, because in the training data those cases are exactly zero by construction.
- **It is coarse.** It knows *which arm segment* something is near, not the fine pattern within a
  sensor. Scrambling which sensor belongs to which segment wrecks it; scrambling pixels inside a sensor
  barely matters. Do not claim fine spatial resolution.
- **A nearest-neighbor lookup table ties it** on direction accuracy. Its real advantage is magnitude.
- **It under-reacts exactly when it matters most** — its output is 64% of the correct size when the
  obstacle is closest. The direction score hides this completely.

In hindsight this audit predicted the failure of the reflex-signal version: a feature that is coarse,
matchable by a lookup table, and under-calibrated in the danger zone is exactly the kind of feature a
mimicry-trained brain will find nothing useful in.

### Demos and figures

Five demo videos, all driven by the same skin-only reflex, differing only in how the obstacle moves:
`flinch`, `sphere`, `react`, `moving`, `orbit`. `react` is the one to show people — the arm swerves
around three bars mid-motion and rejoins its path, without the clock ever stopping. `sphere` proves the
reflex doesn't care about appearance: trained on orange box-shaped bars, it reacts identically to a blue
sphere, because all it ever sees is a distance blob. Plus a visualizer and one script that produces all
29 paper figures.

### Camera-only brain (the baseline)

Standard ACT: two camera views plus joint positions. No skin. Tested *inside the same simulator the
demonstrations came from*, which is why every number in this document can go in one table.

June sweep, 20 attempts each:

| run | training length | how far ahead it plans | success | strict success | collisions |
|---|---|---|---|---|---|
| A | 2000 | 50 steps | 35% | 20% | 30% |
| B | 2000 | 100 steps | **40%** | 15% | **60%** |
| C | 5000 | 100 steps | 30% | 25% | 20% |

Two clean findings, one variable each:

1. **Planning further ahead doubles the crashes** and doesn't improve success. The robot commits to a
   longer blind sequence and wedges itself deeper before it senses again.
2. **Training longer cuts crashes 3×** and costs a little success — a cleaner, more conservative arm.

And note how the ranking flips: run B looks best on success (40%) and is worst once you count crashes
(15% strict). Run C, the most-trained and safest, is actually the best robot. **Any comparison in this
environment that only reports success is misleading.**

### Skin + cameras (the main experiment)

The skin is wired *into* the brain, not bolted on as a safety override. The skin features become extra
inputs inside the transformer, sitting alongside the camera and joint inputs. Only the small connecting
layer trains; the reflex network is frozen. With the skin switched off, the model is provably identical
to the camera-only version — verified, not assumed.

Three ways to tap the skin: the reflex network's internal state (256 numbers), its output motion (7
numbers), or the **raw closeness of each sensor (40 numbers)**, skipping the reflex entirely. The raw
tap is the one that works.

**The first attempt failed (2026-06-18), and understanding why is the most valuable work in the
project.** No version beat cameras alone (p = 0.76). A line-by-line audit found no bug, and identified
four real causes:

1. **The brain was ignoring the skin, and the training gave it no reason not to.** Blanking the entire
   skin changed the brain's output by almost nothing. All three versions had statistically identical
   prediction error — the skin added *zero* predictive value. The root cause: the demonstrations came
   from a scripted expert whose swerving is completely predictable from the cameras alone. If a sense
   can be inferred from the other senses, mimicry training will not learn to use it. **This is a data
   design problem, not an engineering problem.** It is why version 2 of the dataset hides the bar from
   the cameras.
2. **The danger signal was always on.** In 100% of the demonstration frames, *something* is within
   half a meter of the arm, and in 40–60% of frames the arm is inside the reflex network's "back off"
   zone — while the demonstrated correct action is "keep going into the drawer". The danger feature
   screams constantly during perfectly safe behavior, which teaches the brain to ignore it. The
   standalone demos avoid this by subtracting a reference reading of the empty scene, which is a
   simulator-only luxury.
3. **The freshest information was being averaged away.** With the default settings, the newest plan —
   the only one that saw the current skin reading — contributed about 1.6% of the actual movement, with
   information about 41 steps stale on average. One flag fixes it: `--temp_agg_off`.
4. **One reading, 100 steps.** A single skin snapshot conditions a 100-step plan under equal weighting.
   A "something is 20 cm away" reading is only relevant to the first few steps, so its influence is
   mathematically forced to be tiny.

**The go/no-go check before spending more GPU time.** Before retraining, we asked a simpler question
statistically: can you tell a swerving run from a straight run just by looking at the skin readings?

- *Version 1: no.* Not from the reflex features, not from the raw readings, not from anything — and
  **not from the joint positions either**. Cause proven: the scripted expert kept the bar at the same
  distance as the surrounding walls, so a swerve and a straight run produce statistically identical
  skin readings. Direct confirmation of cause #2 above.
- *Version 2: yes, swerving became detectable* (about 0.75 on a 0.5-is-chance scale), and it is not
  just an echo of the arm's pose. But a second check *failed*: you cannot tell from the skin whether
  the bar is present at all. So version 2 was a **judgment call, not a clean pass** — recorded as such
  at the time.

**Version 2 dataset:** 105 demonstrations. We asked for 200; 3 of 4 collection workers were killed by
the operating system for using too much memory, so three houses produced nothing. The data that landed
is healthy. Refilling means rerunning with at most 2 workers.

**Version 2 prediction error** (lower is better): raw skin **0.0595**, cameras only 0.0755, reflex
signal 0.0830. The raw skin tap predicts the expert's actions **21% better** than cameras alone. That
is the precondition the hidden-bar design was built to create, and it worked. Note also that the reflex
version is *worse* than cameras alone — adding a useless input to a fixed-size model costs something.

Caveat, and §4 is the reason to take it seriously: this is one training run, the versions differ
slightly in size, and prediction error does not reliably predict robot behavior. The driving tests in
§3 are the real evidence; this is supporting.

### Blurry-camera brains

Three camera-only brains trained on permanently blurred images (§4). Before spending GPU time we
verified the dataset was sound — no corrupt or missing data, images genuinely varying, no constant
values. **Passed.**

A limitation we flagged *before* running, which turned out to matter: at this image size, blur
saturates almost immediately. σ = 2 already removes about 98% of the fine detail, so σ = 2, 4, and 8
are nearly three points on the same plateau. σ = 1, 2, 4 would have given a real curve. σ = 2, 4, 8 was
kept at the user's call.

### Test-time blur switch (new, today)

A new option, `--eval_blur_sigma`, that blurs the camera images the robot sees **during the test**,
while leaving the skin and joint readings untouched. It degrades exactly one sense, which is the entire
point. It is verified to produce *numerically identical* blur to the training-time version, so
"train blurry at strength S, test blurry at strength S" is an exact comparison rather than an
approximate one. Default is off, so nothing changes unless you ask for it. The setting is written into
every result file, so results describe themselves.

Not yet run on anything. It is the only thing §8 was waiting for.

---

## 7. What to do next, in order of value per GPU-hour

**1. Make the collision counter tell hazard hits apart from wall brushing.** The biggest measurement
problem in the project, on the list since June, and now quantified: the background brushing rate is
about 60%, and the hazard bar itself adds only ~4–6 points on top. Every collision number in this
document is measured through that fog. It is what watered down the first experiment to nothing, and
what makes §4 unreadable.

The good news: the code already computes exactly what we need and throws it away. It works out the
identity of every body the robot touches, then stores only *how many*. The fix is about five lines —
keep the names, write them into the results. The bad news: it cannot be recovered from past runs, so
whatever needs the split has to be re-run. (There was a window to patch this while the blur grid was
running; the grid finished at 08:32, so that window is closed.)

**2. Put the blur result in the README** as the negative result it is, with the ±40-point noise floor
stated. In particular, do not leave the tidy training-error ladder in the README implying a tidy
behavioral effect — it doesn't have one.

**3. Update the submodule pointers.** *Correcting an error in the previous version of this file:* the
work in the `act` submodule **is** committed. What is stale is the parent repository's *pointer* to it.
A fresh clone of `main` would get a version without the blur features. Today's edit is additionally
uncommitted inside the submodule.

**4. Run the experiment in §8.** Nothing blocks it now.

**5. Repeat the main result with different random seeds.** The §3 result is one seed. §4 just showed
empirically how large single-seed swings can be here, which makes this more urgent than it looked last
week. Three seeds turns "promising" into "defensible". (The two situations aren't identical — §4
compares *different* brains, while §3 is a matched comparison with a working control — but the point
stands.)

**6. Force the brain to depend on the skin more.** Three untried levers: randomly hide the cameras
during training (the flags exist and have never been used), plan in shorter blind stretches (attacks
cause #4), and refill the dataset from 105 toward 200 demonstrations.

**7. Measure whether the brain actually looks at the skin.** The numbers that would answer this already
exist inside the model at test time and are discarded. This is the cheapest available direct answer to
the question the whole 2026-07-02 audit had to infer.

---

## 8. The research idea

### "The sense that still works when the cameras don't"

**What's missing from the story.** Right now we have one strong result: when the hazard is hidden from
the cameras, the skin cuts collisions by 26 points. The obvious objection is that "hidden from the
cameras" is an artificial condition we manufactured with a rendering trick. Real cameras see real bars.

The answer is already sitting in this repo, unused. **Real cameras fail in real ways.** We proved back
in June that skin readings are bit-for-bit identical through blur and near-darkness while cameras
collapse. The hidden bar is not a gimmick — it is the clean laboratory limit of an entirely ordinary
failure: fog, smoke, dim light, motion blur, a dirty lens, a bar the same color as the wall behind it.

So the hidden-bar result is one point on a curve nobody has measured. **Measure the curve.**

**The design, and why it is done at test time rather than training time.** §4 is the cautionary tale:
training one brain per corruption level gives one run per point, and the noise ate the effect whole.
The fix is to make the comparison **paired**. Take one finished brain, freeze it, keep the test
scenarios identical, and turn up the blur only at test time. Then run-to-run training variation cannot
enter the comparison at all — and it's cheaper, because nothing is retrained.

- **Brains:** the camera-only one and the raw-skin one. Plus the three blurry-trained brains as a
  control (train blurry, test at the same blur), which separates "information was lost" from "the test
  didn't look like training".
- **Corruption:** blur on the camera images at test time, strengths 0, 1, 2, 4. Dim lighting as an
  optional second axis. The skin and joint readings are untouched by construction.
- **Setups:** bar visible (the honest test — the bar *is* renderable, the cameras just can't resolve
  it), plus no-bar as the control. The hidden-bar setup is the infinite-blur endpoint and is already
  measured.
- **Numbers:** collisions, success, strict success, 50 attempts per point. Statistics from a script, not
  by eyeballing.

**What we expect, and why every outcome is informative.**

- *Camera-only brain:* success falls as blur rises — finding and grabbing the cup genuinely needs
  vision. Collisions stay flat and high, because vision was never doing avoidance anyway. §4 adds a
  wrinkle: expect success to survive further than intuition suggests, since blur-4 training didn't hurt
  it. This task's visual needs are coarse.
- *Skin brain:* success falls the same way (same visual need for the task), but **collisions stay low
  and flat**, because camera corruption doesn't touch the skin. The two collision curves separate and
  stay separated. **That separation is the figure.**
- *Blurry-trained control brains:* if training and testing at the same blur restores success, then blur
  is a recoverable handicap and the drop in the other brains splits cleanly into "lost information" and
  "mismatch". If it doesn't restore success, §4 already tells us why: those three brains are
  individually unreliable.

**Why this is a much stronger claim than what we have.** It turns one significant condition into a
**dose–response relationship with a mechanism**, and it answers the question anyone actually asks about
proximity skin — *why add a skin when you already have cameras?* — with the only answer that survives
scrutiny: **because cameras have a failure mode the skin structurally does not, and that failure mode is
not exotic.** The headline figure is two collision curves against corruption strength, one flat and low,
one flat and high, with the camera-only brain's success collapsing underneath both. The sensor-level
half of this argument was proven in June; this is the policy-level half, which has been sitting
half-finished for two months.

**Cost.** Code is done. The full version is 16 runs, about 47 hours. **The overnight version is 4 runs,
about 12 hours** — two brains × blur {0, 4} × the bar-visible setup at 50 attempts — enough to see
whether the two collision curves separate at all.

**Honest risks.**

- *Blur saturates fast.* σ = 2 already destroys 98% of fine detail. But §4 pushes the opposite way:
  blur-4 training didn't hurt success at all, so the interesting transition may be *above* σ = 4, not
  below σ = 2. Pilot two or three values cheaply just to find where the transition is before committing
  to 50 attempts everywhere.
- *Two things change at once.* Testing a sharp-trained brain on blurry images tests lost information
  *and* train/test mismatch together. The blurry-trained brains are exactly the control that separates
  them. Report both.
- *The collision counter is still blunt.* §7 item 1 is really a prerequisite for the clean version of
  this figure. Burying a bar-specific effect under a 60% any-contact floor is precisely what neutered
  the first experiment and what makes §4 unreadable.
- *50 attempts minimum.* §4 measured ±40 points of swing at 25. Don't sweep wide and cheap.
- *The task may not need much vision.* If success barely moves at any blur level, we lose one of the
  figure's two halves and the claim weakens to "the skin is unaffected" — true, but less interesting.
  Dim lighting is the hedge, since it destroys the coarse cue that blur preserves.

---

## 9. Traps — every one of these has already cost real time

- **Two argument parsers.** Deep inside the model-building code, the command line gets parsed a second
  time, strictly. So every new option added to the training script must *also* be declared as a
  do-nothing option in a second file, or the model refuses to build. The evaluation script is exempt
  because it shields that second parser — which is why today's new option needed no twin.
- **`--temp_agg_off` was broken until 2026-07-04.** The broken version asked the brain for a fresh plan
  every step and executed only its first move. Since that first move is nearly a copy of where the arm
  already is (the cheapest thing for the training loss to learn), the arm creeps forward, freezes about
  30 cm short, and holds there until the clock runs out. Symptom: 0 successes with *low* collisions, for
  every brain, in every setup. **Any number using this flag from before 2026-07-04 is invalid.**
- **A cosmetic setting causes out-of-memory crashes.** The environment defaults to rendering all 40
  distance sensors in full color, every step, purely for pretty video. The camera-only brain never even
  reads those images, but the pipeline holds every frame in memory until the scene finishes saving — 3
  GB per attempt, which gets the process killed. The evaluation config forces it off; memory then grows
  a manageable 0.5 GB per attempt.
- **Evaluation must run one at a time** on this machine (8 GB baseline plus 0.5 GB per attempt; a full
  run has hit 41 GB). Do not parallelize it blindly.
- **A graphics driver mismatch is invisible until the GPU is actually used.** After a driver package
  updates without a reboot, the loaded kernel driver and the userspace libraries disagree, and every GPU
  program fails at startup. There is now a preflight check.
- **Test the summary step of a long script before launching the long part.** The blur grid ran 13 hours
  and printed an empty table because of a file-matching pattern that had never been tried against a
  real folder name.
- **Sensor order.** The 40 sensors must be stacked in the reflex network's canonical order
  *everywhere*. That order puts one arm segment's sensors in the opposite order from the environment's
  own naming. One source of truth: the extractor's own order list.
- **Depth minimum range.** The simulator's depth minimum scales with scene size and, at room scale,
  silently deletes every reading closer than 20 cm — the whole band this project studies. Any new
  sensor wiring must set it small *and* re-check existing datasets.
- **Hiding something from the cameras hides it from the depth sensors too**, if you do it the obvious
  ways (transparency, or moving it to another display layer). Making it invisible to one and not the
  other requires separate display masks — which is exactly the mechanism the hidden-bar design exploits
  on purpose.
- **Small samples lie.** An early skin "win" at 25 attempts (24% vs 10%) evaporated at a matched 50, and
  §4 measured ±40 points of swing at 25. Always match sample sizes. 25 or fewer is noise here.
- **Training error does not predict robot behavior.** The blurry brains' training errors form a perfect
  ladder while their actual behavior is contradictory (§4). Never ship a behavioral claim supported only
  by training numbers.

---

## 10. Every number in one place

| Thing | Value | Where it came from |
|---|---|---|
| Skin sensors | 40, each an 8×8 depth image, 45° cone | `model_hybrid.xml` |
| Skin directional coverage | 83% of all directions, vs 10% for the wrist camera | sensor proofs, 06-11 |
| Skin accuracy | perfectly linear response; 5.6 mm error reconstructing a pipe | sensor proofs |
| Skin under blur / darkness | readings bit-for-bit identical; cameras collapse | sensor proofs |
| Reflex net direction accuracy (honest split) | 0.924 | audit, 06-13 |
| Reflex net error magnitude (honest split) | 69% worse than the cheating split | audit |
| Reflex net when obstacle is closest | outputs only 64% of the needed size | audit |
| Lookup-table baseline | ties the reflex net on direction (0.923) | audit |
| Camera-only baseline, best-success run | 40% success / 60% collisions (20 attempts) | June sweep |
| First skin attempt, 50 attempts | nothing beats cameras alone, p = 0.76 | 06-18 |
| Can you detect swerving from the skin? v1 | no — chance, from every feature | v1 probes |
| Can you detect swerving from the skin? v2 | yes, ~0.75 | v2 probes |
| Can you detect the bar's presence? v2 | no — chance. This check formally failed. | v2 probes |
| Dataset v2 | 105 demonstrations (47 visible / 49 hidden / 29 none) | `obstacle_prox_v2` |
| Prediction error: raw skin / cameras / reflex | **0.0595** / 0.0755 / 0.0830 | wandb |
| Cameras-only collisions (none / hidden / visible) | 60% / 66% / 64% | main grid, 50 each |
| Cameras-only success | 22% / 36% / 28% | main grid |
| **Raw-skin collisions** | 58% / **40%** / 50% | main grid |
| **The headline** | **26 points better, p = 0.016** | main grid |
| Reflex-signal version | no effect anywhere (p ≥ 0.67) | main grid |
| Background brushing rate (no bar present) | ~60% | main grid |
| How much the bar itself adds to collisions | only ~4–6 points | main grid |
| Blur training-error ladder (0 / 2 / 4 / 8) | 0.0755 / 0.0836 / 0.0948 / 0.1100 | 07-24 |
| Blur robot behavior | no pattern; noise dominates (§4) | 08-10 |
| **Measured noise at 25 attempts** | **±40 points** | §4 |
| Blur saturation | σ = 2 removes ~98% of fine detail | preview image |
| Test throughput | 3.56 min per attempt, measured over 225 | blur grid |
| Test memory | 8 GB baseline + 0.5 GB per attempt | measured |

---

## 11. Files and commands

Skip this section unless you are going to run something.

```
STATUS.md                       this file — the report
README.md                       the manual: method, math, setup, full recipes
train_blur_baseline.sh          trains the blurry brains (one per blur level)
eval_blur_baseline.sh           the 9-condition blur test + summary table
blur_eval.log                   log from the 2026-08-09/10 run (5.3 MB)

scripts/
  train_safety_cvae.py  safety_sweep.py    reflex network: data prep + training
  safety_{flinch,react,moving,orbit,sphere}_demo.py   the five demo videos
  convert_obstacle_to_act.py               demonstrations -> training files
  probe_prox_decodability.py                the go/no-go statistical check
  compare_pact.py                           statistics on two result folders
  figures.py                                all 29 paper figures
  build_hybrid_on_franka_skin.py            builds the skin arm model
  verify_hybrid_skin_sensors.py             re-run after ANY sensor change

submodules/act/                  the brain: training and testing
  imitate_episodes.py            training      (--use_proximity, --blur_sigma0, --blur_mode)
  eval_act_obstacle.py           testing       (--eval_cell, --temp_agg_off, --eval_blur_sigma)
  prox_cvae.py                   the frozen reflex network as a feature extractor
  attn_heatmap.py                attention maps (throws away the skin weights — §7 item 7)
  ckpts/obstacle_pact_v2/        6 trained brains: 3 main + 3 blurry

submodules/molmospaces/          the simulator and demonstration collection
  molmo_spaces/tasks/pick_task.py    where the §7 item 1 fix goes
  molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py

assets/robots/franka_skin/model_hybrid.xml   the 40-sensor arm
assets/safety/cvae_v3/                       the reflex network weights (and sensor order!)
act_style_data/obstacle_prox_v2/             the 105-demonstration dataset
eval_output/<run>_<setup>/                    videos + eval_summary.json per test
```

```bash
# --- collect demonstrations ---
conda activate mlspaces
cd submodules/molmospaces
python -m molmo_spaces.data_generation.main <ConfigName>     # num_workers <= 2 or it gets OOM-killed

# --- train one brain (from submodules/act) ---
PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python imitate_episodes.py \
    --task_name obstacle_pact_v2 --policy_class ACT --ckpt_dir ckpts --kl_weight 10 \
    --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 --batch_size 8 --lr 1e-5 \
    --seed 0 --num_epochs 2000 --wandb_run_name <name> \
    [--use_proximity --prox_feature raw] \
    [--blur_sigma0 4 --blur_mode constant]

# --- test one brain in one setup (clear images, correct settings) ---
PYTHONPATH="$PWD:$PYTHONPATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval_act_obstacle.py \
    --ckpt_dir ckpts/obstacle_pact_v2/<run> \
    --output_dir /home/jaydv/code/prox_learning/eval_output/<run>_<setup> \
    --num_rollouts 50 --chunk_size 100 --temp_agg_off --eval_cell <visible|invisible|free>

# --- same, but with the cameras blurred at test time (the §8 experiment) ---
    ... --eval_cell visible --eval_blur_sigma 4

# --- statistics on two result folders ---
python scripts/compare_pact.py <folder_a> <folder_b>
```

---

## 12. What we don't know

Written down so nobody has to rediscover it.

- **Whether the main result holds with a different random seed.** One seed, one dataset.
- **How much of any collision number is actually the hazard.** The counter can't tell hazard from wall,
  and the wall-brushing floor is ~60%. Everything in §3 and §4 is measured through this.
- **Whether the brain actually pays attention to the skin.** Inferred from indirect tests, never
  measured directly.
- **Why the skin brain collides *less* with a hidden bar present (40%) than with no bar at all (58%).**
  A bar being there appearing to *reduce* total contact is strange. Best guesses: the skin triggers a
  retreat that also happens to avoid the walls, or it's noise. Unresolved.
- **Whether the σ = 8 brain's 28% collision rate with no bar is real or a fluke.** §4 treats it as a
  fluke on the strength of the internal contradiction, but it was never re-run.
- **Whether this task needs sharp vision at all.** Blur-4 training says no. If that holds at test time,
  §8's figure loses one of its two halves.
- **Whether 105 demonstrations is the binding limit.** The diagnosis says data *design* matters more
  than data *volume*, but nobody ever collected the 200-demonstration version, so it's untested.
