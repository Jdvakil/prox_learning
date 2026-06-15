# Safety-CVAE — status report (data, training, evaluation)

A simple check of what we have so far, before moving on to the ACT model.
Bottom line up front: **the data is clean, the CVAE trains well, and it predicts the
retreat direction reliably. Good to proceed to ACT.** Details below.

---

## 1. Data collection

**Obstacle dataset** (`hybrid_obstacle_v1`, run `20260612_183855`) — red-cup pick with
0–2 hazard bars in the fumehood:

- 151 episodes, 122 successful (**80%**). One object (the red cup).
- Bars present in ~75% of episodes.
- The arm bows **~38 mm** around a bar vs ~5 mm with no bar → a clear avoidance signal is
  actually in the data.
- Clean: 40 sensors consistent, almost no dead pixels, no empty episodes.

**Training data for the CVAE** (`sweep_v3.h5`) — built from the dataset's real arm poses
with hazard bars planted near the skin:

- 15,000 samples drawn from 10,335 real arm poses.
- **63%** of samples have something close enough to produce a retreat label.
- Well balanced: 33% close, 37% far, 30% in-between.

**Verdict: data is clean and has the signal we need.**

---

## 2. CVAE training

- 60 epochs, ~2 min on the 4090. Smooth convergence (see `curves.png`).
- Logged to wandb (`prox-safety-cvae` / `cvae_v3`) and saved as plots next to the weights.

| metric | value | meaning |
|---|---|---|
| direction cosine (close) | **0.926** | head pushes the right way ~93% aligned with the correct retreat |
| val MSE (scaled) | 0.0091 | overall reconstruction error |
| quiet-when-clear | 0.031 | output is near zero when nothing is near |

**Verdict: training works and converges cleanly.**

---

## 3. Evaluation / inference

**Direction is good and it generalizes (not memorizing).**
On a clean split where no arm pose is shared between train and test, the direction cosine
stays **0.924** — essentially the same as 0.926. So it learned the mapping, it is not just
recalling poses it saw.

**It beats simple baselines** (direction cosine, higher = better):

| predictor | cosine |
|---|---|
| guess zero | 0.00 |
| guess the average | 0.57 |
| linear map | 0.85 |
| **CVAE** | **0.926** |

**It uses the skin shape, not just "something is near."**
If we scramble which sensor is which, the direction cosine drops to 0.63 — so the head
really reads *where* the contact is, not just that something is close.

**Magnitude is a little soft.**
The push points the right way, but its size is about **0.84×** the target (softer on the
largest retreats). Not a blocker — easy to rescale or tune later if needed.

**Quiet when clear works** (≈0 output with nothing near), and it is **robust to small
depth noise**. It gets weaker if many sensors drop out at once (expected — not trained for
that).

**Where the signal comes from:** mostly the **wrist sensors (link5/link6)** and the first
four joints — because that is where the bars showed up in this dataset. Joint 7 is never
used (its label is always zero).

**Inference is simple:** `SafetyHead.load("assets/safety/cvae_v3")(skin)` → 7 joint
deltas. Deterministic, fast, skin-only (no scene info needed). The four demo videos confirm
it reacts live to a moving obstacle.

**Verdict: the CVAE predicts the retreat direction reliably and generalizes. Good to go.**

---

## Things to keep in mind for later (not blockers)

- Magnitude runs a bit soft, especially on the biggest retreats — rescale if a tighter
  margin is needed.
- Signal is concentrated at the wrist because that is where the bars were; if you want
  whole-arm coverage, collect obstacles near the upper arm too.
- All numbers so far are in simulation against the analytic retreat label. A closed-loop
  test (drive the head live and measure clearance) is the natural next eval once ACT is in.
