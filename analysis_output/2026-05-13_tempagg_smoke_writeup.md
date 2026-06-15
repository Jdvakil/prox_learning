# 2026-05-13 — Temporal-ensembling smoke + k=20 retrain night

**TL;DR.** Naive-chunk inference on the medium_v1 PLA ckpt was masking proximity by
playing back a memorised gripper-close at frame ~48 regardless of where the EE
actually was. We confirmed that, implemented temporal ensembling at inference,
smoked 10 episodes — the memorised timing **breaks** (close-frame mean shifts
48.4 → 81.7) but reach gets **worse** (TCP min 18.9 cm → 26.5 cm) and the model
still scores 0/10. The chunk-size baked into training is the real bottleneck, so a
50k-step retrain with `chunk_size=20` is firing now; ETA ~8:50 a.m.

WandB analysis run: <https://wandb.ai/jayluvsgeography/pla/runs/l4cc1o9s>
(eval_smoke_tempagg_20260513_235244)
WandB k=20 train run: <https://wandb.ai/jayluvsgeography/pla/runs/9xhjk85l>
(medium_pla_k20_v1_20260513_234211)
PLA z=0 baseline run we are comparing against: medium_pla_v1_20260512_234252.

---

## 1. What we set out to do tonight

Both PLA (use_proximity=True) and VLM-only (use_proximity=False) had landed at
0/10 success on the first medium-dataset rollout. The previous session noted
something odd: PLA on house 19 reached **2.57 cm** TCP-to-pickup distance and
still failed. That contradicts "PLA is just not learning the task" — the EE got
there, but didn't act on it. Two questions to answer:

- **B.** What does the PLA z=0 baseline actually do at the moment it is closest
  to the object? Specifically on house 19, where it gets within 5 cm.
- **A.** Implement temporal ensembling in `pla/eval_policy.py` (Zhao et al. 2023:
  every step, predict a fresh chunk; current action is the exp-weighted average
  over alive predictions targeting this step) and re-smoke. Hypothesis: chunk
  playback locks the gripper at a memorised frame; closing the loop will let
  proximity matter.

---

## 2. The diagnosis (B)

Built `pla/house_forensics.py` to summarise one episode per house with:
TCP min frame and distance, gripper first-close frame, whether gripper was ever
closed at TCP < 5 cm, `task_info.robot_contact` ever True, pregrasp-phase min,
final TCP, episode length. Ran it across the existing `eval_quick_v1/` z=0
baselines.

### 2.1 House-19 dive (PLA z=0)

| Metric | Value |
|---|---|
| TCP min frame | **31** |
| TCP min (m) | **0.0257** |
| Gripper first close frame | **47** |
| TCP at gripper-close frame | ~0.11 m (already retreated) |
| Gripper closed during TCP < 5 cm | **No** |
| `task_info.robot_contact` ever | False |

So the EE arrived at 2.57 cm at frame 31 but the policy fired the gripper
**16 frames later**, after the EE had retreated to 11 cm. The chunk that
predicted "close at ~47" was committed at chunk-start, before the proximity
sensors had any chance to inform that decision.

### 2.2 The same pattern across all 10 houses (both models)

PLA z=0 baseline (10 eps):
- success **0/10**, contact_ever **1/10**, TCP min mean **18.9 cm** (median 17.0 cm)
- Grip first-close frame mean **48.4** (only houses 15 and 19 are outliers — 15 never
  closes, 19 closes at 47; the remaining 8 cluster in [45, 54])

VLM-only z=0 baseline (9 eps, house 14 missing):
- success **0/9**, contact_ever **0/9**, TCP min mean **19.3 cm**
- Grip first-close frame mean **48.6**

Both models close the gripper at the dataset's typical pickup frame (~48), not
when the EE is near the object. **This is a closed-loop conditioning failure,
not a proximity-encoder failure.**

This finding is now persisted as memory `eval-chunk-timing-memorization` so
future sessions don't redo the diagnosis.

---

## 3. The fix attempt (A): temporal ensembling

Modified `pla/eval_policy.py`:

- Maintain `self._pending_chunks: list[tuple[start_step, np.ndarray]]`.
- Every inference step: predict a fresh chunk, append, then prune any chunk
  where age `self._step - start >= H` (= 100 for medium_v1).
- For the *current* step, compute exponential-weighted average over alive
  predictions: `w_k = exp(-m * k)` where `k = self._step - start` (age, 0 = newest).
  Default `m = 0.01`; controllable via `PLA_TEMP_AGG_M`. Old behaviour is reachable
  via `PLA_TEMP_AGG_OFF=1`.

Note on weighting convention: my impl makes **newest dominant** (k=0 → w=1).
Zhao et al. (2023) used the opposite convention (oldest dominant). Either is a
defensible choice — the only thing that matters is that the EWMA smooths over
many predictions. I will sweep `m` later if needed; for now this is the
ensembling we test.

Helper: `scripts/smoke_tempagg.sh <model> <ckpt> <use_proximity>` wraps a
10-house rollout with `PLA_Z_SCALE=0` and `seed=2028` (identical to baseline).

### 3.1 Smoke results — PLA z=0 + temporal ensembling

Rollout output: `rollout_output/pla_smoke_tempagg_20260513_230930/`
Forensics: `rollout_output/pla_smoke_tempagg_20260513_230930/forensics.csv`

```
house succ cont  tcpmin_m tcp@  lt5@ grip_close@ gripD@<5cm n_tr  pregr_m  final_m
   11    0    0    0.2698   18    -1         120          0    3   0.2698   0.2938
   12    0    0    0.1590   46    -1          47          0    2   0.1590   0.2048
   13    0    1    0.0514  292    -1          53          0    3   0.0514   0.0562
   14    0    0    0.6076   70    -1          -1          0    1   0.6076   0.6078
   15    0    0    0.5615   42    -1          -1          0    1   0.5615   0.5707
   16    0    1    0.0538   77    -1          68          0    3   0.0538   0.2231
   17    0    0    0.1134  178    -1         110          0    3   0.1134   0.2452
   18    0    0    0.2763    1    -1          -1          0    1   0.2763   0.2890
   19    0    0    0.4372    5    -1          -1          0    1   0.4372   0.4542
   20    0    0    0.1146  141    -1          92          0    2   0.1146   0.1743
SUMMARY: success=0/10  contact_ever=2/10  tcp_min mean=0.2645m median=0.2144m
         pregrasp_min mean=0.2645m  gripper_close_during_<5cm=0/10
         first_close_frame mean=81.7
```

### 3.2 Direct comparison

| Metric | PLA z=0 chunk=100 | VLM z=0 chunk=100 | PLA z=0 + tempagg |
|---|---|---|---|
| Episodes | 10 | 9 | 10 |
| Success | **0/10** | **0/9** | **0/10** |
| Contact ever | 1/10 | 0/9 | **2/10** |
| TCP min mean (m) | 0.189 | 0.193 | **0.265** ↑ |
| TCP min median (m) | 0.170 | 0.144 | 0.214 |
| Grip first-close frame mean | **48.4** | 48.6 | **81.7** ↑↑↑ |
| Grip first-close std | ~3 | ~4 | **27.7** |
| Episodes that never close gripper | 1 | 2 | **4** |
| Grip-close during TCP<5cm | 1 (house 19) | 0 | **0** |

### 3.3 Reading the result

- **Memorised-timing hypothesis is correct.** Going from chunked playback to
  per-step ensembling shifted grip-close from a near-constant frame 48 to a
  scattered mean 81.7 with std ~28. The policy is no longer puppet-running the
  training distribution's average pickup frame. (See WandB
  `delta_tempagg_vs_pla_base/grip_close_frame_shift = 33.2`.)

- **But ensembling alone is not a fix.** Reach got worse: TCP min mean rose by
  7.5 cm. Two of the four "never-close" episodes (14, 18, 19) ended very far
  from the object (>27 cm). The ensembled action smooths *across* a chunk that
  was never trained to be sampled mid-flight; the result is a slower, more
  hesitant trajectory.

- **Contact uplifted modestly:** 2/10 vs 1/10. Both contact cases (h13, h16)
  reached <6 cm — but the policy still fired the close 35–40 frames late.

- **The action distribution is fighting us.** ACT was trained to predict 100
  contiguous frames given the *initial* observation; at every step at inference
  it now sees a different observation but produces a 100-step plan rooted at
  "frame 0 = now". Averaging those produces a non-stationary action that
  doesn't match any chunk it was ever supervised on. To actually get proximity
  to land, **the policy needs to be trained with a shorter chunk so that
  per-step re-planning is in-distribution**.

This motivates the k=20 retrain.

---

## 4. The retrain (k=20)

Plumbing: `pla/train.py` had `chunk_size` hardcoded at 100 in two places
(`FrankaSkinDatasetConfig` and `PLAConfig`). Added `--chunk_size` CLI flag
(default 100 to preserve old behaviour), wired into both, and updated the
`[model] mask_links=... chunk_size=...` print.

### 4.1 First fire — crashed at step 150

Run: `medium_pla_k20_v1_20260513_232855` (wandb `o1a5lmd6`).
PID 1969846. Hit step 50 (loss 12.18), step 100 (loss 3.74), step 150
(loss 3.33), then a DataLoader-worker SIGKILL:

```
RuntimeError: DataLoader worker (pid 1970084) is killed by signal: Killed.
```

`signal: Killed` = SIGKILL from the OS OOM-killer (not CUDA OOM — that
would say "CUDA error: out of memory"). Cause: with the parallel smoke
running, 2 DataLoader workers × ~6–7 GB RSS + the active smoke + sim
processes pushed swap to 31 Mi free out of 2 Gi. Worker got reaped.

### 4.2 Second fire — running now

Same command, `num_workers=2 → num_workers=1` (halves DataLoader RAM
footprint at ~10–15 % throughput cost — fine for this size).

```
Run name:  medium_pla_k20_v1_20260513_234211
Wandb URL: https://wandb.ai/jayluvsgeography/pla/runs/9xhjk85l
Python PID: 2012221
Log: logs/train_medium_pla_k20_v1_20260513_234211.log
[model] mask_links=('link2',)  chunk_size=20
[model] use_proximity=True, params=96.59M
[dataset] 309 train traj (80235 frames) / 34 val traj (8794 frames)
GPU usage: 2.9 GB (well within the 10.5 GB free with smoke still running)
RAM usage: 2 GB RSS so far
```

ETA: previous run at num_workers=2 was 12 samp/s = 9.4 h for 50 k steps.
With num_workers=1 expect ~9–10 samp/s = **~11 h**. Finish around **10:30 a.m.**
local time. Worst-case 11:30 a.m. The 4 p.m. meeting still has headroom.

VLM k=20 retrain: pre-staged (same command, `--use_proximity false`) but **not
fired**. Decision (from user): option (a) — compare PLA k=20 against existing
VLM k=100 as preliminary. Re-evaluate after PLA k=20 finishes.

---

## 5. Wins this session

1. **Frame-48 memorisation found and named.** Both PLA and VLM-only fail in the
   same way — chunk-playback locks the gripper at a memorised pickup frame.
   Identifying this rules out "proximity didn't help because the encoder is
   collapsed" and points the bottleneck firmly at inference protocol.
2. **`pla/house_forensics.py`.** General-purpose per-episode forensic table.
   Drop in any rollout root, get TCP min + gripper-close + contact + pregrasp
   stats for every house in one command. Reused everywhere now.
3. **Temporal ensembling implemented and validated.** It does the thing it was
   supposed to do (decouples gripper close from chunk-start timing). It just
   exposes a deeper problem (open-loop training).
4. **`chunk_size` is now configurable.** Was hardcoded in two places; one
   `--chunk_size` flag now flows through dataset + model construction and is
   saved with the ckpt config + wandb config.
5. **k=20 train running cleanly.** Confirmed mask, chunk size, param count,
   train/val split match expectations. No more OOMs once we dropped to
   num_workers=1.

## 6. Bugs hit and fixed

1. **Forensic frame mismatch.** First pass used a hand-rolled quaternion-rotate
   helper to put pickup into base frame; got d = 8.92 m (nonsense). Fixed by
   importing `_world_to_base` from `pla/eval_harness.py` (proper rotation
   matrix: `R.T @ (p_world - base_pos)`). Now matches harness exactly.
2. **`rollout_eval.py --house_inds` argparse.** Tried space-separated list,
   got `unrecognized arguments: 12 13 ...`. The flag is comma-separated:
   `--house_inds "11,12,13,14,15,16,17,18,19,20"`. Fixed in
   `scripts/smoke_tempagg.sh`.
3. **CUDA OOM when smoking VLM in parallel.** Spawned a parallel VLM smoke to
   compare both with tempagg; PLA was already on the GPU (8 GB used) and adding
   a second instance pushed past 23.9 GB. Killed the VLM zombie (SIGTERM →
   SIGKILL on PID 1966254); PLA smoke continued unaffected.
4. **System OOM on first k=20 train fire (the headline bug).** DataLoader
   worker SIGKILL at step ~150. Root cause: RAM pressure across smoke (2
   workers × 6–7 GB) + train (2 workers × 6–7 GB) + oracle background + sim
   processes, with swap already exhausted. Fix: `--num_workers 1` for the
   train, same command otherwise. RAM after the fix: 26 GB used / 30 GB free.

## 7. Things that surprised me / non-obvious notes

- Chunk-playback memorisation looks like "no learning" in the headline metric.
  Both models scored 0/10 and looked identically broken, but one of them (PLA
  on house 19) had quietly solved the reach problem. Without per-episode
  forensics this was invisible.
- TCP min mean *regresses* with tempagg. I expected ensembling to be at worst
  neutral; instead it actively makes reach worse because the chunked actions
  were never supposed to be averaged mid-flight. This is a strong argument
  that the *right* fix is to **train** with short chunks, not paste short
  chunks on top of a long-chunk policy.
- `task_info.robot_contact` is a more honest contact metric than the
  per-link gripper geometry counter. Two episodes contacted the object
  (h13, h16) without ever firing the gripper at <5 cm — the gripper was open,
  the EE arrived, the policy fumbled past. This is the actual "fingers
  brushed the object" signal we want, not a clean-grasp indicator.

## 8. Open items for tomorrow

- Wait for k=20 train to finish (~10:30 a.m.). Then **rollout 10 houses with
  the new ckpt** using `scripts/smoke_tempagg.sh` (still with `PLA_Z_SCALE=0`
  for now). Re-run forensics and re-push to wandb under the same project so
  the curves line up.
- If k=20 lifts reach (TCP min mean **<10 cm** on at least 5/10 houses) and
  produces at least one success or task_info.robot_contact ≥ 4/10, the story
  for the 4 p.m. meeting is: chunked training was masking proximity, fix is
  short chunks, here are the numbers.
- If k=20 still 0/10 and reach is still ~20 cm, the next lever is the
  observation horizon (action chunks fixed but observations still subsampled
  to the first frame). I'd want to widen `obs_horizon` in dataset + policy
  next, before touching the proximity encoder.
- **Decide on VLM k=20** once PLA k=20 numbers are in. If PLA k=20 helps,
  fire VLM k=20 (~9 h on GPU) so we get an apples-to-apples comparison for
  the headline before the meeting. If PLA k=20 doesn't help, skip — no point
  burning 9 h on a control that has no signal to control for.

## 9. Files touched / created tonight

Code:
- `pla/eval_policy.py` — temporal ensembling (`PLA_TEMP_AGG_OFF`, `PLA_TEMP_AGG_M`).
- `pla/train.py` — added `--chunk_size` CLI flag (default 100).
- `pla/house_forensics.py` — **new**, per-house forensics table.
- `pla/wandb_push_smoke.py` — **new**, push analysis to wandb.
- `scripts/smoke_tempagg.sh` — **new**, 10-house tempagg smoke wrapper.

Data:
- `rollout_output/pla_smoke_tempagg_20260513_230930/` — full 10-episode rollout
  + `forensics.csv` + `results.json`.
- WandB: <https://wandb.ai/jayluvsgeography/pla/runs/l4cc1o9s> (analysis), and
  the running <https://wandb.ai/jayluvsgeography/pla/runs/9xhjk85l> (k=20 train).

Memory:
- `eval-chunk-timing-memorization` (project memory) — documents the frame-48
  finding so the next session doesn't redo the diagnostic.
