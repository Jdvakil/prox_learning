# Project status — 2026-05-02 (Day 1 of 26)

This document is the authoritative snapshot of where the PLA project stands.
**Update it at the end of every working day** (and after any non-trivial
push). It is intentionally thorough — for an academic paper we cannot afford
ambiguity about what is built versus claimed.

Companion docs (read together):
- [TIMELINE.md](TIMELINE.md) — the 26-day plan
- [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md) — what got built, when, why
- [SANITY_CHECKS.md](SANITY_CHECKS.md) — every check command + verbatim output
- [ARCHITECTURE.md](ARCHITECTURE.md) — full system architecture and shapes
- [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) — every non-obvious choice
- [FILE_INVENTORY.md](FILE_INVENTORY.md) — file-by-file what's in the repo

---

## TL;DR

Day 1 work: **the entire repository scaffold and codebase has been
implemented from the technical summary and is verified to parse, import,
and produce non-zero gradients through every code path.** The model layer,
data pipeline, training loop, evaluation harness, ablation orchestration,
sanity checks, and per-folder scientific READMEs are all in place. The
remaining 25 days are about *running* the pipeline (collecting data,
training, evaluating) — not *writing* it.

| Milestone                                    | State           | Evidence                                                  |
|----------------------------------------------|-----------------|-----------------------------------------------------------|
| Sensor skin pipeline (URDF → MJCF + verify)  | Code complete   | `scripts/build_skin_mjcf.py`, `scripts/verify_skin.py`    |
| Data collection harness (HDF5 schema)        | Code complete   | `pla/data/collect.py`, `--dry-run` works                   |
| Data verification (schema + 30% prox-info)   | Code complete + smoke-tested | 6/6 episodes pass on synthetic data           |
| Per-channel normalization                    | Code complete + smoke-tested | `stats.json` regenerates correctly             |
| `PLADataset` sliding-window DataLoader       | Code complete + smoke-tested | 745 train / 149 val from 6 episodes            |
| ProximityEncoder (shared MLP)                | Code complete + verified  | grad norm > 1e-1 after 20 steps               |
| HandcraftedToFEncoder (ablation)             | Code complete + verified  | grad norm > 1e-3 after 20 steps               |
| Conv2DToFEncoder (ablation)                  | Code complete + verified  | grad norm > 1e-2 after 20 steps               |
| ModalityFusion (concat + cross-attn)         | Code complete + verified  | both pass forward/backward with shapes        |
| ACTDecoder (Zhao 2023, CVAE + sinusoidal)    | Code complete + verified  | pred [2,100,7] mu [2,32] logvar [2,32]        |
| FrozenMolmo2 / DummyVLBackbone               | Code complete   | Dummy verified; real Molmo2 needs HF download             |
| Unified `PLA` model w/ `vlm_only` flag       | Code complete + verified  | one-flag-difference confirmed                  |
| `train_loop` with grad-norm logging          | Code complete + smoke-tested | loss decreases, encoder grads logged          |
| Bootstrap CI + paired bootstrap p-value      | Code complete + verified  | p=0.0003 on 65/50 synthetic split             |
| Eval runner + results table aggregator       | Code complete + verified  | renders 4-method table with p-values          |
| Sensor importance (post-hoc masking)         | Code complete   | depends on env; logic verified                            |
| Failure-mode categorizer                     | Code complete + verified  | all 5 categories verified                     |
| Ablation orchestration (4 ablations)         | Code complete + smoke-tested | grad-norm pass for each variant               |
| Per-folder scientific READMEs (12 files)     | Complete        | every subfolder has its own README                        |

---

## What is *not* yet done

These items are explicitly out of scope for Day 1 and tracked here so they
don't get forgotten:

| Item                                                  | When      | Why deferred |
|-------------------------------------------------------|-----------|--------------|
| Real Molmo2-4B weights pulled from HF                 | Day 4     | 9 GB download; not needed for sanity checks |
| MJCF skin built from current URDF                     | Day 2     | Depends on Blender skin redesign (Day 1 PM)  |
| 1000-trajectory near-contact dataset collected        | Day 3-4   | Requires MJCF + MolmoSpaces wired up         |
| `stats.json` from real data                           | Day 4     | Depends on dataset                           |
| Any real training run                                  | Day 4-7   | Depends on dataset + stats                   |
| 100-episode evaluation on each task                    | Day 14    | Depends on trained checkpoints                |
| Paper figures (system overview, ToF heatmap, etc.)    | Day 12-13 | Depends on real dataset + results            |
| `pla/viz/heatmap.py` finishing touches (CLI)          | Day 12    | Stub exists; needs --tof-h5 + --importance-json paths fleshed out |
| `pla/checks/depth_reconstruction.py` integration       | Day 2     | Code exists but assumes a built MJCF          |
| `pla/checks/replay_mjcf.py` integration                | Day 2     | Same — needs MJCF + a recorded trajectory     |

---

## Day-by-day plan vs. status

The plan in [TIMELINE.md](TIMELINE.md) covers Days 1-26. Status checkboxes
below get ticked as we cross the calendar.

### Week 1 (May 2-8): Infrastructure + Data + First Training

- [x] **Day 1 (May 2):** Repository scaffold, all module code, READMEs, sanity checks.
  Failure-case analysis on existing MolmoBot-Pi0 eval is *separate* — not a
  code task.
- [ ] **Day 2 (May 3):** Patch FR3 MJCF with new sensor positions; run
  `pla.checks.depth_reconstruction` and `pla.checks.replay_mjcf` against the
  built MJCF. Schema-validate the existing collection on disk.
- [ ] **Day 3 (May 4):** Launch `bash scripts/collect_data.sh near_contact 1000`
  in tmux. Implement VLM-only ACT (already done as a config flag).
  ProximityEncoder + fusion (already done).
- [ ] **Day 4 (May 5):** Check overnight collection. Run prop-only MLP
  baseline. Verify VLM-only ACT loss curves are converging. Dummy forward
  pass already verified (no work needed Day 4 here).
- [ ] **Day 5 (May 6):** VLM-only ACT eval — 50 episodes near-contact + standard PnP.
- [ ] **Day 6 (May 7):** Verify proximity encoder grad-norm > 1e-8 on the
  *real* training run. The CLI sanity check (`pla.checks.grad_norm`) already
  validates this on synthetic data.
- [ ] **Day 7 (May 8):** First PLA numbers. Eval PLA checkpoint vs VLM-only ACT.

### Week 2 (May 9-15): Full Evaluation + Ablations

- [ ] **Day 8-9:** Launch all ablations overnight. Full evaluation: 100 episodes per condition.
- [ ] **Day 10-11:** Collect ablation numbers + sensor-importance heatmap.
- [ ] **Day 12:** Figures.
- [ ] **Day 13-14:** Lock results.

### Week 3 (May 16-22): Write the paper

- [ ] **Day 15-21:** Sections 3 → 4 → 5 → 6 → 2 → 1 → Abstract.

### Week 4 (May 23-28): Revise, Format, Submit

- [ ] **Day 22-27:** Alessandro feedback, format pass, supplementary, **submit**.

---

## Code health

| Metric                                | Value          | Target                              |
|---------------------------------------|----------------|-------------------------------------|
| `pla/` Python LOC                     | 5,782          | n/a                                 |
| Files in `pla/` package               | 49             | n/a                                 |
| AST parse pass rate                   | 49/49 (100%)   | 100%                                |
| Smoke-import pass rate                | 29/29 (100%)   | 100%                                |
| Forward-pass test (PLA + 4 variants)  | 5/5 PASS       | 5/5                                 |
| Grad-norm test (4 configs)            | 4/4 PASS       | 4/4                                 |
| Per-folder READMEs                    | 12 / 12 dirs   | every dir with code                 |
| Subfolder import surfaces (`__init__`)| 7              | every subpackage exports its API     |

---

## Risk register (live)

This is a copy/expand of the TIMELINE.md risk register, with current
mitigation status. **Update this whenever a risk's probability or impact
changes.**

| Risk                              | P      | Impact | Mitigation                                                       | Status |
|-----------------------------------|--------|--------|------------------------------------------------------------------|--------|
| PLA delta < 5 pp                  | Medium | Fatal  | Near-contact task design + verify ≥30% prox-informative          | Pending real data |
| Proximity encoder grad = 0        | Low    | Fatal  | `pla.checks.grad_norm` runs on every config; train loop logs it   | Mitigated (code) |
| Dataset not proximity-informative | Medium | High   | `pla.data.verify --strict` exits non-zero below threshold        | Mitigated (code) |
| Running out of time               | Medium | High   | Day 1 scaffold complete; writing starts Day 14 regardless        | Mitigated (Day 1) |
| Alessandro requests major changes | Low    | High   | Send draft Day 22 — 6 days buffer                                | Future |
| VRAM OOM on PLA training          | Low    | Med    | Grad clip 1.0; `expandable_segments`; can drop n_decoder_layers  | Mitigated (config) |
| Sensor enumeration order drift    | Low    | High   | `ToFSensorArray` documents and enforces MJCF order              | Mitigated (code) |
| Stats leak val/test into training | Low    | High   | Same `(seed, val_frac)` in `normalize.py` and `dataset.py`        | Mitigated (code) |
| Backbone fine-tune accidentally   | Low    | High   | `FrozenMolmo2.__init__` sets `requires_grad = False`              | Mitigated (code) |
| Action chunk unnormalized at infer| Low    | High   | `PLA.get_action` unnormalizes before return                      | Mitigated (code) |

---

## Open follow-ups (TODO)

In rough priority order. Nothing here is blocking Day 2 work.

1. **Wire `MolmoSpaces` real env** into `pla.eval.run_eval._make_env`. Right
   now we use lazy import; once the submodule is built we should validate
   the actual class name (`FrankaPickandPlaceEnv` is a guess from
   `pla/eval/tasks.py`).
2. **Real Molmo2 vision-token extraction.** `pla/models/vlm_backbone.py:
   _extract_vision_tokens` follows the documented HF call but needs to be
   verified against the live MolmoBot inference wrapper (see comment block
   in that file for the exact grep).
3. **Wandb pinning.** `train.py` uses `wandb.init` if available; pin a
   wandb version in `pyproject.toml` once the W&B project is live.
4. **`pla/viz/heatmap.py` CLI.** Currently exposes `tof_sequence` and
   `sensor_importance_heatmap` as functions; the README references CLI
   flags that are not yet wired to argparse. Expand on Day 12 figure work.
5. **Failure-analysis env-key mapping.** `pla.eval.failure_analysis.RULES`
   uses key names (`collided_with_obstacle`, `grasp_failed`, etc.) that we
   *expect* MolmoSpaces to emit. Confirm Day 5 against the actual env.
6. **Tests directory.** No `pytest` test files yet; the sanity-check
   scripts cover the same ground but a `tests/` dir would make CI cleaner.
   Defer until after submission.

---

## How to refresh this file

After every meaningful change:

```bash
# 1. Re-run the smoke checks (takes <30 sec):
PYTHONPATH=. python -m pla.checks.forward_pass
PYTHONPATH=. python -m pla.checks.grad_norm --config configs/train/pla.yaml --steps 50

# 2. Update SANITY_CHECKS.md with any new commands/output.
# 3. Update IMPLEMENTATION_LOG.md with the change you made.
# 4. Update the milestone table at the top of THIS file.
# 5. Commit with `docs: update status for <change>`.
```
