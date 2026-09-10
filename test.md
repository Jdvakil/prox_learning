**One frozen eval recipe. Three ckpts. Three dirs.** Skin/vision is decided at **train**, not by eval flags. Do not lesion the raw net.

Leave wrist n=50. Do not redo raw n=50. Do not mix with hallway.

---

**Lock this eval (matches the 200 demos):**

- `eval_act_v1011d.py`
- n=50, `seed_base=2026`, house `i % 24`
- `--skin_substeps snapshot`
- **no** `--clutter_xy_scale` (full = 1)
- **no** `--cameras` (exo+wrist)
- `--save_video`
- defaults: `--skin egl`, `--history query`

That **is** `simple_v1011d_smoke_video/` for raw. Copy those flags. Change only `--ckpt_dir` and `--output_dir`.

Easy 0.25 is a **second table**. Raw already has it. Do not put ACT/readout on 0.25 until the full-randomize 3-way exists.

Wrist-only is a **third table**. Raw only, still running. Do not start ACT/readout wrist.

---

**Names. Do not rename done dirs.**

| Arm | Train | Ckpt | Eval dir | Status |
|---|---|---|---|---|
| `raw` | already | `submodules/act/ckpts/pact_pick_n_place_v2/20260903_171108_pact_pick_n_place_v2_v1011d_s0` | `eval_output/simple_v1011d_smoke_video/` | **done** 7/50 |
| `act` | `--run v1011d_act_s0 --arm act --seed 0` | `runs/pact/v1011d_act_s0` | `eval_output/v1011d_act_full_n50/` | train first |
| `readout` | `--run v1011d_readout_s0 --arm readout --seed 0` | `runs/pact/v1011d_readout_s0` | `eval_output/v1011d_readout_full_n50/` | train first |

Smoke dirs (n=2, wiring only): `eval_output/v1011d_act_full_smoke/`, `eval_output/v1011d_readout_full_smoke/`. Then n=50 in the `_n50` dirs. Resume skips finished eps; protocol mismatch refuses the dir.

Logs: `runs/pact_batch_logs/` **outside** the run folder.

When `completed: 50`, copy JSON to `reports/eval_summaries/<same_basename>.json`. Do not overwrite hallway JSON.

---

**Train (you run). Unique `--run`. Seed 0 to match raw.**

```bash
mkdir -p runs/pact_batch_logs
conda activate mlspaces
cd /home/jaydv/code/prox_learning
export OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export MLSPACES_ASSETS_DIR="$PWD/assets"

# cameras only
PYTHONUNBUFFERED=1 python scripts/pact.py train v1011d \
  --run v1011d_act_s0 --arm act --seed 0 \
  > runs/pact_batch_logs/v1011d_act_s0.log 2>&1

# full PACT (after ACT, or other GPU)
PYTHONUNBUFFERED=1 python scripts/pact.py train v1011d \
  --run v1011d_readout_s0 --arm readout --seed 0 \
  > runs/pact_batch_logs/v1011d_readout_s0.log 2>&1
```

Wait for `runs/pact/NAME/policy_best.ckpt` (+ `prox_encoder_best.pt` on readout). Do not start a second train on the GPU while wrist eval is live unless VRAM is clearly free.

---

**Eval look. Same command, two swaps.**

ACT smoke:

```bash
python eval_act_v1011d.py \
  --ckpt_dir runs/pact/v1011d_act_s0 \
  --num_rollouts 2 --skin_substeps snapshot --save_video \
  --output_dir eval_output/v1011d_act_full_smoke
```

ACT n=50 (after smoke JSON looks sane):

```bash
python eval_act_v1011d.py \
  --ckpt_dir runs/pact/v1011d_act_s0 \
  --num_rollouts 50 --skin_substeps snapshot --save_video \
  --output_dir eval_output/v1011d_act_full_n50
```

Readout: same, `--ckpt_dir runs/pact/v1011d_readout_s0`, dirs `v1011d_readout_full_smoke` then `v1011d_readout_full_n50`.

No extra encoder flag. Readout loads `prox_config.json` + `prox_encoder_best.pt` from that folder. ACT has no `prox_config.json` → skin off. That is the whole difference.

Sanity on load:
- ACT: no `PACT ckpt -> proximity ON`
- raw: `feature=raw`
- readout: `feature=surface_embedding` (or readout tap in that print)

Then cite only `completed: 50`.

---

**Later (not now).** Easy 0.25 3-way: add `--clutter_xy_scale 0.25`, dirs `v1011d_{act,readout}_easy025_n50`. Raw already in `simple_v1011d_easy025_n50/`. Separate table. Optimistic vs train.

---

**Do not.**
- Paste into `simple_v1011d_smoke_video/` or `simple_v1011d_easy025_n50/`
- `pact.py eval`
- `--cameras wrist_camera` on the 3-way
- `--history consecutive` on the 3-way (train mismatch is the frozen protocol)
- Retrain raw
- Mix 7/50 raw-full with 14/50 raw-easy with hallway 36/40
- Call a zeroed-skin raw ckpt “ACT”

Headline paper-style table later: **ACT vs raw vs readout, full randomize, exo+wrist, n=50.** Three rows. Same houses/seeds. That is surgical.


That file is the **current hallway surface encoder**, not a leftover ACT policy. It was trained 2026-08-25 on hallway native rows (`data/pact_place_corridor_v5`). Readout training **starts** there, then `--finetune_prox_encoder` updates it with ACT. Wrapper `pact.py train --arm readout` uses the **same** default, including for v12.

**Hallway readout:** do not retrain the compressor first. That `.pt` is the intended init. Joint finetune on hallway demos is the experiment.

**ACT / PACT-raw:** ignore this path. Those arms do not load it.

**New dumps (v12, v1011d, …):** you do **not** have to retrain the compressor. Current recipe: same hallway `.pt` as init, then finetune on the **new** demos. That is transfer, not a held-out encoder. README says changing the start encoder is a **new** experiment; put it in the run name. Do **not** init from a hallway policy’s `prox_encoder_best.pt` unless you mean that transfer.

Retrain the compressor on new data only if you want geometry pretraining **on that dump** (new clutter, new houses), or you do not want hallway scenes in the encoder. That is `python -m encoders.train --src <raw dump>` ([§4.4](README.md)), then `--prox_encoder_ckpt` / `--encoder-checkpoint` on that new `.pt`. It grades millimetre recon, not place rate.

Same 40-sensor, 8×8, min-pool skin. Cameras differ by task; the encoder never sees RGB.