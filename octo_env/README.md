# octo_env — Octo with proximity as a new modality, in the fume-hood env

Implements the "follow Octo's format and append our data" idea: instead of
ACT, finetune **Octo** (octo-small-1.5) on our fume-hood episodes with the
proximity skin folded into the tokenized low-dim observation, using Octo's own
recipe for new observation/action spaces (`examples/02_finetune_new_observation_action.py`,
pinned commit `241fb3514b7c`).

**Design in one paragraph.** The RLDS builder emits each episode with
`image_primary` (exo), `image_wrist`, an 8-dim joint action, and
`state = qpos(9) ++ min-depth-per-sensor(40)`. Octo tokenizes `state` with its
stock `LowdimObsTokenizer`, so proximity rides the standard pipeline with zero
changes to Octo itself — exactly "our data in Octo's format". A second builder
config (`vision_only`, state = qpos only) gives the matched V-Octo baseline, so
the V vs V+P comparison from plan.md carries over unchanged. The action head is
replaced with an 8-dim L1 head at horizon 8 (short chunk = reactive, addressing
plan.md's chunking-responsiveness concern). Inference runs as a small HTTP
server in the JAX venv (same remote-policy pattern this stack already uses for
pi0), and the molmospaces-side client evaluates closed-loop in the fumehood_env
scenes — clutter, 27 hood sizes and deep reach all apply.

## Files

| File | What |
|---|---|
| `setup_octo.sh` | one-shot env setup + GPU/model smoke test |
| `rlds_builder/fumehood_proximity/` | TFDS/RLDS builder reading raw datagen h5+mp4; configs `with_proximity` / `vision_only` |
| `finetune_proximity.py` | example-02-style finetune: + proprio tokenizer, 8-dim L1 head |
| `octo_policy_server.py` | JAX-side HTTP action server (`/reset`, `/act`, `/info`) |
| `eval_octo_fumehood.py` | molmospaces-side closed-loop eval in the fumehood_env scenes |

## Run order (GPU machine)

```bash
# 0. once: octo env + smoke (downloads octo-small, verifies GPU)
bash octo_env/setup_octo.sh

# 1. build the datasets from a fume-hood collection (both configs)
export FUMEHOOD_RAW_DIR=/path/to/datagen/cluttered_fumehood_v1
cd octo_env/rlds_builder/fumehood_proximity
~/octo/.venv/bin/tfds build --config with_proximity
~/octo/.venv/bin/tfds build --config vision_only
cd -

# 2. finetune both policies (V+P and V), ~hours on one GPU
~/octo/.venv/bin/python octo_env/finetune_proximity.py \
    --data_dir ~/tensorflow_datasets --dataset_name fumehood_proximity/with_proximity \
    --save_dir ~/octo_runs/vp_octo
~/octo/.venv/bin/python octo_env/finetune_proximity.py \
    --data_dir ~/tensorflow_datasets --dataset_name fumehood_proximity/vision_only \
    --save_dir ~/octo_runs/v_octo

# 3. serve + evaluate closed-loop (two terminals)
~/octo/.venv/bin/python octo_env/octo_policy_server.py --checkpoint ~/octo_runs/vp_octo --port 8555
# molmospaces venv, PYTHONPATH=<prox_learning>:
python octo_env/eval_octo_fumehood.py --server http://127.0.0.1:8555 \
    --houses 1,313 --samples 2 --output_dir /tmp/octo_check
# vision-only pairing: serve v_octo on another port, add --no_proximity
```

**Plumbing-only smoke without any training**: run the server with
`--checkpoint hf://rail-berkeley/octo-small-1.5` and the eval with
`--no_proximity`. The pretrained model has no proprio tokenizer and a 7-dim
action head, so actions will be meaningless for our robot — it only proves the
obs→server→action loop end to end. Real behavior requires step 2.

## Verification status

Authored against octo `241fb3514b7c` (example 02, `LowdimObsTokenizer`,
`OctoModel.sample_actions` signatures read from source) and against the datagen
h5 layout verified earlier in this project (JSON-encoded qpos/action rows,
`(T,4,8,8)` proximity streams, per-episode MP4s). All files compile; **nothing
here has executed yet** — this machine cannot run the stack. Run order above is
also the debug order: each step fails independently and early.

Known first-run risks, in order of likelihood:
1. `tfds build` schema complaints (feature dtype/shape mismatches) — fix in the
   builder, it re-runs in minutes.
2. JAX/CUDA wheel mismatch on the GPU machine — use the `jax[cuda12]` line in
   `setup_octo.sh`, or match the machine's CUDA.
3. `make_single_dataset` kwargs drift if octo is not at the pinned commit.
4. Gripper: the L1 head regresses the raw 0/255 command; the client snaps at
   127.5. If gripper behavior is poor, binarize the action's last dim in the
   builder instead.

## Timeline fit (IROS workshop → ICRA)

The V vs V+P comparison, the visibility×clearance grid, and the paired-eval
statistics from plan.md are all policy-agnostic — swapping ACT for Octo changes
step 2 only. fumehood_env's 27 hood sizes remain the clearance axis; its
clutter remains the occlusion contributor.
