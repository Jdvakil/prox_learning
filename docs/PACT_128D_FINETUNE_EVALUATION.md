# Evaluating the seed-3103 PACT-128D fine-tuned checkpoint

Use [the portable inference adapter](../scripts/hf_pact3103_inference.py) to load the published checkpoint on another machine and integrate it with an evaluation loop. It loads the matching fine-tuned encoder, original normalization, and bundled model definitions. The original simulator workers are linked below for reproducing the recorded experiment.

These files are on branch **`experiment/pact-valid-ablation-followup-v1`**, not `main`.

## Download and verify the policy

The [PACT model repository](https://huggingface.co/Lundii/pact-128d-finetune-v1010c-seed3103) contains the exact seed-3103 final-update-60,000 policy and corresponding fine-tuned encoder. While private, downloading requires a Hugging Face account authorized to access it; GitHub access alone does not grant model access.

From a fresh checkout, using Python 3.11:

```bash
git clone --single-branch --branch experiment/pact-valid-ablation-followup-v1 \
  https://github.com/Jdvakil/prox_learning.git prox_learning_eval
cd prox_learning_eval
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install huggingface_hub
hf auth login
hf download Lundii/pact-128d-finetune-v1010c-seed3103 \
  --revision 8b207f202f735c6b973256a8012da1cc51a23a88 \
  --local-dir ./models/pact-128d-seed3103
python -m pip install -r ./models/pact-128d-seed3103/requirements.txt
python scripts/hf_pact3103_inference.py \
  --model-dir ./models/pact-128d-seed3103 --device cuda
```

The inference adapter uses `runtime/` inside the model bundle; this loading check does not require initializing the simulator submodules. Keep the entire bundle together, including `checksums.json`, `checkpoint_pairs.json`, `normalization.npz`, `model_config.json`, and `runtime/`.

The command verifies bundle hashes, strictly loads the full policy, checks the policy/encoder pairing, runs the included example observation, and checks the CUDA output against its retained reference. The printed result includes `checksums_verified` and `reference_matches_cuda_tolerance`. This is an inference check, not a complete simulated rollout. CPU mode is available with `--device cpu`, but it reports numerical differences without enforcing agreement with the CUDA reference.

The adapter was previously compared against the original controller over **105 input steps** with zero maximum action difference on the validated CUDA runtime, including eight-frame history and 100-chunk aggregation boundaries and episode reset. That evidence is in the bundle's `validation.json`; it is not an additional closed-loop evaluation result.

## Use in an evaluation loop

From the checkout root:

```python
from scripts.hf_pact3103_inference import RobotPolicy

policy = RobotPolicy("./models/pact-128d-seed3103", device="cuda")
policy.reset()  # Call once before every new episode.

# Repeat for every consecutive control step, using fresh observations:
command = policy.step(
    qpos=qpos,             # shape (9,): seven arm joints, two gripper joints
    wrist_rgb=wrist_rgb,   # HWC RGB, normally uint8; not BGR
    proximity=proximity,   # metres; shape (40, 4, 8, 8), or (40, 8, 8)
)
arm_targets = command["arm"]       # shape (7,), absolute joint targets
gripper_command = command["gripper"]  # shape (1,), 0=open or 255=close
```

Connect these commands to the existing simulator/controller and record its task/contact outcomes. `RobotPolicy.step` does not advance a simulator, infer task success, or audit contacts.

For proximity, a dictionary keyed by the 40 names in `policy.sensor_order` is also accepted. With arrays, use that exact order. In particular, **link-5 front precedes link-5 back** in this checkpoint. The adapter minimum-pools four subframes, maintains eight consecutive control frames, pads the initial history, and consumes **40 × 128-D CLS readouts**. Do not substitute the older frozen 32-D representation or an unpaired pretrained encoder.

| Setting | Source evaluation |
|---|---|
| Image | Wrist RGB, resized internally to 240 × 320 and ImageNet-normalized |
| Joint input | Nine original joint positions; normalization applied internally |
| Action chunk | 100 actions |
| Policy queries | Every consecutive control step |
| Temporal aggregation | Up to 100 chunks, weights proportional to `exp(-0.01 * age)` |
| Horizon | 900 actions; no early stop after success |
| Proximity | 40 sensors, 128-D readout, eight-frame causal history |
| Weights and statistics | Frozen for evaluation; no refitting |

Do not execute an entire 100-action chunk between adapter calls: the recorded comparison queries and updates history every control step. Both policy and encoder are placed in evaluation mode with gradients disabled.

## Original simulator evaluation code

These workers were already published in commit `e0d6a40ca402530c898dc2a0e845394ef6b9df0d` on the experiment branch:

| File | Purpose |
|---|---|
| [pact_v1010c_eval.py](../scripts/pact_v1010c_eval.py) | Original seed-3103 fine-tuned PACT rollout worker; `ReadoutInferencePolicy`, `run` |
| [pact_v1010c_core.py](../scripts/pact_v1010c_core.py) | Matched policy/encoder loading and encoder preprocessing helpers; `load_pair` |
| [pact_v1010c_multi_eval.py](../scripts/pact_v1010c_multi_eval.py) | Corresponding workers for training seeds 3104 and 3105 |
| [pact_v1010c_multi_core.py](../scripts/pact_v1010c_multi_core.py) | Seed-specific artifact selection through `PACT_FINETUNE_SEED` |
| [pact_wrist288_eval_worker.py](../scripts/pact_wrist288_eval_worker.py) | Shared simulator/controller and contact-audit integration |

The original seed-3103 worker's CLI is:

```bash
python scripts/pact_v1010c_eval.py --job /path/to/existing/job.json
```

This command requires an existing valid job, the original evaluation manifest and contract, checkpoint directory, retained reference artifacts, upstream source snapshot, and simulator environment/assets. It is not a standalone launcher that reconstructs those from the Hugging Face weights. Its paths are tied to the recorded experiment and its hash checks deliberately reject changed inputs. Use the portable adapter above for a new evaluation integration; use the original worker when the archived experiment dependencies are available.

When reproducing the recorded metrics, retain the original contact taxonomy: `hazard_bar`, `clutter`, `other_environment`, and `mounted_fixture` are forbidden; intended `grasp_target` and `place_receptacle` contact is allowed. Collision-free placement means final task success with no forbidden contact throughout the complete rollout. Compare policies on identical initial scenes and report task completion and contacts separately.

## ACT baseline and model identity

The same portable adapter loads [the seed-3103 ACT bundle](https://huggingface.co/Lundii/act-v1010-seed3103); omit proximity in its `step` call. Its pinned revision is `ccd21685374ea1a20f86632cbe2283603ab202fb`. Use that bundle's own configuration and normalization.

| PACT artifact | SHA-256 |
|---|---|
| Policy | `3d01957cd3d86bb95db13bb4b96db05ebe07794134e4f5f64b683205afd46dcd` |
| Fine-tuned encoder | `c1127d13cc195197c7375b5bb03081b65f9a44dabe477cb28d9408b5308e7cbc` |
| Original statistics pickle | `c15e9673e5c619ba1c86d54ab45d8a90843ec296ed768dfeb4e588808aaae107` |

Model weights remain in their existing Hugging Face repositories. This GitHub handoff contains inference code and instructions, not checkpoint files or additional evaluation results.
