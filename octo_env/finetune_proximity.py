"""Finetune Octo on the fume-hood dataset with proximity in the observation.

Follows octo/examples/02_finetune_new_observation_action.py (pinned commit
241fb3514b7c): keep both camera tokenizers, add a LowdimObsTokenizer over
`state` (qpos + per-sensor min depth), and replace the action head with an
8-dim joint-space L1 head at a short horizon so the policy stays reactive.

Train the paired comparison by building both dataset configs and running this
twice:

    python octo_env/finetune_proximity.py \
        --pretrained_path hf://rail-berkeley/octo-small-1.5 \
        --data_dir ~/tensorflow_datasets --dataset_name fumehood_proximity/with_proximity \
        --save_dir ~/octo_runs/vp_octo

    python octo_env/finetune_proximity.py ... --dataset_name fumehood_proximity/vision_only \
        --save_dir ~/octo_runs/v_octo
"""
import os

# jax 0.4.20 on this stack rebuilds the CUDA graph for jit_train_step and can
# hit CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE mid-run, which takes the process down
# with a segfault. Command buffers buy little here and cost the whole run, so
# they are off unless the caller overrides XLA_FLAGS.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

from absl import app, flags, logging
import flax
import jax
import optax
import tensorflow as tf
import tqdm

from octo.data.dataset import make_single_dataset
from octo.model.components.action_heads import L1ActionHead
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import TrainState, freeze_weights, merge_params, process_text

FLAGS = flags.FLAGS
flags.DEFINE_string("pretrained_path", "hf://rail-berkeley/octo-small-1.5", "Octo checkpoint.")
flags.DEFINE_string("data_dir", None, "TFDS data dir holding fumehood_proximity.")
flags.DEFINE_string("dataset_name", "fumehood_proximity/with_proximity", "TFDS name/config.")
flags.DEFINE_string("save_dir", None, "Checkpoint output directory.")
flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_integer("steps", 20000, "Finetuning steps.")
flags.DEFINE_integer("action_horizon", 8, "Predicted chunk length (short = reactive).")
flags.DEFINE_bool("freeze_transformer", False, "Freeze pretrained transformer blocks.")


def main(_):
    assert FLAGS.batch_size % jax.device_count() == 0
    initialize_compilation_cache()
    tf.config.set_visible_devices([], "GPU")

    logging.info("Loading pretrained model...")
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    logging.info("Loading dataset %s...", FLAGS.dataset_name)
    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name=FLAGS.dataset_name,
            data_dir=FLAGS.data_dir,
            image_obs_keys={"primary": "image_primary", "wrist": "image_wrist"},
            proprio_obs_key="state",
            language_key="language_instruction",
        ),
        traj_transform_kwargs=dict(
            window_size=2,
            action_horizon=FLAGS.action_horizon,
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256), "wrist": (128, 128)},
        ),
        train=True,
    )
    train_data_iter = (
        dataset.repeat().unbatch().shuffle(10000).batch(FLAGS.batch_size).iterator()
    )

    text_processor = pretrained_model.text_processor

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)

    # keep both camera tokenizers; add proximity-bearing lowdim state; new action head
    config = pretrained_model.config
    config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
        LowdimObsTokenizer,
        n_bins=256,
        bin_type="normal",
        low=-2.0,
        high=2.0,
        obs_keys=["proprio"],
    )
    config["model"]["heads"]["action"] = ModuleSpec.create(
        L1ActionHead,
        action_horizon=FLAGS.action_horizon,
        action_dim=8,
        readout_key="readout_action",
    )

    logging.info("Re-instantiating model for the new observation/action space...")
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=True,
        dataset_statistics=dataset.dataset_statistics,
    )
    model = model.replace(params=merge_params(model.params, pretrained_model.params))
    del pretrained_model

    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)], [100]
    )
    tx = optax.adamw(learning_rate)
    frozen_keys = model.config["optimizer"]["frozen_keys"]
    if FLAGS.freeze_transformer:
        frozen_keys.append("BlockTransformer_0")
    tx = freeze_weights(tx, model.params, frozen_keys)
    train_state = TrainState.create(rng=jax.random.PRNGKey(1234), model=model, tx=tx)

    def loss_fn(params, batch, rng, train=True):
        bound = model.module.bind({"params": params}, rngs={"dropout": rng})
        embeddings = bound.octo_transformer(
            batch["observation"], batch["task"],
            batch["observation"]["timestep_pad_mask"], train=train)
        return bound.heads["action"].loss(
            embeddings, batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"], train=train)

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True)
        return state.apply_gradients(grads=grads, rng=rng), info

    logging.info("Finetuning for %d steps...", FLAGS.steps)
    for i in tqdm.tqdm(range(FLAGS.steps), dynamic_ncols=True):
        batch = next(train_data_iter)
        train_state, info = train_step(train_state, batch)
        if (i + 1) % 100 == 0:
            info = jax.device_get(info)
            logging.info("step %d: %s",
                         i + 1, flax.traverse_util.flatten_dict({"train": info}, sep="/"))
        if (i + 1) % 2000 == 0 and FLAGS.save_dir:
            train_state.model.save_pretrained(step=i + 1, checkpoint_path=FLAGS.save_dir)


if __name__ == "__main__":
    app.run(main)
