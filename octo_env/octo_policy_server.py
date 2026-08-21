"""HTTP action server wrapping an Octo checkpoint (pretrained or finetuned).

Runs inside the Octo (JAX) virtualenv so the molmospaces (torch) environment
never has to import JAX — mirroring how the pi0 policies are already served
remotely in this stack. Protocol is deliberately tiny:

    POST /reset                       -> clears the observation history
    POST /act    body: npz{image_primary u8(256,256,3), image_wrist u8(128,128,3),
                          state f32(state_dim)}
                 reply: npz{action_chunk f32(horizon, action_dim)}
    GET  /info                        -> checkpoint metadata (json)

Run:
    python octo_env/octo_policy_server.py --checkpoint ~/octo_runs/vp_octo --port 8555
    # or --checkpoint hf://rail-berkeley/octo-small-1.5 for a plumbing-only smoke
"""
from __future__ import annotations

import argparse
import io
import json
from collections import deque

import jax
import numpy as np
from flask import Flask, request, send_file

from octo.model.octo_model import OctoModel

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--step", type=int, default=None, help="checkpoint step (finetuned dirs)")
p.add_argument("--port", type=int, default=8555)
p.add_argument("--instruction", default="pick up the red cup in the fume hood")
p.add_argument("--window", type=int, default=2)
args = p.parse_args()

print(f"[octo-server] loading {args.checkpoint} ...")
model = OctoModel.load_pretrained(args.checkpoint, step=args.step)
tasks = model.create_tasks(texts=[args.instruction])

# unnormalization stats: single-dataset finetunes store them flat; pretrained
# checkpoints nest them per dataset name.
stats = model.dataset_statistics
if "action" not in stats:
    name = sorted(stats.keys())[0]
    print(f"[octo-server] using dataset statistics from '{name}'")
    stats = stats[name]
action_stats = stats["action"]

history: deque = deque(maxlen=args.window)
rng = jax.random.PRNGKey(0)
app = Flask(__name__)


@app.post("/reset")
def reset():
    history.clear()
    return {"ok": True}


@app.get("/info")
def info():
    return json.dumps({
        "checkpoint": args.checkpoint,
        "window": args.window,
        "instruction": args.instruction,
        "tokenizers": sorted(model.config["model"]["observation_tokenizers"].keys()),
    })


@app.post("/act")
def act():
    global rng
    payload = np.load(io.BytesIO(request.get_data()), allow_pickle=False)
    frame = {
        "image_primary": payload["image_primary"],
        "image_wrist": payload["image_wrist"],
        "proprio": payload["state"].astype(np.float32),
    }
    history.append(frame)
    n = len(history)

    def stack(key, pad_shape):
        arrs = [h[key] for h in history]
        while len(arrs) < args.window:            # left-pad with the oldest frame
            arrs = [arrs[0]] + arrs
        return np.stack(arrs)[None]               # (1, window, ...)

    obs = {
        "image_primary": stack("image_primary", None),
        "image_wrist": stack("image_wrist", None),
        "proprio": stack("proprio", None),
        "timestep_pad_mask": np.array(
            [[i >= args.window - n for i in range(args.window)]], dtype=bool),
    }
    rng, key = jax.random.split(rng)
    actions = model.sample_actions(
        obs, tasks, unnormalization_statistics=action_stats, rng=key)
    chunk = np.asarray(actions[0])                # (horizon, action_dim)

    buf = io.BytesIO()
    np.savez_compressed(buf, action_chunk=chunk.astype(np.float32))
    buf.seek(0)
    return send_file(buf, mimetype="application/octet-stream")


if __name__ == "__main__":
    print(f"[octo-server] ready on :{args.port}  "
          f"tokenizers={sorted(model.config['model']['observation_tokenizers'].keys())}")
    app.run(host="0.0.0.0", port=args.port, threaded=False)
