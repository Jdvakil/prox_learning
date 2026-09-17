"""Latency and peak GPU memory of the exported seed-3103 ACT / PACT-128D bundles.

Per bundle, on the bundle's own example observation, batch size 1:

  encoder_forward     the proximity-encoder forward alone, on a steady-state
                      8-frame pooled window (PACT only; ACT has no encoder)
  fresh_policy_query  prepared tensors -> ACT transformer forward -> denormalised
                      action chunk on the CPU (one complete fresh model query)
  obs_to_action       the bundle's RobotPolicy.step() end to end: raw numpy
                      observation in, action dict out - image resize and
                      normalisation, proximity pooling and encoding, the model
                      forward, and temporal aggregation

plus GPU memory resident after load and the peak over the whole run. Every
timed section is bracketed by torch.cuda.synchronize, and each bundle runs in
its own subprocess so peak-memory numbers cannot bleed between models and the
bundles' runtime copies cannot shadow one another.

Usage (run inside the venv built from the bundles' requirements.txt):

  python scripts/bench_latency.py --compare ACT_DIR PACT_DIR \
      [--iters 1000] [--warmup 50] [--control_period_ms 50] [--json_out report.json]

The single-bundle form (--model_dir) is what the subprocesses invoke.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np


def _stats(seconds: list[float]) -> dict:
    ms = np.asarray(seconds, dtype=np.float64) * 1e3
    return {"mean_ms": round(float(ms.mean()), 4),
            "p50_ms": round(float(np.percentile(ms, 50)), 4),
            "p99_ms": round(float(np.percentile(ms, 99)), 4),
            "n": int(ms.size)}


def bench_one(model_dir: str, iters: int, warmup: int, device: str) -> dict:
    import torch

    spec = importlib.util.spec_from_file_location(
        "bundle_inference", Path(model_dir) / "inference.py")
    bundle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bundle)
    policy = bundle.RobotPolicy(model_dir=model_dir, device=device)
    dev = policy.device
    cuda = dev.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize(dev)

    weights_mb = torch.cuda.memory_allocated(dev) / 2**20 if cuda else float("nan")
    if cuda:
        torch.cuda.reset_peak_memory_stats(dev)

    with np.load(Path(model_dir) / "example_observation.npz", allow_pickle=False) as z:
        qpos = z["qpos"].copy()
        rgb = z["wrist_rgb"].copy()
        prox = z["proximity"].copy() if "proximity" in z.files else None

    # Prepared tensors, built exactly the way RobotPolicy.step() builds them.
    import cv2
    normalized_qpos = (qpos.astype(np.float32) - policy.stats["qpos_mean"]) / policy.stats["qpos_std"]
    state = torch.from_numpy(normalized_qpos).float().to(dev).unsqueeze(0)
    r = np.asarray(rgb)
    if r.dtype != np.uint8:
        r = (r * 255. if float(np.max(r)) <= 1. else r).astype(np.uint8)
    if r.shape[:2] != (240, 320):
        r = cv2.resize(r, (320, 240), interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(
        np.transpose(r.astype(np.float32) / 255., (2, 0, 1))[None, None]).to(dev)
    image_n = policy.normalize(image)

    tokens = None
    window_t = None
    encode = None
    if policy.encoder is not None:
        from encoders.pact import causal_pooled_window, encode_for_act
        from encoders.peak_closeness import stack_obs_proximity
        raw = dict(zip(policy.sensor_order, np.asarray(prox, dtype=np.float32)))
        pooled = stack_obs_proximity(raw, policy.sensor_order, pool="min")
        window = causal_pooled_window(np.stack([pooled] * 8), 7)  # steady state
        window_t = torch.from_numpy(window).to(dev).unsqueeze(0)
        encode = encode_for_act

    def timed(fn, n):
        out = []
        for _ in range(n):
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            out.append(time.perf_counter() - t0)
        return out

    latency = {}
    with torch.inference_mode():
        if encode is not None:
            for _ in range(warmup):
                encode(policy.encoder, window_t)
            latency["encoder_forward"] = _stats(
                timed(lambda: encode(policy.encoder, window_t), iters))
            tokens = encode(policy.encoder, window_t)

        def query():
            predicted, _, _ = policy.policy.model(
                state, image_n, None, proximity_positions=tokens)
            chunk = predicted.squeeze(0).cpu().numpy()
            return chunk * policy.stats["action_std"] + policy.stats["action_mean"]

        for _ in range(warmup):
            query()
        latency["fresh_policy_query"] = _stats(timed(query, iters))

    policy.reset()
    for _ in range(warmup):
        policy.step(qpos, rgb, prox)
    latency["obs_to_action"] = _stats(timed(lambda: policy.step(qpos, rgb, prox), iters))

    memory = {"weights_resident_mb": round(weights_mb, 1)}
    if cuda:
        memory["peak_inference_mb"] = round(torch.cuda.max_memory_allocated(dev) / 2**20, 1)
        memory["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved(dev) / 2**20, 1)

    return {"model": policy.config["model_name"],
            "model_dir": str(Path(model_dir).resolve()),
            "device": str(dev),
            "gpu": torch.cuda.get_device_name(dev) if cuda else "cpu",
            "torch": torch.__version__,
            "params_m": round(sum(p.numel() for p in policy.policy.parameters()) / 1e6, 2),
            "iters": iters, "warmup": warmup,
            "latency": latency, "memory_mb": memory}


def compare(act_dir: str, pact_dir: str, args) -> dict:
    runs = {}
    for tag, d in (("act", act_dir), ("pact", pact_dir)):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out = tmp.name
        subprocess.run(
            [sys.executable, __file__, "--model_dir", d, "--iters", str(args.iters),
             "--warmup", str(args.warmup), "--device", args.device, "--json_out", out],
            check=True)
        runs[tag] = json.loads(Path(out).read_text())
        Path(out).unlink()

    act_l, pact_l = runs["act"]["latency"], runs["pact"]["latency"]
    deltas = {}
    for key in ("fresh_policy_query", "obs_to_action"):
        deltas[key + "_mean_ms"] = round(pact_l[key]["mean_ms"] - act_l[key]["mean_ms"], 4)
        deltas[key + "_p99_ms"] = round(pact_l[key]["p99_ms"] - act_l[key]["p99_ms"], 4)
    if "encoder_forward" in pact_l:
        deltas["encoder_forward_mean_ms"] = pact_l["encoder_forward"]["mean_ms"]
    deltas["peak_inference_mb"] = round(
        runs["pact"]["memory_mb"].get("peak_inference_mb", float("nan"))
        - runs["act"]["memory_mb"].get("peak_inference_mb", float("nan")), 1)
    deltas["weights_resident_mb"] = round(
        runs["pact"]["memory_mb"]["weights_resident_mb"]
        - runs["act"]["memory_mb"]["weights_resident_mb"], 1)

    # Both bundles are configured with query_every_control_step: true, so the
    # observation-update rate is the control rate itself and the honest framing
    # is the added obs->action latency as a share of one control period.
    fractions = {}
    added = deltas["obs_to_action_mean_ms"]
    periods = ([args.control_period_ms] if args.control_period_ms
               else [1000. / hz for hz in (10., 20., 30., 50.)])
    for period in periods:
        fractions[f"{period:g}ms_period"] = f"{100. * added / period:.2f}%"

    return {"act": runs["act"], "pact": runs["pact"],
            "pact_minus_act": deltas,
            "added_obs_to_action_share_of_control_period": fractions}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare", nargs=2, metavar=("ACT_DIR", "PACT_DIR"))
    ap.add_argument("--model_dir")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--control_period_ms", type=float, default=None)
    ap.add_argument("--json_out")
    args = ap.parse_args()

    if bool(args.compare) == bool(args.model_dir):
        ap.error("exactly one of --compare or --model_dir is required")

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    if args.model_dir:
        report = bench_one(args.model_dir, args.iters, args.warmup, args.device)
    else:
        report = compare(args.compare[0], args.compare[1], args)

    text = json.dumps(report, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
