"""Smoke the public API: ``python -m encoders`` from the repo root."""
from __future__ import annotations

import torch

from encoders import list_encoders, load_encoder


def main() -> None:
    print("encoders/ — function names\n")
    for name, blurb in list_encoders().items():
        print(f"  {name}")
        print(f"    {blurb}\n")

    prox = torch.full((1, 40, 8, 8), 0.10)
    prox[0, 5, 2, 2] = 0.05

    raw = load_encoder("peak_closeness", device="cpu")
    raw_feat = raw.policy_features(prox)
    print(f"peak_closeness    {tuple(raw_feat.shape)}  sensor5={float(raw_feat[0, 5, 0]):.3f}")

    geom = load_encoder("nearest_surface", device="cpu")
    xyz = geom.policy_features(prox)
    print(f"nearest_surface   {tuple(xyz.shape)}  (untrained weights)")

    emb = load_encoder("surface_embedding", device="cpu")
    z = emb.policy_features(prox)
    print(f"surface_embedding {tuple(z.shape)}  (untrained weights)")


if __name__ == "__main__":
    main()
