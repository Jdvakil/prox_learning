#!/usr/bin/env python3
"""Environment for Hugging Face ``data/v1011d`` (200 accepted episodes).

This is ``pact_place_corridor_v10_11d_randomized_clutter`` driven by
``PactPlaceCorridorV1011DRandomizedLayoutSampler``, not the four-object sampler
named in the published folder README.
"""

from __future__ import annotations

from ..registry import HF_V1011D as SPEC

__all__ = ["SPEC"]
